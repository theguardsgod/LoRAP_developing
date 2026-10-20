"""Snapshot of model definitions from lr_gcn_zinc_bit.py."""
import os
import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU, Identity, BatchNorm1d as BN
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, remove_self_loops
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset, Planetoid, GNNBenchmarkDataset, ZINC
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
from torch.autograd.function import InplaceFunction
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_add_pool, TopKPooling
from tqdm import tqdm
import random
import numpy as np
from quantize_function.u_quant_gc_bit_debug import *
import argparse

class ResettableSequential(nn.Sequential):

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()

    def forward(self, input, edge_index):
        for model in self:
            input = model(input, edge_index)[0]
        return input

class MyGCNConv(MessagePassing):

    def __init__(self, in_channels, out_channels, bit, all_positive=False, add_self_loops=False, num_deg=1000, para_dict={'alpha_init': 0.01, 'alpha_std': 0.01, 'gama_init': 0.01, 'gama_std': 0.01}, is_q=False, uniform=True, init='norm'):
        super().__init__(aggr='add')
        self.self_loop = add_self_loops
        self.is_q = is_q
        if is_q:
            self.lin = QLinear(in_channels, out_channels, num_deg, bit, para_dict=para_dict, all_positive=all_positive, uniform=uniform, init=init)
        else:
            self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.q_xw = u_quant_xw(in_channels, out_channels, bit, alpha_init=0.01, alpha_std=0.01)
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        torch.nn.init.zeros_(self.bias)
        torch.nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x, edge_index, bit_sum):
        if self.self_loop:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        if self.is_q:
            x, _, bit_sum = self.lin(x, edge_index, bit_sum)
        else:
            x = self.lin(x)
            bit_sum = 0
        x = x.T
        x = self.q_xw(x)
        x = x.T
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = self.propagate(edge_index, x=x, norm=norm)
        out = out + self.bias
        return (out, bit_sum)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class GCNlayer(torch.nn.Module):

    def __init__(self, in_dim, out_dim, activation, dropout, batch_norm, residual=True, bit=4, all_positive=False, para_dict={'alpha_init': 0.01, 'alpha_std': 0.01, 'gama_init': 0.01, 'gama_std': 0.01}, num_deg=1000, add_self_loops=False, is_q=False, uniform=True, init='norm'):
        super().__init__()
        self.residual = residual
        self.batch_norm = batch_norm
        if in_dim != out_dim:
            self.residual = False
        self.bn = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.conv = MyGCNConv(in_dim, out_dim, bit=bit, all_positive=all_positive, add_self_loops=add_self_loops, is_q=is_q, num_deg=num_deg, para_dict=para_dict, uniform=uniform, init=init)

    def forward(self, x, edge_index, bit_sum):
        x_in = x.clone()
        x, bit_sum = self.conv(x, edge_index, bit_sum)
        if self.batch_norm:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        if self.residual:
            x = x + x_in
        x = self.dropout(x)
        return (x, bit_sum)

class GCN(torch.nn.Module):

    def __init__(self, dataset, hidden_units, num_layers, dropout=0, batch_norm=True, residual=True, bit=4, add_self_loops=False, is_q=False, num_deg=200, uniform=False, init='norm'):
        super().__init__()
        self.is_q = is_q
        para_list = [[{'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}, {'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}, {'gama_init': 0.7, 'gama_std': 0.1}], [{'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}, {'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}, {'gama_init': 0.6, 'gama_std': 0.7}], [{'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}, {'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}, {'gama_init': 0.76, 'gama_std': 0.68}], [{'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}, {'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}, {'gama_init': 0.6, 'gama_std': 0.5}], [{'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}], [{'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}], [{'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}]]
        if is_q == None:
            self.embeding = QLinear(dataset[0].x.size()[1] + dataset[0].pos.size()[1], hidden_units, num_deg, bit, para_dict=para_list[0][0], all_positive=False, quant_fea=False, uniform=uniform)
        else:
            self.embeding = nn.Embedding(28, hidden_units)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNlayer(hidden_units, hidden_units, F.relu, dropout=0, batch_norm=batch_norm, residual=residual, bit=bit, all_positive=False, add_self_loops=add_self_loops, is_q=is_q, uniform=uniform, para_dict=para_list[0][0], num_deg=num_deg, init=init))
        if is_q:
            self.lin1 = QLinear(hidden_units, hidden_units, num_deg, bit, para_dict=para_list[0][0], all_positive=False, quant_fea=True, uniform=uniform, init=init)
            self.lin2 = QLinear(hidden_units, 1, num_deg, bit, para_dict=para_list[0][0], all_positive=True, quant_fea=True, uniform=uniform, init=init)
        else:
            self.lin1 = nn.Linear(hidden_units, hidden_units, bias=True)
            self.lin2 = nn.Linear(hidden_units, 1, bias=True)
        self.L = 1

    def forward(self, data):
        x, edge_index, edge_weight, batch = (data.x, data.edge_index, data.edge_attr, data.batch)
        x = x.reshape(-1)
        x = self.embeding(x)
        bit_sum = x.new_zeros(1)
        for conv in self.convs:
            x, bit_sum = conv(x, edge_index, bit_sum)
        x = global_mean_pool(x, batch)
        if self.is_q:
            x, _, bit_sum = self.lin1(x, edge_index, bit_sum)
        else:
            x = self.lin1(x)
        x = F.relu(x)
        if self.is_q:
            x, _, bit_sum = self.lin2(x, edge_index, bit_sum)
        else:
            x = self.lin2(x)
        return (x, bit_sum)
