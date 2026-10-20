"""Snapshot of model definitions from lr_gat_mnist.py."""
import os
import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU, Identity, BatchNorm1d as BN
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, remove_self_loops, softmax
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset, Planetoid, GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
from torch.autograd.function import InplaceFunction
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_add_pool, TopKPooling, GATConv
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

class qGATConv(MessagePassing):

    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0, add_self_loops=True, bias=True, num_nodes=1, bit=4, all_positive=True, quant_fea=True, uniform=True, is_q=True, para_dict={'alpha_init': 0.01, 'alpha_std': 0.02, 'gama_init': 0.1, 'gama_std': 0.2}, init='norm'):
        super(qGATConv, self).__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.add_self_loops = add_self_loops
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.is_q = is_q
        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        if bias and concat:
            self.bias = torch.nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and (not concat):
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        if self.is_q:
            self.lin = QLinear(in_channels, heads * out_channels, num_nodes, bit=bit, all_positive=all_positive, quant_fea=quant_fea, uniform=uniform, para_dict=para_dict, init=init)
        else:
            self.lin = nn.Linear(in_channels, heads * out_channels)
        self.q_gat_attn = u_quant_weight(1, heads, bit=bit)
        self.q_gat_fea = u_quant_fea(dim_fea=num_nodes, bit=bit, uniform=uniform, gama_init=0.01, gama_std=0.01, init=init)
        self.q_gat_edge = u_quant_fea(dim_fea=num_nodes, bit=bit, uniform=uniform, gama_init=0.1, gama_std=0.01, init=init)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, bit_sum, size=None):
        if self.is_q:
            fea, _, bit_sum = self.lin(x, edge_index, bit_sum)
            fea_q = self.q_gat_fea(fea, edge_index)[0]
        else:
            fea = self.lin(x)
            fea_q = fea
        if size is None and torch.is_tensor(fea_q) and self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=fea_q.size(0))
        return (self.propagate(edge_index, size=size, x=fea_q), bit_sum)

    def message(self, edge_index, edge_index_i, x_i, x_j, size_i):
        x_j = x_j.view(-1, self.heads, self.out_channels)
        att = self.att
        if self.is_q:
            att = self.att.squeeze(0)
            att = self.q_gat_attn(att)
            att = att.unsqueeze(0)
        if x_i is None:
            alpha = (x_j * att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * att).sum(dim=-1)
        alpha = self.q_gat_edge(alpha, edge_index)[0]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(src=alpha, index=edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)

class GATlayer(torch.nn.Module):

    def __init__(self, in_dim, out_dim, activation, batch_norm, residual=True, bit=4, all_positive=False, para_dict={'alpha_init': 0.01, 'alpha_std': 0.01, 'gama_init': 0.01, 'gama_std': 0.01}, num_deg=200, add_self_loops=False, is_q=False, uniform=True, heads=1, dropout=0, init='norm'):
        super().__init__()
        self.residual = residual
        self.batch_norm = batch_norm
        if in_dim != out_dim * heads:
            self.residual = False
        self.bn = nn.BatchNorm1d(out_dim * heads)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.conv = qGATConv(in_dim, out_dim, heads=heads, bit=bit, all_positive=all_positive, add_self_loops=add_self_loops, num_nodes=num_deg, para_dict=para_dict, uniform=uniform, dropout=dropout, is_q=is_q, init=init)

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

class GAT(torch.nn.Module):

    def __init__(self, dataset, hidden_units, num_layers, dropout=0, batch_norm=True, residual=True, bit=4, add_self_loops=False, is_q=False, num_deg=200, uniform=False, heads=1, out_dim=152, init='norm'):
        super().__init__()
        self.is_q = is_q
        para_list = [[{'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}, {'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}, {'gama_init': 0.7, 'gama_std': 0.1}], [{'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}, {'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}, {'gama_init': 0.6, 'gama_std': 0.7}], [{'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}, {'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}, {'gama_init': 0.76, 'gama_std': 0.68}], [{'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}, {'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}, {'gama_init': 0.6, 'gama_std': 0.5}], [{'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}], [{'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}], [{'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}]]
        if is_q == True:
            self.embeding = QLinear(dataset[0].x.size()[1] + dataset[0].pos.size()[1], hidden_units * heads, num_deg, bit, para_dict=para_list[0][0], all_positive=False, quant_fea=True, uniform=uniform, init=init)
        else:
            self.embeding = nn.Linear(dataset[0].x.size()[1] + dataset[0].pos.size()[1], hidden_units * heads, bias=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GATlayer(hidden_units * heads, hidden_units, F.relu, dropout=0, batch_norm=batch_norm, residual=residual, bit=bit, all_positive=False, add_self_loops=add_self_loops, is_q=is_q, uniform=uniform, para_dict=para_list[0][0], num_deg=num_deg, heads=heads, init=init))
        self.convs.append(GATlayer(hidden_units * heads, out_dim, F.relu, dropout=0, batch_norm=batch_norm, residual=residual, bit=bit, all_positive=False, add_self_loops=add_self_loops, is_q=is_q, uniform=uniform, para_dict=para_list[0][0], num_deg=num_deg, heads=1, init=init))
        if is_q:
            self.lin1 = QLinear(out_dim, out_dim, num_deg, bit, para_dict=para_list[0][0], all_positive=False, quant_fea=True, uniform=uniform, init=init)
            self.lin2 = QLinear(out_dim, dataset.num_classes, num_deg, bit, para_dict=para_list[0][0], all_positive=True, quant_fea=True, uniform=uniform, init=init)
        else:
            self.lin1 = nn.Linear(out_dim, out_dim, bias=True)
            self.lin2 = nn.Linear(out_dim, dataset.num_classes, bias=True)
        self.L = 1

    def forward(self, data):
        x, pos, edge_index, edge_weight, batch = (data.x, data.pos, data.edge_index, data.edge_attr, data.batch)
        x = torch.cat((x, pos), dim=1)
        bit_sum = x.new_zeros(1)
        if self.is_q:
            x, _, bit_sum = self.embeding(x, edge_index, bit_sum)
        else:
            x = self.embeding(x)
        for conv in self.convs:
            x, bit_sum = conv(x, edge_index, bit_sum)
        x = global_mean_pool(x, batch)
        if self.is_q:
            x, _, bit_sum = self.lin1(x, edge_index, bit_sum)
            x = F.relu(x)
            x, _, bit_sum = self.lin2(x, edge_index, bit_sum)
        else:
            x = self.lin1(x)
            x = F.relu(x)
            x = self.lin2(x)
        return (x, bit_sum)
