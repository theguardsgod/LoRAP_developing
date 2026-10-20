"""Snapshot of model definitions from gat_nc_lsb.py."""
import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax, remove_self_loops
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
from torch.autograd.function import InplaceFunction, Function
from torch_geometric.nn import GCNConv, GINConv, GATConv
from torch_geometric.nn.inits import glorot, zeros
from tqdm import tqdm
import argparse
from quantize_function.u_quant_func_bit_debug import *
import pdb
from utils.quant_utils import analysis_bit

class qGATConv(MessagePassing):

    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0, bias=True, num_nodes=1, bit=4, all_positive=True, quant_fea=True, is_q=True):
        super(qGATConv, self).__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
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
            self.lin = QLinear(in_channels, heads * out_channels, num_nodes, bit=bit, all_positive=all_positive, quant_fea=quant_fea)
        else:
            self.lin = nn.Linear(in_channels, heads * out_channels)
        self.q_gat_attn = u_quant_weight(1, heads, bit=bit)
        self.q_gat_fea = u_quant_fea(dim_fea=num_nodes, bit=bit)
        self.q_gat_edge = u_quant_weight(1, heads, bit=bit)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, size=None, first_layer=True):
        fea = self.lin(x)
        if self.is_q:
            fea_q = self.q_gat_fea(fea)
        else:
            fea_q = fea
        if size is None and torch.is_tensor(fea_q):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=fea_q.size(0))
        return self.propagate(edge_index, size=size, x=fea_q)

    def message(self, edge_index_i, x_i, x_j, size_i):
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
        if self.is_q:
            alpha = alpha.T
            alpha = self.q_gat_edge(alpha)
            alpha = alpha.T
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

class GAT(torch.nn.Module):

    def __init__(self, hidden_units, bit, is_q=False, heads=4, drop_out=0, drop_attn=0):
        super().__init__()
        num_nodes = dataset.data.num_nodes
        self.is_q = is_q
        self.drop_out = drop_out
        self.conv1 = qGATConv(dataset.num_node_features, hidden_units, heads=heads, dropout=drop_attn, num_nodes=num_nodes, bit=bit, all_positive=False, quant_fea=False, is_q=is_q)
        self.conv2 = qGATConv(hidden_units * heads, dataset.num_classes, heads=1, dropout=drop_attn, num_nodes=num_nodes, bit=bit, all_positive=True, quant_fea=True, is_q=is_q)

    def forward(self, data):
        x, edge_index = (data.x, data.edge_index)
        x = F.dropout(x, p=self.drop_out, training=self.training)
        x = self.conv1(x, edge_index, first_layer=True)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_out, training=self.training)
        x = self.conv2(x, edge_index, first_layer=False)
        return F.softmax(x, dim=1)
