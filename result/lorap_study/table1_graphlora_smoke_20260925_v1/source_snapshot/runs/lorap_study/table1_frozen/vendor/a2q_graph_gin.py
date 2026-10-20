"""Snapshot of model definitions from lr_gin_mnist_bit.py."""
from cmath import log
from operator import index
import os
from re import T
import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU, Identity, BatchNorm1d as BN
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, remove_self_loops
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset, Planetoid, GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
from torch.autograd.function import InplaceFunction
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_add_pool
from tqdm import tqdm
import random
import numpy as np
from quantize_function.u_quant_gc_bit_debug import *
from quantize_function.MessagePassing_gc_bit import GINConvMultiQuant
import argparse
from prompt import GPF, GPF_plus

class relu(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index, bit_sum):
        x[x < 0] = 0
        return (x, edge_index, bit_sum)

class bn(nn.Module):

    def __init__(self, hidden_units):
        super().__init__()
        self.bn = nn.BatchNorm1d(hidden_units)

    def forward(self, x, edge_index, bit_sum):
        x = self.bn(x)
        return (x, edge_index, bit_sum)

class ResettableSequential(nn.Sequential):

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()

    def forward(self, input, edge_index, bit_sum):
        for model in self:
            input, _, bit_sum = model(input, edge_index, bit_sum)
        return (input, bit_sum)

class qGIN(nn.Module):

    def __init__(self, dataset, num_layers, hidden_units, bit, num_deg=1000, is_q=False, uniform=False, init='norm', num_prompts=20):
        super(qGIN, self).__init__()
        self.is_q = is_q
        gin_layer = GINConvMultiQuant
        self.bit = bit
        self.num_deg = num_deg
        para_list = [[{'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}, {'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}, {'gama_init': 0.01, 'gama_std': 0.1}], [{'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}, {'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}, {'gama_init': 0.6, 'gama_std': 0.7}], [{'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}, {'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}, {'gama_init': 0.76, 'gama_std': 0.68}], [{'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}, {'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}, {'gama_init': 0.6, 'gama_std': 0.5}], [{'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}], [{'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}], [{'alpha_init': 0.01, 'gama_init': 0.01, 'alpha_std': 0.1, 'gama_std': 0.1}]]
        if is_q:
            self.embedding = QLinear(dataset[0].x.size()[1] + dataset[0].pos.size()[1], hidden_units, num_deg, bit, para_dict=para_list[0][0], all_positive=False, quant_fea=True, uniform=uniform, init=init)
        else:
            self.embedding = nn.Linear(dataset[0].x.size()[1] + dataset[0].pos.size()[1], hidden_units)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            if is_q:
                self.convs.append(gin_layer(ResettableSequential(QLinear(hidden_units, hidden_units, num_deg, bit, para_dict=para_list[0][0], all_positive=False, uniform=uniform, init=init), bn(hidden_units), relu(), QLinear(hidden_units, hidden_units, num_deg, bit, para_dict=para_list[0][1], all_positive=True, uniform=uniform, init=init), bn(hidden_units), relu()), train_eps=True, in_features=num_deg, out_features=hidden_units, bit=bit, para_dict=para_list[0][2], uniform=uniform, quant_fea=True))
            else:
                self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_units, hidden_units), nn.BatchNorm1d(hidden_units), nn.ReLU(), nn.Linear(hidden_units, hidden_units), nn.BatchNorm1d(hidden_units), nn.ReLU()), train_eps=True))
        self.lin = torch.nn.ModuleList()
        if is_q:
            self.lin1 = QLinear(hidden_units, hidden_units, num_deg, bit, para_dict=para_list[-1][0], all_positive=False, uniform=uniform, init=init)
            self.lin2 = QLinear(hidden_units, dataset.num_classes, num_deg, bit, para_dict=para_list[-1][0], all_positive=True, uniform=uniform, init=init)
        else:
            self.lin1 = nn.Linear(hidden_units, hidden_units)
            self.lin2 = nn.Linear(hidden_units, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, pos, edge_index, edge_weight, batch = (data.x, data.pos, data.edge_index, data.edge_attr, data.batch)
        x = torch.cat((x, pos), dim=1)
        bit_sum = x.new_zeros(1)
        if self.is_q:
            x, _, bit_sum = self.embedding(x, edge_index, bit_sum)
        else:
            x = self.embedding(x)
        i = 1
        for conv in self.convs:
            x, bit_sum = conv(x, edge_index, bit_sum)
            i = i + 1
        x = global_mean_pool(x, batch)
        x, _, bit_sum = self.lin1(x, edge_index, bit_sum)
        x = F.relu(x)
        x, _, bit_sum = self.lin2(x, edge_index, bit_sum)
        return (x, bit_sum)
