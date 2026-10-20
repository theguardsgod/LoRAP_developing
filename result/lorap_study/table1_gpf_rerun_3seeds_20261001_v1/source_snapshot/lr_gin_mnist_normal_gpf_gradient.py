#*************************************************************************
#   > Filename    : lr_gin_mnist_bit.py
#   > Description : GIN trained on MNIST
#*************************************************************************
from cmath import log
from operator import index
import os
from re import T
import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU, Identity, BatchNorm1d as BN
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree,remove_self_loops
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset,Planetoid,GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
from torch.autograd.function import InplaceFunction
from torch_geometric.nn import GCNConv,GINConv,global_mean_pool,global_add_pool
from tqdm import tqdm
import random
import numpy as np
from quantize_function.u_quant_normal_debug import *
from quantize_function.MessagePassing_gc_bit import GINConvMultiQuant
import argparse
from prompt import GPF, GPF_plus_gradient
from torch_geometric.data import Batch
def evaluate_prob_mask(data):
    return torch.bernoulli(data.prob_mask).to(torch.bool)
fea_nn = []
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

class ProbabilisticHighDegreeMask:
    def __init__(self, low_quantise_prob, high_quantise_prob, per_graph=True):
        self.low_prob = low_quantise_prob
        self.high_prob = high_quantise_prob
        self.per_graph = per_graph

    def _process_graph(self, graph):
        # Note that:
        # 1. The probability of being protected increases as the indegree increases
        # 2. All nodes with the same indegree have the same bernoulli p
        # 3. you can set this such that all nodes have some probability of being quantised

        n = graph.num_nodes
        # Step 1: Calculate the bin edges based on quantiles
        num_bins = 20
        sum_grad_p = graph.sum_grad_p.cpu()  # Ensure it's on the CPU
        bin_edges = torch.quantile(sum_grad_p, torch.linspace(0, 1, num_bins + 1))

        # Step 2: Assign each tensor value to a bin
        bin_assignments = torch.bucketize(sum_grad_p, bin_edges)

        # Group the tensor values by bin
        binned_values = {i: sum_grad_p[bin_assignments == i] for i in range(1, len(bin_edges))}

        # Step 3: Calculate the counts (assuming sum_grad_p is still a tensor of float values)
        counts = torch.bincount(bin_assignments, minlength=num_bins)

        # Step 4: Calculate the step size (based on your specific range of 0.1 to 0.2)
        step_size = (self.high_prob - self.low_prob) / n

        # Step 5: Convert the counts to probabilities
        tensor_ps = counts.float() * step_size  # Ensure counts are converted to float

        # Step 6: Cumulative sum of the tensor_ps array
        tensor_ps = torch.cumsum(tensor_ps, dim=0)

        # Step 7: Add low_prob to the cumulative sum
        tensor_ps += self.low_prob
        graph.prob_mask = tensor_ps[bin_assignments]

        return graph

    def __call__(self, data):
        if self.per_graph and isinstance(data, Batch):
            graphs = data.to_data_list()
            # 利用 ptr 来获取每个图的节点数量
            node_counts = (data.ptr[1:] - data.ptr[:-1]).tolist()  # [73, 82, ..., n]

            # 根据节点数量拆分 sum_grad_p
            split_sum_grad_p = torch.split(data.sum_grad_p, node_counts)
            # 给每个 graph 加上对应的 sum_grad_p
            for g, g_sum_grad_p in zip(graphs, split_sum_grad_p):
                g.sum_grad_p = g_sum_grad_p
            processed = []
            for g in graphs:
                g = self._process_graph(g)
                processed.append(g)
            return Batch.from_data_list(processed)
        else:
            return self._process_graph(data)


def layer_hook(module, inp, out):
    fea_nn.append(out)

class relu(nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self,x,edge_index,bit_sum):
        x[x<0] = 0
        return x,edge_index,bit_sum

class bn(nn.Module):
    def __init__(self,hidden_units):
        super().__init__()
        self.bn = nn.BatchNorm1d(hidden_units)
    def forward(self,x,edge_index,bit_sum):
        x = self.bn(x)
        return x,edge_index,bit_sum

def load_checkpoint(model, checkpoint):
    if checkpoint != 'No':
        print("loading checkpoint...")
        model_dict = model.state_dict()
        modelCheckpoint = torch.load(checkpoint)
        pretrained_dict = modelCheckpoint['state_dict']
        new_dict = {k: v for k, v in pretrained_dict.items() if ((k in model_dict.keys()))}
        model_dict.update(new_dict)
        print('Total : {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
        model.load_state_dict(model_dict)
        print("loaded finished!")
    return model

def paras_group(model):
    all_params = model.parameters()
    weight_paras=[]
    quant_paras_bit_weight = []
    quant_paras_bit_fea = []
    quant_paras_scale_weight = []
    quant_paras_scale_fea = []
    quant_paras_scale_xw = []
    quant_paras_bit_xw = []
    other_paras = []
    for name,para in model.named_parameters():
        if('quant' in name and 'bit' in name and 'weight' in name):
            quant_paras_bit_weight+=[para]
            # para.requires_grad = False
        elif('quant' in name and 'bit' in name and 'fea' in name):
            quant_paras_bit_fea+=[para]
        elif('quant' in name and 'bit' not in name and 'weight' in name):
            quant_paras_scale_weight+=[para]
            # para.requires_grad = False
        elif('quant' in name and 'bit' not in name and 'fea' in name):
            quant_paras_scale_fea+=[para]
        elif('xw'in name and 'q' in name and 'bit' not in name):
            quant_paras_scale_xw+=[para]
        elif('xw'in name and 'q' in name and 'bit' in name):
            quant_paras_bit_xw+=[para]
        elif('weight' in name and 'quant' not in name ):
            weight_paras+=[para]
    params_id = list(map(id,quant_paras_bit_fea))+list(map(id,quant_paras_bit_weight))+list(map(id,quant_paras_scale_weight))+list(map(id,quant_paras_scale_fea))+list(map(id,weight_paras))\
    +list(map(id,quant_paras_scale_xw))+list(map(id,quant_paras_bit_xw))
    other_paras = list(filter(lambda p: id(p) not in params_id, all_params))
    return weight_paras,quant_paras_bit_weight,quant_paras_bit_fea,quant_paras_scale_weight,quant_paras_scale_fea,quant_paras_scale_xw,quant_paras_bit_xw,other_paras

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
    #  torch.backends.cudnn.deterministic = True


class ResettableSequential(nn.Sequential):
    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, "reset_parameters"):
                child.reset_parameters()
    def forward(self,input,edge_index,bit_sum):
        for model in self:
            input,_,bit_sum = model(input,edge_index,bit_sum)
        return input,bit_sum 
        



class qGIN(nn.Module):
    def __init__(self, dataset, num_layers, hidden_units, bit, num_deg=1000, is_q=False,
                    uniform=False,init='norm'):
        super(qGIN, self).__init__()
        self.is_q = is_q
        gin_layer = GINConvMultiQuant
        self.bit = bit
        self.num_deg = num_deg
        para_list=[[{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'gama_init':0.01,'gama_std':0.1}],
                   [{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'gama_init':0.6,'gama_std':0.7}],
                   [{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'gama_init':0.76,'gama_std':0.68}],
                   [{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'gama_init':0.6,'gama_std':0.5}],
                   [{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1}],
                   [{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1}],
                   [{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1}]]
        if(is_q):
            self.embedding = QLinear(dataset[0].x.size()[1]+dataset[0].pos.size()[1],hidden_units, num_deg, bit,para_dict=para_list[0][0], all_positive=False, 
                                    quant_fea=True,
                                    uniform=uniform,init=init)
        else:
            self.embedding = nn.Linear(dataset[0].x.size()[1]+dataset[0].pos.size()[1],hidden_units)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            if(is_q):
                self.convs.append(
                    gin_layer(
                        ResettableSequential(
                            QLinear(hidden_units, hidden_units, num_deg,bit, para_dict=para_list[0][0],all_positive=False,
                                    uniform=uniform,init=init),
                            bn(hidden_units),
                            relu(),
                            QLinear(hidden_units, hidden_units, num_deg,bit, para_dict=para_list[0][1], all_positive=True,
                                    uniform=uniform,init=init),
                            bn(hidden_units),
                            relu(),
                        ),
                        train_eps=True,
                        in_features=num_deg, out_features=hidden_units, bit=bit, para_dict=para_list[0][2], uniform=uniform,quant_fea=True, 
                    )
                )
            else:
                self.convs.append(
                    GINConv(
                        nn.Sequential(
                            nn.Linear(hidden_units, hidden_units),
                            nn.BatchNorm1d(hidden_units),
                            nn.ReLU(),
                            nn.Linear(hidden_units, hidden_units),
                            nn.BatchNorm1d(hidden_units),
                            nn.ReLU(),
                        ),
                        train_eps=True,
                    )
                )
        self.lin = torch.nn.ModuleList()
        if(is_q):
            self.lin1 = QLinear(hidden_units, hidden_units, num_deg, bit, para_dict=para_list[-1][0], all_positive=False,
                                        uniform=uniform,init=init)
            self.lin2 = QLinear(hidden_units, dataset.num_classes, num_deg, bit, para_dict=para_list[-1][0], all_positive=True,
                                        uniform=uniform,init=init)
        else:
            self.lin1 = nn.Linear(hidden_units, hidden_units)
            self.lin2 = nn.Linear(hidden_units, dataset.num_classes)
        
        self.prompt = GPF_plus_gradient(dataset[0].x.size()[1]+dataset[0].pos.size()[1], 20)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prompt.reset_parameters()

    def forward(self, data):
        x, pos, edge_index, edge_weight, batch = data.x, data.pos, data.edge_index, data.edge_attr, data.batch
        x = torch.cat((x,pos),dim=1)
        x = self.prompt.add(x)
        bit_sum=x.new_zeros(1)
        if(self.is_q):
            x,_,bit_sum = self.embedding(x,edge_index,bit_sum)
        else:
            x = self.embeding(x)
        i = 1
        for conv in self.convs:
            x,bit_sum = conv(x,edge_index,bit_sum)
            i = i+1
        x = global_mean_pool(x, batch)
        x,_,bit_sum = self.lin1(x,edge_index,bit_sum)
        x = F.relu(x)
        x,_,bit_sum = self.lin2(x,edge_index,bit_sum)
        return x,bit_sum
    
    def forward_mask(self, data, mask):
        x, pos, edge_index, edge_weight, batch = data.x, data.pos, data.edge_index, data.edge_attr, data.batch
        x = torch.cat((x,pos),dim=1)
        x[mask] = self.prompt.add(x[mask])
        bit_sum=x.new_zeros(1)
        if(self.is_q):
            x,_,bit_sum = self.embedding(x,edge_index,bit_sum)
        else:
            x = self.embeding(x)
        i = 1
        for conv in self.convs:
            x,bit_sum = conv(x,edge_index,bit_sum)
            i = i+1
        x = global_mean_pool(x, batch)
        x,_,bit_sum = self.lin1(x,edge_index,bit_sum)
        x = F.relu(x)
        x,_,bit_sum = self.lin2(x,edge_index,bit_sum)
        return x,bit_sum

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer,loader,a_loss, a_storage=0.1, gradient_mask=None):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out,bit_sum = model(data)

        loss = F.cross_entropy(out, data.y.view(-1))
        loss.backward()

        # 获取 p 的梯度，形状为 [235, 128]，用于计算每个节点对 p_list 的贡献
        grad_p = model.prompt.node_grad_buffer
        # 同时也可以查看 weight 的梯度，形状为 [235, p_num]
        grad_weight = model.prompt.weight_grad_buffer
        # 计算每个节点对 self.p_list 的梯度贡献
        # 每个节点 i 对 p_list 的第 j 行的贡献为：weight[i, j] * grad_p[i, :]
        node_contrib = grad_weight.unsqueeze(2) * grad_p.unsqueeze(1)  # shape: [235, p_num, 128]
        sum_grad_p = torch.abs(grad_p).sum(dim=1)
        data.sum_grad_p = sum_grad_p
        data = gradient_mask(data)
        mask = evaluate_prob_mask(data)
        optimizer.zero_grad()

        out,bit_sum = model.forward_mask(data,mask)
        loss = F.cross_entropy(out, data.y.view(-1))
        loss.backward()

        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)[0].max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)[0]
        loss += F.cross_entropy(out, data.y.view(-1), reduction="sum").item()
    return loss / len(loader.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str,default='GIN')
    parser.add_argument('--gpu_id',type=int,default=0)
    parser.add_argument('--dataset_name',type=str,default='MNIST')
    parser.add_argument('--num_deg',type=int,default=1000)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hidden_units',type=int,default=110)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--bit',type=int,default=4)
    parser.add_argument('--max_epoch',type=int,default=200)
    parser.add_argument('--max_cycle',type=int,default=10)
    parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--weight_decay',type=float,default=0)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--a_loss',type=float,default=0.00005)
    parser.add_argument('--lr_quant_scale_fea',type=float,default=5e-3)
    parser.add_argument('--lr_quant_scale_xw',type=float,default=5e-3)
    parser.add_argument('--lr_quant_scale_weight',type=float,default=1e-3)
    parser.add_argument('--lr_quant_bit_fea',type=float,default=0.0001)
    parser.add_argument('--lr_quant_bit_weight',type=float,default=0.0001) 
    parser.add_argument('--lr_step_size',type=int, default=50)
    parser.add_argument('--lr_decay_factor',type=float,default=0.5)
    parser.add_argument('--lr_schedule_patience',type=int,default=5)
    parser.add_argument('--init',type=str,default='uniform')
    parser.add_argument('--prob_mask_low',type=float,default=0.5)
    parser.add_argument('--prob_mask_change',type=float,default=0.5)
    ###############################################################
    parser.add_argument('--resume',type=bool,default=True)
    parser.add_argument('--store_ckpt',type=bool,default=True)
    parser.add_argument('--uniform',type=bool,default=True)
    parser.add_argument('--use_norm_quant',type=bool,default=True)
    ###############################################################
    # The target memory size of nodes features
    parser.add_argument('--a_storage',type=float,default=0.1)
    # Path to results
    parser.add_argument('--result_folder',type=str,default='result')
    # Path to checkpoint
    parser.add_argument('--check_folder',type=str,default='checkpoint')
    # Path to dataset
    parser.add_argument('--path2dataset',type=str,default='/data/cyliu/dataset/gnn-mnist')
    args = parser.parse_args()
    ##############################################################
    model = args.model
    dataset_name = args.dataset_name
    num_layers = args.num_layers
    hidden_units=args.hidden_units
    bit=args.bit
    max_epoch = args.max_epoch
    resume = args.resume
    path2result = args.result_folder+'/'+args.model+'_normal_gpf_gradient'+dataset_name
    path2check = args.check_folder+'/'+args.model+'_normal_gpf_gradient'+dataset_name
    if not os.path.exists(path2result):  
        os.makedirs(path2result)
    if not os.path.exists(path2check):  
        os.makedirs(path2check)
    ##############################################################
    setup_seed(41)
    if(args.resume==True):
        file_name = path2result+'/'+args.model+'_'+str(hidden_units)+'_'+'_on_'+dataset_name+'_'+str(bit)+'bit-'+str(max_epoch)+'.txt'
        if not os.path.exists(file_name):
                with open(file_name, 'w') as f:
                    for key, value in vars(args).items():
                        f.write('%s:%s\n'%(key, value))
    train_dataset = GNNBenchmarkDataset(root=args.path2dataset,name=dataset_name,split="train")
    val_dataset = GNNBenchmarkDataset(root=args.path2dataset,name=dataset_name,split="val")
    test_dataset = GNNBenchmarkDataset(root=args.path2dataset,name=dataset_name,split="test")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DQ = {"prob_mask_low": args.prob_mask_low, "prob_mask_change": args.prob_mask_change}
    print(f"Generating ProbabilisticHighDegreeMask: {DQ}")
    gradient_mask = ProbabilisticHighDegreeMask(
        DQ["prob_mask_low"], min(DQ["prob_mask_low"] + DQ["prob_mask_change"], 1.0)
    )
    # writer = SummaryWriter(log_dir=path2log)
    accu = []
    max_acc = 0.85
    for i in range(args.max_cycle):
        print_max_acc=0
        model=qGIN(train_dataset, args.num_layers,hidden_units=args.hidden_units,bit=args.bit, is_q=True,
                    num_deg=args.num_deg,
                    uniform=args.uniform,
                    init=args.init).to(device)
        # Load the saved state_dict
        state_dict = torch.load('/home/cyliu/code/A-2Q/checkpoint/GIN_normal_MNIST/GIN_110_on_MNIST_4bit-200.pth.tar')  # Replace with your file path

        # Load the state_dict into the model
        model.load_state_dict(state_dict['state_dict'], strict=False)
        weight_paras,quant_paras_bit_weight, quant_paras_bit_fea, quant_paras_scale_weight, quant_paras_scale_fea, quant_paras_scale_xw, quant_paras_bit_xw, other_paras = paras_group(model)
            # quant_paras_bit.requires_grad = False
        optimizer = torch.optim.Adam([{'params':weight_paras}, 
                                    {'params':quant_paras_scale_weight,'lr':args.lr_quant_scale_weight,'weight_decay':0},
                                    {'params':quant_paras_scale_fea,'lr':args.lr_quant_scale_fea,'weight_decay':0},
                                    {'params':quant_paras_scale_xw,'lr':args.lr_quant_scale_xw,'weight_decay':0},
                                    # {'params':quant_paras_bit_weight,'lr':args.lr_quant_bit_weight,'weight_decay':0},
                                    {'params':quant_paras_bit_fea,'lr':args.lr_quant_bit_fea,'weight_decay':0}, 
                                    {'params':other_paras}],
                                    lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=args.lr_decay_factor,
                                                     patience=args.lr_schedule_patience,
                                                     verbose=True,threshold=0.0001)
        # if (os.path.exists(path2check)):
        #     model = load_checkpoint(model,path2check)
        for epoch in range(args.max_epoch):
            t = tqdm(epoch)
            train_loss = train(model,optimizer,train_loader,args.a_loss, args.a_storage, gradient_mask)
            val_loss = eval_loss(model,val_loader)
            acc = eval_acc(model,test_loader)
            scheduler.step(val_loss)
            t.set_postfix(
                            {
                                "Train_Loss": "{:05.3f}".format(train_loss),
                                "Val_Loss": "{:05.3f}".format(val_loss),
                                "Acc": "{:05.3f}".format(acc),
                                "Epoch":"{:05.1f}".format(epoch),
                            }
                        )
            accu.append(acc)
            if(acc>print_max_acc):
                print_max_acc = acc
            if(acc>=max_acc):
                # path = path2check
                path = path2check+'/'+args.model+'_'+str(hidden_units)+'_on_'+dataset_name+'_'+str(bit)+'bit-'+str(max_epoch)+'.pth.tar'
                # max_acc = acc
                torch.save({'state_dict': model.state_dict(), 'best_accu': acc, 'hidden_units':args.hidden_units, 'layers':
               args.num_layers}, path)
            if(args.resume==True):
                f = open(file_name,'a')
                f.write(str(acc))
                f.write('\n')
        print(print_max_acc)
        if(resume==True):
            f = open(file_name,'a')
            f.write('The max accu of the {} runs is:'.format(i))
            f.write(str(print_max_acc))
            f.write('\n')
    accu = torch.tensor(accu)
    accu = accu.view(args.max_cycle,args.max_epoch)
    _,indices = accu.max(dim=1)
    accu = accu[torch.arange(args.max_cycle, dtype=torch.long),indices]
    acc_mean = accu.mean()
    acc_std = accu.std()
    desc = "{:.3f} ± {:.3f}".format(acc_mean,acc_std)
    print("Result - {}".format(desc))
    if(resume==True):
        f = open(file_name,'a')
        f.write('The result is:')
        f.write(desc)
        f.write('\n')
    print("finish")
    