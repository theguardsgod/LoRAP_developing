"""Parallel graph adapters; native quantization and task heads are retained.

GIN/GCN reuse the actual message aggregate (including DQ's training mask).
GAT learns an independent low-rank attention branch on the features already
provided to native propagation. No extra observer/quantizer is evaluated.
"""
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch

from ..graphlora import contrastive_loss, structure_loss
from ..table1_frozen.backends import load_state, optimizer, predict
from ..table1_frozen.common import cpu_tree, rng_state, restore_rng
from ..table1_frozen.data import labels_and_output
from ..table1_frozen.deterministic import mean_pool, stable_softmax, sum_by_index
from ..table1_frozen.train import task_loss


class Factors(nn.Module):
    def __init__(self, inputs, outputs, rank, zero=True):
        super().__init__()
        self.down = nn.Parameter(torch.empty(rank, inputs))
        self.up = nn.Parameter(torch.empty(outputs, rank))
        nn.init.kaiming_normal_(self.down)
        nn.init.zeros_(self.up) if zero else nn.init.kaiming_normal_(self.up)

    def forward(self, x):
        return F.linear(F.linear(x, self.down), self.up)


class Branch(nn.Module):
    def __init__(self, architecture, inputs, outputs, rank, heads):
        super().__init__()
        self.architecture, self.heads = architecture, heads
        if architecture == 'GIN':
            self.eps = nn.Parameter(torch.zeros(1))
            self.first = Factors(inputs, outputs, rank, zero=False)
            self.norm = nn.BatchNorm1d(outputs)
            self.last = Factors(outputs, outputs, rank)
        else:
            self.projection = Factors(inputs, outputs, rank)
            if architecture == 'GAT':
                if outputs % heads:
                    raise ValueError('Incompatible attention output/head dimensions')
                self.attention = nn.Parameter(torch.empty(2, heads, outputs // heads))
                nn.init.xavier_uniform_(self.attention)

    def forward(self, x, aggregate, edges):
        if self.architecture == 'GIN':
            hidden = self.first((1 + self.eps) * x + aggregate.flatten(1))
            if self.training and len(hidden) == 1:
                hidden = F.batch_norm(hidden, self.norm.running_mean, self.norm.running_var,
                                      self.norm.weight, self.norm.bias, False, eps=self.norm.eps)
            else:
                hidden = self.norm(hidden)
            return self.last(F.relu(hidden))
        if self.architecture == 'GCN':
            return self.projection(aggregate.flatten(1))
        projected = self.projection(x).reshape(len(x), self.heads, -1)
        src, dst = edges
        score = ((projected[src] * self.attention[0]).sum(-1)
                 + (projected[dst] * self.attention[1]).sum(-1))
        attention = stable_softmax(F.leaky_relu(score, .2), dst, num_nodes=len(x))
        return sum_by_index(projected[src] * attention[..., None], dst, len(x)).flatten(1)


def sites(model):
    return [(name, layer) for name, layer in model.named_modules()
            if all(hasattr(layer, key) for key in ('propagate', 'aggregate', 'message'))]


class GraphLoRA(nn.Module):
    def __init__(self, backbone, config, seed, example, rank=2, frozen=False):
        super().__init__()
        if rank < 1:
            raise ValueError('rank must be positive')
        self.backbone, self.config, self.frozen = backbone, config, frozen
        self.graphlora = nn.ModuleList()
        self.handles, self.originals, self.states, self.dimensions = [], [], [], []
        self.last_views = None
        self.enabled = False
        for name, layer in sites(backbone):
            i = len(self.states)
            self.states.append({})
            self.dimensions.append(dict(name=name))
            native_prop, native_aggr = layer.propagate, layer.aggregate
            self.originals.extend([(layer, 'propagate', native_prop), (layer, 'aggregate', native_aggr)])

            def propagate(module, edge_index, *args, i=i, native=native_prop, **kwargs):
                self.states[i] = dict(x=kwargs['x'], edges=edge_index)
                return native(edge_index, *args, **kwargs)

            def aggregate(module, *args, i=i, native=native_aggr, **kwargs):
                value = native(*args, **kwargs)
                self.states[i]['aggregate'] = value
                return value

            def output(module, args, value, i=i):
                y = value[0] if isinstance(value, tuple) else value
                state = self.states[i]
                if not self.enabled:
                    self.dimensions[i].update(inputs=state['x'].size(-1), outputs=y.size(-1),
                                              heads=getattr(module, 'heads', 1))
                    return value
                delta = self.graphlora[i](state['x'], state['aggregate'], state['edges'])
                mixed = y + delta
                self.last_views = (y, mixed)
                return (mixed, *value[1:]) if isinstance(value, tuple) else mixed

            layer.propagate = MethodType(propagate, layer)
            layer.aggregate = MethodType(aggregate, layer)
            self.handles.append(layer.register_forward_hook(output))
            if config['architecture'] == 'GIN' and hasattr(layer, 'q_xw'):
                def quantized_root(module, args, value, i=i):
                    transposed = config['node'] or not isinstance(module, nn.Identity)
                    self.states[i]['x'] = value.T if transposed else value
                self.handles.append(layer.q_xw.register_forward_hook(quantized_root))
        if len(self.states) != config['layers']:
            raise ValueError(f"Expected {config['layers']} graph layers, found {len(self.states)}")
        # Infer changing node/head widths from training features only. Restore
        # ALL observer/BN state and RNG, so this probe is not extra calibration.
        state, rng, training = cpu_tree(backbone.state_dict()), rng_state(example.x.device), backbone.training
        try:
            with torch.no_grad():
                backbone.train()
                backbone(example)
        finally:
            load_state(backbone, state)
            restore_rng(rng, example.x.device)
            backbone.train(training)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 930000)
            for d in self.dimensions:
                self.graphlora.append(Branch(config['architecture'], d['inputs'], d['outputs'], rank, d['heads']))
        self.graphlora.to(example.x.device)
        self.enabled = True
        if frozen:
            backbone.requires_grad_(False)
        self.train(training)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, data):
        self.last_views = None
        return self.backbone(data)

    def close(self):
        for handle in self.handles:
            handle.remove()
        for module, name, original in reversed(self.originals):
            setattr(module, name, original)
        self.handles.clear()
        self.originals.clear()
        self.states.clear()
        self.last_views = None


def make_optimizer(model, c):
    if model.frozen:
        return torch.optim.Adam([dict(params=list(model.graphlora.parameters()), name='graphlora',
                                      lr=c['lr'], weight_decay=0.)])
    opt = optimizer(model.backbone, c)
    opt.add_param_group(dict(params=list(model.graphlora.parameters()), name='graphlora',
                            lr=c['lr'], weight_decay=0.))
    ids = [id(p) for group in opt.param_groups for p in group['params']]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters() if p.requires_grad}:
        raise AssertionError('Optimizer has omitted or duplicated parameters')
    return opt


def counts(model):
    return dict(total_parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                adapter_parameters=sum(p.numel() for p in model.graphlora.parameters()),
                adapter_fp32_bytes=sum(p.numel() * p.element_size() for p in model.graphlora.parameters()),
                adapter_buffer_bytes=sum(p.numel() * p.element_size() for p in model.graphlora.buffers()),
                prompt_parameters=0, sites=model.dimensions)


def regression_structure(nodes, edges, batch):
    # A one-dimensional regression head has constant softmax. Reconstruct
    # adjacency from normalized latent embeddings instead; no extra classifier.
    dense, valid = to_dense_batch(F.normalize(nodes, dim=-1), batch)
    n = dense.size(1)
    adjacency = to_dense_adj(edges, batch, max_num_nodes=n).clamp_max(1)
    eligible = valid[:, :, None] & valid[:, None, :] & ~torch.eye(n, device=nodes.device, dtype=torch.bool)[None]
    count = eligible.sum((1, 2))
    positive = (adjacency * eligible).sum((1, 2))
    pos_weight = torch.where(positive > 0, (count - positive) / positive.clamp_min(1), 1.)
    weights = torch.where(adjacency.bool(), pos_weight[:, None, None], 1.)
    loss = F.binary_cross_entropy_with_logits(dense @ dense.transpose(1, 2), adjacency, reduction='none')
    return ((loss * weights * eligible).sum((1, 2)) / count.clamp_min(1)).mean()


def objective(model, c, batch, hp):
    out, budget = predict(model, c, batch)
    logits, y = labels_and_output(c, batch, out, 'train')
    task = task_loss(c, logits, y)
    left, right = model.last_views
    if c['node']:
        mask = batch.train_mask
        contrast = contrastive_loss(left[mask], right[mask], batch.y[mask], tau=hp['tau'],
                                    supervised_weight=hp['supervised_weight'])
        structural = structure_loss(out, batch.edge_index, torch.zeros(len(out), dtype=torch.long, device=out.device))
    else:
        contrast = contrastive_loss(mean_pool(left, batch.batch), mean_pool(right, batch.batch),
                                    None if c['regression'] else batch.y.reshape(-1), tau=hp['tau'],
                                    supervised_weight=hp['supervised_weight'])
        if c['regression']:
            structural = regression_structure(right, batch.edge_index, batch.batch)
        else:
            head1, head2 = model.backbone.lin1, model.backbone.lin2
            node_logits = F.linear(F.relu(F.linear(right, head1.weight, head1.bias)), head2.weight, head2.bias)
            structural = structure_loss(node_logits, batch.edge_index, batch.batch)
    storage = c['a_loss'] * F.relu(budget - c['a_storage']).square() if not model.frozen else task.new_zeros(())
    loss = task + storage + hp['contrast_weight'] * contrast + hp['structure_weight'] * structural
    return loss, logits, y, dict(task=task, contrast=contrast, structure=structural, storage=storage)
