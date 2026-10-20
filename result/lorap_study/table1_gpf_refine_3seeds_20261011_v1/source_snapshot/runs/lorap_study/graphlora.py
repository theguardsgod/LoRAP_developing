"""GraphLoRA-adapted for the Table V GIN joint-QAT protocol.

Parallel low-rank graph branches, supervised contrast and graph-local structure
regularization. This same-domain graph-classification adaptation has no source
graph, pretrained frozen encoder, feature projector, or SMMD alignment.
Official reference: AllminerLab/GraphLoRA@aa9ab3dec32cee143c27fdbf07806d33356690db.
"""
import hashlib

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import remove_self_loops, to_dense_adj, to_dense_batch

from .lora import build_baseline
from .models import optimizer_for, sum_messages


def is_adapter(name):
    return name.startswith('graphlora_')


class LowRankLinear(nn.Module):
    def __init__(self, width, rank, zero_output):
        super().__init__()
        self.down = nn.Parameter(torch.empty(rank, width))
        self.up = nn.Parameter(torch.empty(width, rank))
        nn.init.kaiming_normal_(self.down)
        if zero_output:
            nn.init.zeros_(self.up)
        else:
            nn.init.kaiming_normal_(self.up)

    def forward(self, x):
        return F.linear(F.linear(x, self.down), self.up)


class LowRankGIN(nn.Module):
    """A graph branch whose last factor starts at zero, with a live gradient.

    GIN needs a two-layer MLP, unlike the official GAT projection. Zeroing both
    layers or applying ReLU to its zero final output would kill its gradients.
    Hence only the final up factor is zero and the residual output is signed.
    """
    def __init__(self, width, rank):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.first = LowRankLinear(width, rank, False)
        self.norm = nn.BatchNorm1d(width)
        self.last = LowRankLinear(width, rank, True)

    def forward(self, quantized_nodes, edges):
        aggregate = (1 + self.eps) * quantized_nodes + sum_messages(quantized_nodes, edges)
        hidden = self.first(aggregate)
        # A one-node training batch has no variance estimate. Use saved stats.
        if self.training and hidden.size(0) == 1:
            hidden = F.batch_norm(hidden, self.norm.running_mean, self.norm.running_var,
                                  self.norm.weight, self.norm.bias, training=False, eps=self.norm.eps)
        else:
            hidden = self.norm(hidden)
        return self.last(F.relu(hidden))


class GraphLoRAAdaptedGIN(nn.Module):
    def __init__(self, base, seed, rank=2):
        super().__init__()
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError('GraphLoRA rank must be a positive integer')
        for name in ('embedding', 'convs', 'lin1', 'lin2', 'prompt_node', 'prompt_aggr'):
            setattr(self, name, getattr(base, name))
        self.graphlora_rank = rank
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 930_000)
            self.graphlora_branches = nn.ModuleList([
                LowRankGIN(self.embedding.out_features, rank) for _ in self.convs])

    def forward(self, data, diagnostics=None, node_mask=None, return_aux=False):
        x = self.prompt_node.add(torch.cat((data.x, data.pos), dim=1))
        x, _, budget = self.embedding(x, data.edge_index, x.new_zeros(1))
        edges, _ = remove_self_loops(data.edge_index)
        for conv, branch in zip(self.convs, self.graphlora_branches):
            # Both graph branches consume this single quantized message tensor.
            qx = conv.q_xw(x.T).T
            aggregate = (1 + conv.eps) * qx + sum_messages(qx, edges)
            base, budget = conv.mlp(aggregate, edges, budget)
            delta = branch(qx, edges)
            x = base + delta
        mixed_nodes = x
        pooled = global_mean_pool(x, data.batch)
        output, _, budget = self.lin1(pooled, data.edge_index, budget)
        output, _, budget = self.lin2(F.relu(output), data.edge_index, budget)
        if not return_aux:
            return output, budget.squeeze()
        # Training-only node probabilities share existing FP32 master head
        # weights: no added classifier, quantizer call, or budget accumulation.
        node_logits = F.linear(F.relu(F.linear(mixed_nodes, self.lin1.weight, self.lin1.bias)),
                               self.lin2.weight, self.lin2.bias)
        return output, budget.squeeze(), dict(base_graph=global_mean_pool(base, data.batch),
            adapted_graph=pooled, node_logits=node_logits)


def build_graphlora(config, variant, seed, data):
    base = build_baseline(config, 'no_prompt', seed, data)
    if variant == 'no_prompt':
        return base
    if variant != 'graphlora':
        raise ValueError(variant)
    if config['dataset'] != 'MNIST' or config['task'] != 'graph':
        raise ValueError('This declared GraphLoRA adaptation is for MNIST graph classification')
    return GraphLoRAAdaptedGIN(base, seed, config.get('graphlora_rank', 2))


def backbone_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith('prompt_') or is_adapter(name):
            continue
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def graphlora_optimizer(model, config):
    optimizer = optimizer_for(model, config)
    adapters = [p for name, p in model.named_parameters() if is_adapter(name) and p.requires_grad]
    if adapters:
        identifiers = {id(p) for p in adapters}
        for group in optimizer.param_groups:
            group['params'] = [p for p in group['params'] if id(p) not in identifiers]
        optimizer.param_groups[:] = [g for g in optimizer.param_groups if g['params']]
        optimizer.add_param_group(dict(params=adapters, lr=config['lr'], weight_decay=0., name='graphlora'))
    identifiers = [id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(identifiers) == len(set(identifiers))
    assert set(identifiers) == {id(p) for p in model.parameters() if p.requires_grad}
    return optimizer


def model_counts(model):
    named = list(model.named_parameters())
    adapters = [p for n, p in named if is_adapter(n)]
    return dict(total_parameters=sum(p.numel() for _, p in named),
        trainable_parameters=sum(p.numel() for _, p in named if p.requires_grad),
        adapter_parameters=sum(p.numel() for p in adapters),
        adapter_fp32_bytes=sum(p.numel()*p.element_size() for p in adapters),
        adapter_buffer_bytes=sum(b.numel()*b.element_size() for n, b in model.named_buffers() if is_adapter(n)),
        prompt_parameters=sum(p.numel() for n, p in named if n.startswith('prompt_')))


def contrastive_loss(left, right, labels=None, train_mask=None, tau=.5, supervised_weight=.2):
    """Stable symmetric official batched_gct_loss, with labels only where allowed.

    The Table V extension contrasts pooled backbone and combined embeddings,
    rather than normalizing an initially all-zero residual branch.
    """
    if tau <= 0 or not 0 <= supervised_weight <= 1:
        raise ValueError('Invalid contrastive temperature or supervised weight')
    if left.shape != right.shape or left.ndim != 2 or left.size(0) < 1:
        raise ValueError('Contrastive views must have the same nonempty [N,D] shape')
    n = left.size(0)
    eye = torch.eye(n, device=left.device, dtype=torch.bool)
    weights = eye.to(left.dtype)
    if labels is not None and supervised_weight:
        known = torch.ones(n, device=left.device, dtype=torch.bool) if train_mask is None else train_mask.bool()
        positive = (labels[:, None] == labels[None, :]) & known[:, None] & known[None, :] & ~eye
        weights = weights + supervised_weight * positive.to(left.dtype)
    log_weights = torch.where(weights > 0, weights.clamp_min(torch.finfo(left.dtype).tiny).log(), -torch.inf)
    a, b = F.normalize(left, dim=1), F.normalize(right, dim=1)
    def one_way(x, y):
        cross, within = (x @ y.T) / tau, (x @ x.T) / tau
        numerator = torch.logsumexp(cross + log_weights, dim=1)
        denominator = torch.logsumexp(torch.cat((cross, within.masked_fill(eye, -torch.inf)), dim=1), dim=1)
        return (denominator - numerator).mean()
    return .5 * (one_way(a, b) + one_way(b, a))


def structure_loss(node_logits, edges, batch, include_self=False):
    """Weighted adjacency reconstruction within each graph, excluding padding.

    Graph labels are never copied onto nodes. Superpixel adjacency supplies the
    structural target. Default excludes self pairs; include_self permits an
    exact numerical comparison with the official self-loop implementation.
    """
    probability, valid = to_dense_batch(node_logits.softmax(-1), batch)
    size = probability.size(1)
    adjacency = to_dense_adj(edges, batch, max_num_nodes=size).clamp_max(1)
    eligible = valid[:, :, None] & valid[:, None, :]
    if not include_self:
        eligible = eligible & ~torch.eye(size, device=batch.device, dtype=torch.bool)[None]
    count = eligible.sum((1, 2))
    positive = (adjacency * eligible).sum((1, 2))
    positive_weight = torch.where(positive > 0, (count - positive) / positive.clamp_min(1), 1.)
    weights = torch.where(adjacency.bool(), positive_weight[:, None, None], 1.)
    scores = probability @ probability.transpose(1, 2)
    losses = F.binary_cross_entropy_with_logits(scores, adjacency, reduction='none') * weights * eligible
    per_graph = losses.sum((1, 2)) / count.clamp_min(1)
    return per_graph.mean()


def training_objective(model, batch, config):
    if isinstance(model, GraphLoRAAdaptedGIN):
        logits, budget, aux = model(batch, return_aux=True)
        contrast = contrastive_loss(aux['base_graph'], aux['adapted_graph'], batch.y.reshape(-1),
            tau=config['contrast_tau'], supervised_weight=config['contrast_supervised_weight'])
        structural = structure_loss(aux['node_logits'], batch.edge_index, batch.batch)
    else:
        logits, budget = model(batch)
        contrast = structural = logits.new_zeros(())
    ce = F.cross_entropy(logits, batch.y.reshape(-1))
    storage = config['a_loss'] * F.relu(budget - config['a_storage']).square()
    loss = ce + storage + config['contrast_weight'] * contrast + config['structure_weight'] * structural
    return loss, logits, dict(loss=ce, contrast=contrast, structure=structural, storage=storage, budget=budget)
