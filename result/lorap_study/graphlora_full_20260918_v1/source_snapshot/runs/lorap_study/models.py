"""GIN layouts and quantizers from the existing MNIST/CiteSeer experiments."""

import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter_add

from prompt import GPF_plus
from quantize_function.u_quant_func_bit_debug import QLinear as NodeLinear, u_quant_xw as NodeXW
from quantize_function.u_quant_gc_bit_debug import QLinear as GraphLinear, u_quant_xw as GraphXW
from .prompts import make_prompt, quantization_context, parameter_count

PARA = dict(alpha_init=0.01, alpha_std=0.1, gama_init=0.01, gama_std=0.1)


def sum_messages(x, edge_index):
    return scatter_add(x[edge_index[0]], edge_index[1], dim=0, dim_size=x.size(0))


class GraphMLP(nn.Module):
    def __init__(self, width, num_deg, bit):
        super().__init__()
        self.linear1 = graph_linear(width, width, num_deg, bit, False)
        self.bn1 = nn.BatchNorm1d(width)
        self.linear2 = graph_linear(width, width, num_deg, bit, True)
        self.bn2 = nn.BatchNorm1d(width)

    def forward(self, x, edges, budget):
        x, _, budget = self.linear1(x, edges, budget)
        x = F.relu(self.bn1(x))
        x, _, budget = self.linear2(x, edges, budget)
        return F.relu(self.bn2(x)), budget


def graph_linear(inputs, outputs, num_deg, bit, positive):
    layer = GraphLinear(inputs, outputs, num_deg, bit, para_dict=PARA,
                        all_positive=positive, uniform=True, init="uniform")
    layer.fea_quant.study_unsigned = positive
    return layer


def node_linear(inputs, outputs, num_nodes, bit):
    layer = NodeLinear(inputs, outputs, num_nodes, bit, para_dict=PARA,
                       all_positive=True, quant_fea=True)
    layer.fea_quant.study_unsigned = True
    return layer


class AggregationLayer(nn.Module):
    def __init__(self, mlp, width, bit, graph_level, quantize=True):
        super().__init__()
        self.mlp = mlp
        self.eps = nn.Parameter(torch.zeros(1))
        self.graph_level = graph_level
        cls = GraphXW if graph_level else NodeXW
        self.q_xw = cls(1, width, bit, alpha_init=0.01, alpha_std=0.01) if quantize else nn.Identity()

    def forward(self, x, edge_index, budget, prompt=None, diagnostics=None, layer_name="", node_mask=None):
        edges, _ = remove_self_loops(edge_index)
        quantized = self.q_xw(x.T).T
        aggregate = sum_messages(quantized, edges)
        context = None
        if prompt is not None and prompt.needs_context:
            if isinstance(self.q_xw, nn.Identity):
                raise ValueError("Conditioning requires a message quantizer")
            context = quantization_context(quantized, edges, self.q_xw)
        corrected = prompt.add(aggregate, context) if prompt is not None else aggregate
        if diagnostics is not None and prompt is not None:
            with torch.no_grad():
                reference = sum_messages(x.detach(), edges)
                diagnostics.add(layer_name, reference, aggregate, corrected, node_mask)
        # Legacy GIN uses the quantized x for its self contribution as well.
        out = (1 + self.eps) * quantized + corrected
        if self.graph_level:
            return self.mlp(out, edges, budget)
        return self.mlp(out), budget


class MNISTGIN(nn.Module):
    def __init__(self, variant, seed, bit=4, num_prompts=20, rank=2, width=110, num_deg=1000):
        super().__init__()
        self.embedding = graph_linear(3, width, num_deg, bit, False)
        self.convs = nn.ModuleList([
            AggregationLayer(GraphMLP(width, num_deg, bit), width, bit, True) for _ in range(4)])
        self.lin1 = graph_linear(width, width, num_deg, bit, False)
        self.lin2 = graph_linear(width, 10, num_deg, bit, True)
        self.prompt_node = GPF_plus(3, num_prompts)
        self.prompt_aggr = make_prompt(variant, width, num_prompts, rank, seed)

    def forward(self, data, diagnostics=None, node_mask=None):
        x = self.prompt_node.add(torch.cat((data.x, data.pos), dim=1))
        x, _, budget = self.embedding(x, data.edge_index, x.new_zeros(1))
        for index, conv in enumerate(self.convs):
            x, budget = conv(x, data.edge_index, budget, self.prompt_aggr,
                             diagnostics, f"layer_{index + 1}", node_mask)
        x = global_mean_pool(x, data.batch)
        x, _, budget = self.lin1(x, data.edge_index, budget)
        x, _, budget = self.lin2(F.relu(x), data.edge_index, budget)
        return x, budget.squeeze()


class CiteSeerGIN(nn.Module):
    def __init__(self, variant, seed, num_nodes, input_width=3703, classes=6,
                 bit=4, num_prompts=20, rank=2, width=16, dropout=0.5):
        super().__init__()
        self.width, self.dropout = width, dropout
        self.conv1 = AggregationLayer(nn.Sequential(
            node_linear(input_width, width, num_nodes, bit), nn.ReLU()), input_width, bit, False, False)
        self.conv2 = AggregationLayer(nn.Sequential(
            node_linear(width, classes, num_nodes, bit), nn.ReLU()), width, bit, False)
        # The original node runner explicitly applies Glorot to backbone weights.
        from torch_geometric.nn.inits import glorot
        for module in self.modules():
            if isinstance(module, nn.Linear):
                glorot(module.weight)
        self.prompt_aggr = make_prompt(variant, width, num_prompts, rank, seed)

    def forward(self, data, diagnostics=None, node_mask=None):
        x, budget = self.conv1(data.x, data.edge_index, data.x.new_zeros(()))
        x = F.dropout(x, self.dropout, training=self.training)
        x, _ = self.conv2(x, data.edge_index, budget, self.prompt_aggr,
                          diagnostics, "layer_2", node_mask)
        # Preserve the original node-level storage proxy (continuous parameters).
        budget = sum(self.width * p.abs().sum() / (8 * 1024)
                     for name, p in self.named_parameters() if "fea_quant.bit" in name)
        return x, budget


def build_model(dataset_name, variant, seed, data=None, **kwargs):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if dataset_name == "MNIST":
            model = MNISTGIN(variant, seed, **kwargs)
        elif dataset_name == "CiteSeer":
            model = CiteSeerGIN(variant, seed, data.num_nodes, data.num_node_features,
                               int(data.y.max()) + 1, **kwargs)
        else:
            raise ValueError(dataset_name)
    # These bits are omitted from the historical optimizers. Explicitly freeze
    # them so gradients do not accumulate in parameters that are never updated.
    for name, p in model.named_parameters():
        if name.endswith(".bit") and "fea_quant" not in name:
            p.requires_grad_(False)
    return model


def state_digest(model, part="backbone"):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if part == "backbone" and name.startswith("prompt_aggr."):
            continue
        if part == "prompt_base" and not any(name == "prompt_aggr." + suffix
                for suffix in ("P_left", "P_right", "a.weight", "a.bias")):
            continue
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def optimizer_for(model, config):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("prompt_"):
            group, lr, decay = "prompts", config["lr"], 0
        elif "fea_quant.bit" in name:
            group, lr, decay = "feature_bits", config["lr_bit"], 0
        elif "fea_quant.gama" in name:
            group, lr, decay = "feature_scale", config["lr_feature_scale"], 0
        elif "weight_quant.alpha" in name:
            group, lr, decay = "weight_scale", config["lr_weight_scale"], 0
        elif "q_xw.alpha" in name:
            group, lr, decay = "message_scale", config["lr_message_scale"], 0
        else:
            group, lr, decay = "backbone", config["lr"], config["weight_decay"]
        groups.setdefault(group, dict(params=[], lr=lr, weight_decay=decay, name=group))["params"].append(p)
    return torch.optim.Adam(list(groups.values()))


def model_counts(model):
    aggregate = parameter_count(model.prompt_aggr)
    node = parameter_count(model.prompt_node) if hasattr(model, "prompt_node") else 0
    return dict(total_parameters=parameter_count(model), aggregate_prompt_parameters=aggregate,
                node_prompt_parameters=node, prompt_fp32_bytes=4 * (aggregate + node),
                mlp_width=getattr(model.prompt_aggr, "hidden_width", None),
                mlp_budget_difference=getattr(model.prompt_aggr, "relative_parameter_difference", None))
