"""Four projected scalar gates; integer residual transport is a detached branch."""

import hashlib

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import remove_self_loops

from ..residual_channel.core import graph_operator, residual_correction, selected_precision


def is_quantizer(name):
    return any(part in name for part in (".fea_quant.", ".weight_quant.", ".q_xw."))


def digest_subset(model, predicate):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if predicate(name):
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def is_bn_stat(name):
    return name.endswith(("running_mean", "running_var", "num_batches_tracked"))


class FixedCode(nn.Module):
    def __init__(self, info):
        super().__init__()
        self.register_buffer("basis", info["basis"].clone())
        self.register_buffer("scale", info["scale"].clone())


class AdaptedGIN(nn.Module):
    def __init__(self, backbone, calibration, variant):
        super().__init__()
        self.backbone = backbone
        self.variant = variant
        self.gated = variant.startswith("gate")
        self.finetune = variant.endswith("_ft")
        if self.gated:
            self.gates = nn.Parameter(torch.zeros(len(backbone.convs)))
            self.codes = nn.ModuleList([FixedCode(c["graph"]) for c in calibration])
        else:
            self.register_parameter("gates", None)
        if variant == "higher16_ft":
            for i, c in enumerate(calibration):
                self.register_buffer(f"upgrades_{i}", c["channels"]["higher_16"].clone())
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(self.finetune and not is_quantizer("backbone."+name))
        self.eval()

    def training_mode(self):
        self.train(self.finetune)

    @torch.no_grad()
    def project_gates(self):
        if self.gates is not None:
            self.gates.clamp_(0, 1)

    def gate_values(self):
        return self.gates.detach().cpu().tolist() if self.gates is not None else None

    def forward(self, data, observer=None, bypass_messages=False):
        if data.x.device.type != "cuda":
            raise RuntimeError("All model computation requires CUDA")
        model = self.backbone
        x = model.prompt_node.add(torch.cat((data.x, data.pos), 1))
        x, _, budget = model.embedding(x, data.edge_index, x.new_zeros(1))
        edges, _ = remove_self_loops(data.edge_index)
        for i, conv in enumerate(model.convs):
            if bypass_messages:
                q = x
            elif self.variant == "higher16_ft":
                q = selected_precision(x, conv.q_xw, getattr(self, f"upgrades_{i}"))
            else:
                q = conv.q_xw(x.T).T
            out = graph_operator(q, edges, conv.eps)
            delta = None
            if self.gated:
                # The true residual and integer transport are recomputed every
                # forward, with stop-gradient. Train the gate and the main route,
                # not an unspecified derivative through integer rounding/casts.
                with torch.no_grad():
                    code = self.codes[i]
                    delta, _ = residual_correction(x.detach()-q.detach(), edges,
                        conv.eps.detach(), code.basis, code.scale)
                out = out + self.gates[i]*delta
            if observer is not None:
                with torch.no_grad():
                    observer(i, x.detach(), q.detach(), edges, conv.eps.detach(),
                             None if delta is None else self.gates[i].detach()*delta)
            x, budget = conv.mlp(out, edges, budget)
        x = global_mean_pool(x, data.batch)
        x, _, budget = model.lin1(x, data.edge_index, budget)
        x, _, budget = model.lin2(F.relu(x), data.edge_index, budget)
        return x


def optimizer_for(model, config):
    groups = []
    if model.gates is not None:
        groups.append(dict(params=[model.gates],lr=config["gate_lr"],weight_decay=0.,name="gates"))
    weights = [p for p in model.backbone.parameters() if p.requires_grad]
    if weights:
        groups.append(dict(params=weights,lr=config["backbone_lr"],weight_decay=0.,name="backbone"))
    return torch.optim.Adam(groups)


@torch.no_grad()
def recalibrate_bn(model, batches, device):
    """Sequential BN population moments with already-corrected prefixes.

    All modules remain in eval mode. Node-weighted sufficient statistics are
    gathered at one BN at a time, avoiding batch-normalized downstream inputs.
    """
    model.eval()
    layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    records = []
    for layer in layers:
        moments = {}

        def capture(module, inputs):
            x = inputs[0].double()
            moments["count"] = moments.get("count",0)+len(x)
            moments["sum"] = moments.get("sum",0)+x.sum(0)
            moments["squared"] = moments.get("squared",0)+x.square().sum(0)
            moments["batches"] = moments.get("batches",0)+1

        handle = layer.register_forward_pre_hook(capture)
        try:
            for batch in batches:
                model(batch.to(device))
        finally:
            handle.remove()
        n = moments["count"]
        if n < 2:
            raise ValueError("BN recalibration requires at least two nodes")
        mean = moments["sum"]/n
        variance = ((moments["squared"]-moments["sum"].square()/n)/(n-1)).clamp_min(0)
        layer.running_mean.copy_(mean.to(layer.running_mean.dtype))
        layer.running_var.copy_(variance.to(layer.running_var.dtype))
        layer.num_batches_tracked.fill_(moments["batches"])
        records.append(dict(nodes=n,batches=moments["batches"],max_variance=float(variance.max())))
    return records
