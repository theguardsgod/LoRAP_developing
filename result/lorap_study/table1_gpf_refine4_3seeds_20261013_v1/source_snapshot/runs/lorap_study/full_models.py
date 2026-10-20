"""Generic dimensions for Cora and CIFAR10, using the existing quantizers."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.datasets import GNNBenchmarkDataset, Planetoid

from prompt import GPF_plus
from .improved import LayerwiseMNIST
from .mechanism import NoPrompt
from .models import AggregationLayer, GraphLinear, PARA, build_model
from .prompts import make_prompt


def load_full_data(config):
    name, root = config["dataset"], config["data_root"]
    if config["task"] == "node":
        data = Planetoid(root, name, split="public")[0]
        manifest = dict(dataset=name, split="public", nodes=data.num_nodes,
                        edges=data.num_edges, features=data.num_node_features,
                        classes=int(data.y.max())+1,
                        indices={s: getattr(data, f"{s}_mask").nonzero().flatten().tolist()
                                 for s in ("train", "val", "test")})
        return data, manifest
    data, manifest = {}, dict(dataset=name, split="official_full", indices={}, class_counts={})
    for split in ("train", "val", "test"):
        data[split] = GNNBenchmarkDataset(root, name, split=split)
        labels = data[split]._data.y.flatten()
        values, counts = labels.unique(return_counts=True)
        manifest["indices"][split] = list(range(len(data[split])))
        manifest["class_counts"][split] = {str(int(k)): int(v) for k, v in zip(values, counts)}
    manifest["features"] = data["train"][0].num_node_features
    manifest["position_features"] = data["train"][0].pos.size(1)
    manifest["classes"] = data["train"].num_classes
    return data, manifest


def linear(inputs, outputs, config, positive=False):
    cls = GraphLinear
    if config.get("quantization_mode") == "fixed4":
        from quantize_function.u_quant_normal_debug import QLinear
        cls = QLinear
    layer = cls(inputs, outputs, config["num_deg"], config["bit"],
                        para_dict=PARA, all_positive=positive, uniform=True, init=config["init"])
    layer.fea_quant.study_unsigned = positive
    return layer


class GraphMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config["hidden_units"]
        self.linear1 = linear(width, width, config)
        self.bn1 = nn.BatchNorm1d(width)
        self.linear2 = linear(width, width, config, True)
        self.bn2 = nn.BatchNorm1d(width)

    def forward(self, x, edges, budget):
        x, _, budget = self.linear1(x, edges, budget)
        x = F.relu(self.bn1(x))
        x, _, budget = self.linear2(x, edges, budget)
        return F.relu(self.bn2(x)), budget


class GraphGIN(LayerwiseMNIST):
    def __init__(self, config, seed, input_width, classes):
        nn.Module.__init__(self)
        width = config["hidden_units"]
        self.embedding = linear(input_width, width, config)
        self.convs = nn.ModuleList([AggregationLayer(GraphMLP(config), width, config["bit"], True)
                                   for _ in range(config["layers"])])
        self.lin1 = linear(width, width, config)
        self.lin2 = linear(width, classes, config, True)
        self.prompt_node = GPF_plus(input_width, config["num_prompts"])
        self.prompt_aggr = make_prompt("original", width, config["num_prompts"], config["rank"], seed)


def build_full_backbone(config, seed, data):
    if config["task"] == "node":
        model = build_model("CiteSeer", "original", seed, data, dropout=config["dropout"],
                            width=config["hidden_units"], bit=config["bit"])
    else:
        example = data["train"][0]
        classes = int(config.get("classes", 10))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = GraphGIN(config, seed, example.x.size(1)+example.pos.size(1), classes)
    model.prompt_aggr = NoPrompt()
    for name, parameter in model.named_parameters():
        if name.endswith(".bit") and ("fea_quant" not in name or config.get("quantization_mode") == "fixed4"):
            parameter.requires_grad_(False)
    return model
