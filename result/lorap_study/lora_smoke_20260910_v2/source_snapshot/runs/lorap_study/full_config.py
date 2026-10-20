"""Full official-split GIN comparisons; keep previous experiments immutable."""

from .config import dataset_config

DATASETS = ("Cora", "CiteSeer", "MNIST", "CIFAR10")
VARIANTS = ("none", "original", "main", "spectral_basis")


def full_config(dataset):
    if dataset not in DATASETS:
        raise ValueError(dataset)
    family = "CiteSeer" if dataset in ("Cora", "CiteSeer") else "MNIST"
    config = dataset_config(family, "full")
    config.update(dataset=dataset, family=family, task="node" if family=="CiteSeer" else "graph",
                  variants=list(VARIANTS), device="cuda", calibration_max_nodes=4096,
                  calibration_max_graphs=128, checkpoint_every=1, diagnostics=True,
                  method_note="Locked prior uniform-rank recipe; initial context statistics remain frozen")
    if dataset == "Cora":
        config.update(data_root="/data/cyliu/dataset/Cora", a_loss=2.)
    elif dataset == "CIFAR10":
        config.update(data_root="/data/cyliu/dataset/gnn-cifar10", lr_feature_scale=.01,
                      lr_message_scale=.01, lr_weight_scale=.001, lr_bit=.0002,
                      a_loss=.0002, init="norm")
    return config


def training_config(config):
    """Legacy helpers branch on family; artifact metadata retains actual dataset."""
    return dict(config, dataset=config["family"])


def validate(config):
    reference = full_config(config["dataset"])
    for key in ("family", "task", "hidden_units", "layers", "rank", "num_prompts", "bit", "device"):
        if config[key] != reference[key]:
            raise ValueError(f"Unsupported {key}: {config[key]}")
    if config["subset_sizes"] is not None:
        raise ValueError("Full training requires the complete official splits")
    if config["epochs"] < 1 or not config["seeds"] or not config["variants"]:
        raise ValueError("Positive epochs and nonempty seeds/variants required")
    if len(config["seeds"]) != len(set(config["seeds"])) or len(config["variants"]) != len(set(config["variants"])):
        raise ValueError("Duplicate seeds or variants")
    if any(v not in VARIANTS for v in config["variants"]):
        raise ValueError("Unsupported variant")
