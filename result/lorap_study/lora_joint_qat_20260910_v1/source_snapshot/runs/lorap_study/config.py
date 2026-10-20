"""Locked dataset settings and pilot/full study profiles."""

VARIANTS = ("original", "conditioned", "mlp")
DATASETS = ("CiteSeer", "MNIST")

DEFAULTS = {
    "CiteSeer": dict(data_root="/data/cyliu/dataset/CiteSeer", lr=0.01,
                     lr_feature_scale=0.05, lr_message_scale=0.005, lr_weight_scale=0.005,
                     lr_bit=0.02, a_loss=0.5, a_storage=5., weight_decay=0.01,
                     dropout=0.5, hidden_units=16, layers=2),
    "MNIST": dict(data_root="/data/cyliu/dataset/gnn-mnist", lr=0.001,
                  lr_feature_scale=0.005, lr_message_scale=0.005, lr_weight_scale=0.0005,
                  lr_bit=0.0001, a_loss=0.00005, a_storage=0.1, weight_decay=0.,
                  hidden_units=110, layers=4, init="uniform", num_deg=1000),
}


def dataset_config(name, profile):
    config = dict(DEFAULTS[name], dataset=name, profile=profile, bit=4,
                  num_prompts=20, rank=2, batch_size=128, split_seed=41,
                  selection="minimum validation loss; earliest epoch on ties",
                  diagnostics=True, diagnostic_max_nodes=4096)
    if profile == "full":
        config.update(epochs=200, seeds=list(range(41, 51)), subset_sizes=None)
    elif profile == "pilot":
        config.update(epochs=100 if name == "CiteSeer" else 2,
                      seeds=[41, 42, 43] if name == "CiteSeer" else [41],
                      subset_sizes=None if name == "CiteSeer" else dict(train=512, val=256, test=256))
    else:
        raise ValueError(profile)
    return config
