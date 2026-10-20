"""Official splits with fixed stratified subsets for the MNIST smoke run."""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import GNNBenchmarkDataset, Planetoid
from torch_geometric.loader import DataLoader


def stratified_indices(labels, size, seed):
    labels = np.asarray(labels).reshape(-1)
    indices = np.arange(len(labels))
    if size is None or size == len(labels):
        return indices.tolist()
    if not 0 < size < len(labels):
        raise ValueError("Subset size must be positive and no larger than its official split")
    selected, _ = train_test_split(indices, train_size=size, stratify=labels, random_state=seed)
    return np.sort(selected).tolist()


def load_data(config):
    if config["dataset"] == "CiteSeer":
        dataset = Planetoid(config["data_root"], "CiteSeer", split="public")
        data = dataset[0]
        return data, dict(kind="public", nodes=data.num_nodes, features=data.num_node_features,
                          edges=data.edge_index.shape[1],
                          indices={split: getattr(data, f"{split}_mask").nonzero().flatten().tolist()
                                   for split in ("train", "val", "test")})
    datasets, manifest = {}, dict(kind="official", indices={}, class_counts={})
    for split in ("train", "val", "test"):
        full = GNNBenchmarkDataset(config["data_root"], "MNIST", split=split)
        labels = full._data.y.cpu().numpy().reshape(-1)
        size = config["subset_sizes"][split] if config["subset_sizes"] else None
        indices = stratified_indices(labels, size, config["split_seed"])
        datasets[split] = full[indices]
        manifest["indices"][split] = indices
        values, counts = np.unique(labels[indices], return_counts=True)
        manifest["class_counts"][split] = dict(zip(map(str, values.tolist()), counts.tolist()))
    return datasets, manifest


def loaders_for(datasets, config, seed):
    return {split: DataLoader(dataset, batch_size=config["batch_size"],
                              shuffle=split == "train", num_workers=0,
                              generator=torch.Generator().manual_seed(seed + 200_000 + index))
            for index, (split, dataset) in enumerate(datasets.items())}
