"""Data partitioning for federated clients (IID and Dirichlet non-IID)."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from typing import List

from .. import config
from .preprocessor import WindowDataset


def partition_iid(y: np.ndarray, n_clients: int, seed: int = config.SEED) -> List[np.ndarray]:
    """Uniform random split of sample indices across ``n_clients``."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    return [np.array(p) for p in np.array_split(idx, n_clients)]


def partition_dirichlet(y: np.ndarray, n_clients: int, alpha: float,
                        seed: int = config.SEED, min_size: int = 10) -> List[np.ndarray]:
    """Dirichlet(alpha) label-skewed partition.

    Smaller ``alpha`` → greater heterogeneity. This is the standard
    non-IID partitioning scheme used in FL benchmarks.
    """
    rng = np.random.default_rng(seed)
    n_classes = int(y.max()) + 1
    while True:
        client_idx: List[List[int]] = [[] for _ in range(n_clients)]
        for c in range(n_classes):
            idx_c = np.where(y == c)[0]
            rng.shuffle(idx_c)
            props = rng.dirichlet([alpha] * n_clients)
            # Cut points
            cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
            splits = np.split(idx_c, cuts)
            for i, s in enumerate(splits):
                client_idx[i].extend(s.tolist())
        sizes = [len(c) for c in client_idx]
        if min(sizes) >= min_size:
            break
    return [np.array(c) for c in client_idx]


def make_client_loaders(dataset: WindowDataset, partitions: List[np.ndarray],
                        batch_size: int = config.BATCH_SIZE,
                        shuffle: bool = True) -> List[DataLoader]:
    """Wrap each partition into a ``DataLoader``."""
    loaders = []
    pin_mem = torch.cuda.is_available()
    for idx in partitions:
        subset = Subset(dataset, idx.tolist())
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                                   drop_last=False, num_workers=0, pin_memory=pin_mem))
    return loaders
