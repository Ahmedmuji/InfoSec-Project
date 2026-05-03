"""Feature engineering and sliding-window sample construction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple

from .. import config


class RunningNormalizer:
    """Online mean/variance normalizer (Welford's algorithm).

    Designed to handle concept drift on edge devices: statistics can be
    updated continuously as new data arrives.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a batch ``x`` of shape (N, dim)."""
        for row in x:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            delta2 = row - self.mean
            self.m2 += delta * delta2

    @property
    def var(self) -> np.ndarray:
        if self.n < 2:
            return np.ones(self.dim, dtype=np.float64)
        return self.m2 / (self.n - 1)

    def transform(self, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.var + self.eps)
        return (x - self.mean) / std

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.update(x)
        return self.transform(x)


class Preprocessor:
    """Full preprocessing pipeline.

    Converts a flat pandas DataFrame of flow records into sliding-window
    tensors of shape (T, N_FEATURES) with the last ``N_CATEGORICAL`` columns
    left as integer ids for the embedding layer.
    """

    def __init__(self, sequence_len: int = config.SEQUENCE_LEN,
                 n_features: int = config.N_FEATURES,
                 n_categorical: int = config.N_CATEGORICAL,
                 cat_vocab: int = config.CATEGORICAL_VOCAB):
        self.T = sequence_len
        self.F = n_features
        self.C = n_categorical
        self.vocab = cat_vocab
        self.n_numeric = n_features - n_categorical
        self.normalizer = RunningNormalizer(self.n_numeric)

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> None:
        numeric = df.iloc[:, : self.n_numeric].values.astype(np.float32)
        self.normalizer.update(numeric)

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(X, y)`` where X has shape (N_windows, T, F)."""
        numeric = df.iloc[:, : self.n_numeric].values.astype(np.float32)
        numeric = self.normalizer.transform(numeric).astype(np.float32)

        cat = df.iloc[:, self.n_numeric : self.F].values.astype(np.int64)
        cat = np.clip(cat, 0, self.vocab - 1)

        feats = np.concatenate([numeric, cat.astype(np.float32)], axis=1)
        labels = df["label"].values.astype(np.int64)

        # Sliding windows (stride=1). Label of window = label of last row.
        N = len(feats) - self.T + 1
        if N <= 0:
            raise ValueError("Not enough rows to form a single window")
        X = np.lib.stride_tricks.sliding_window_view(feats, (self.T, self.F))[:, 0, :, :]
        X = X[:N]
        y = labels[self.T - 1 : self.T - 1 + N]
        return X, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(df)
        return self.transform(df)

    # ------------------------------------------------------------------
    def split(self, X: np.ndarray, y: np.ndarray, seed: int = config.SEED
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Stratified 70/15/15 train/val/test split.

        Stratification is critical for highly-imbalanced IDS datasets so that
        every class is represented in val/test (otherwise per-class F1 and
        ROC-AUC are undefined for minority classes).  When a class has fewer
        than 3 samples we fall back to placing them in the training set only.
        """
        rng = np.random.default_rng(seed)
        tr_idx: list[int] = []
        va_idx: list[int] = []
        te_idx: list[int] = []
        for c in np.unique(y):
            cls_idx = np.where(y == c)[0]
            rng.shuffle(cls_idx)
            n = len(cls_idx)
            if n < 3:
                tr_idx.extend(cls_idx.tolist())
                continue
            n_train = max(1, int(0.70 * n))
            n_val = max(1, int(0.15 * n))
            # Ensure at least 1 in test as well.
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
            tr_idx.extend(cls_idx[:n_train].tolist())
            va_idx.extend(cls_idx[n_train : n_train + n_val].tolist())
            te_idx.extend(cls_idx[n_train + n_val :].tolist())
        tr = np.array(tr_idx, dtype=np.int64)
        va = np.array(va_idx, dtype=np.int64)
        te = np.array(te_idx, dtype=np.int64)
        # Shuffle each split so batches are not class-ordered.
        rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
        return X[tr], y[tr], X[va], y[va], X[te], y[te]


class WindowDataset(Dataset):
    """Torch ``Dataset`` wrapping ``(X, y)`` arrays of shape (N, T, F), (N,)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
