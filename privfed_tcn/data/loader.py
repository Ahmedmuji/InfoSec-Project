"""Dataset loaders for ToN-IoT, CIC-IoT-2023 and a synthetic fallback.

If the real CSV files are not present on disk, a synthetic dataset with
identical schema (41 numerical features + label) is generated so that the
rest of the pipeline remains fully runnable end-to-end.
"""
from __future__ import annotations

import os
import glob
import numpy as np
import pandas as pd
from typing import Tuple

from .. import config


# ---------------------------------------------------------------------------
# Class definitions (order matters: index == label id)
# ---------------------------------------------------------------------------
TON_IOT_CLASSES = [
    "normal", "ddos", "dos", "scanning", "backdoor",
    "injection", "xss", "password", "mitm", "ransomware",
]

CIC_IOT_CLASSES = [
    "benign", "ddos", "dos", "recon",
    "web", "brute_force", "spoofing", "mirai",
]


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------
class SyntheticIoTDataset:
    """Generates a synthetic IoT flow dataset that mimics the ToN-IoT schema.

    Each sample is a vector of ``config.N_FEATURES`` features. The final
    ``config.N_CATEGORICAL`` columns represent categorical ids (0..vocab-1)
    and the remaining columns are float numerical features. Class-conditioned
    means create a learnable signal so models can reach high accuracy.
    """

    def __init__(self, n_samples: int = 50_000, n_classes: int = 10, seed: int = 0):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.rng = np.random.default_rng(seed)

    def generate(self) -> pd.DataFrame:
        n_num = config.N_FEATURES - config.N_CATEGORICAL
        # Class prototypes in numerical feature space
        prototypes = self.rng.normal(0, 3.0, size=(self.n_classes, n_num))
        y = self.rng.integers(0, self.n_classes, size=self.n_samples)
        num = prototypes[y] + self.rng.normal(0, 1.0, size=(self.n_samples, n_num))

        # Categorical fields: class-correlated with noise
        cat = np.zeros((self.n_samples, config.N_CATEGORICAL), dtype=np.int64)
        for c in range(config.N_CATEGORICAL):
            base = (y * (7 + c)) % config.CATEGORICAL_VOCAB
            noise = self.rng.integers(0, 4, size=self.n_samples)
            cat[:, c] = (base + noise) % config.CATEGORICAL_VOCAB

        cols = [f"num_{i}" for i in range(n_num)] + [f"cat_{i}" for i in range(config.N_CATEGORICAL)]
        df = pd.DataFrame(np.concatenate([num, cat.astype(np.float32)], axis=1), columns=cols)
        df["label"] = y
        return df


# ---------------------------------------------------------------------------
# Real dataset loaders
# ---------------------------------------------------------------------------
def _load_csv_folder(folder: str) -> pd.DataFrame | None:
    if not os.path.isdir(folder):
        return None
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        return None
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    return pd.concat(dfs, axis=0, ignore_index=True)


def _coerce_schema(df: pd.DataFrame, n_classes: int) -> pd.DataFrame:
    """Project an arbitrary CSV onto the 41-feature + label schema."""
    label_col = None
    for cand in ("label", "Label", "attack", "Attack", "type", "Attack_type"):
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        raise ValueError("No label column found in dataset")

    # Encode label as integer 0..n_classes-1
    y = pd.Categorical(df[label_col]).codes
    y = np.clip(y, 0, n_classes - 1)

    # Use only numeric columns; pad/truncate to N_FEATURES
    numeric = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).fillna(0.0)
    mat = numeric.values.astype(np.float32)
    if mat.shape[1] < config.N_FEATURES:
        pad = np.zeros((mat.shape[0], config.N_FEATURES - mat.shape[1]), dtype=np.float32)
        mat = np.concatenate([mat, pad], axis=1)
    else:
        mat = mat[:, : config.N_FEATURES]

    cols = [f"num_{i}" for i in range(config.N_FEATURES - config.N_CATEGORICAL)] + \
           [f"cat_{i}" for i in range(config.N_CATEGORICAL)]
    out = pd.DataFrame(mat, columns=cols)
    out["label"] = y
    return out


def load_dataset(name: str = "synthetic", n_samples: int = 50_000,
                 custom_path: str = None, limit_samples: int = None,
                 ciciot_data_dir: str = None, limit_files: int = None,
                 limit_rows: int = None) -> Tuple[pd.DataFrame, list[str]]:
    """Load a dataset by name.

    Parameters
    ----------
    name : {"ton_iot", "cic_iot", "synthetic", "custom"}
    n_samples : synthetic dataset size (ignored for real datasets)
    custom_path: Path to custom CSV file if name == "custom"
    limit_samples: Optional max number of rows to load

    Returns
    -------
    (df, class_names)
    """
    name = name.lower()
    if name == "ciciot":
        if not ciciot_data_dir:
            raise ValueError("dataset='ciciot' requires --ciciot_data_dir")
        from .ciciot_loader import CICIoTLoader
        from .ciciot_preprocessor import preprocess_ciciot
        loader = CICIoTLoader(ciciot_data_dir,
                              limit_files=limit_files,
                              limit_rows_per_file=limit_rows)
        raw_df, classes = loader.load()
        coerced, classes = preprocess_ciciot(raw_df, class_names=classes)
        return coerced, classes
    if name == "custom" and custom_path and os.path.isfile(custom_path):
        df = pd.read_csv(custom_path, low_memory=False, nrows=limit_samples)
        # Determine classes from 'type' if available, else 'label'
        if 'type' in df.columns:
            classes = sorted(df['type'].dropna().unique().tolist())
            df['label'] = pd.Categorical(df['type'], categories=classes).codes
        elif 'label' in df.columns:
            classes = [str(c) for c in sorted(df['label'].dropna().unique().tolist())]
        else:
            classes = ["benign", "malicious"]
        return _coerce_schema(df, len(classes)), classes
    elif name == "ton_iot":
        df = _load_csv_folder(config.TON_IOT_PATH)
        classes = TON_IOT_CLASSES
        if df is not None:
            return _coerce_schema(df, len(classes)), classes
    elif name == "cic_iot":
        df = _load_csv_folder(config.CIC_IOT_PATH)
        classes = CIC_IOT_CLASSES
        if df is not None:
            return _coerce_schema(df, len(classes)), classes
    elif name != "synthetic":
        raise ValueError(f"Unknown dataset: {name}")

    # Fallback: synthetic
    classes = TON_IOT_CLASSES if name in ("synthetic", "ton_iot") else CIC_IOT_CLASSES
    syn = SyntheticIoTDataset(n_samples=n_samples, n_classes=len(classes), seed=config.SEED)
    return syn.generate(), classes
