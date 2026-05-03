"""Global configuration for PrivFed-TCN.

All hyperparameters, architectural constants, and paths are declared here
so that every module imports the same single source of truth.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
SEQUENCE_LEN: int = 32
N_FEATURES: int = 41
EMBEDDING_DIM: int = 16
# Number of categorical fields that should be passed through learned embeddings
# (e.g. source port bucket, destination port bucket, protocol id). They are the
# LAST N_CATEGORICAL columns of the 41-feature vector.
N_CATEGORICAL: int = 2
CATEGORICAL_VOCAB: int = 64  # shared vocab size for each categorical field
N_CLASSES: int = 10  # ToN-IoT default; overwritten at runtime if needed

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
TCN_FILTERS: int = 64
TCN_KERNEL: int = 3
TCN_DILATIONS: list[int] = [1, 2, 4]
ATTENTION_HEADS: int = 4
D_MODEL: int = 64
ATTENTION_DROPOUT: float = 0.1
FC1_DIM: int = 128
FC2_DIM: int = 64
DROPOUT: float = 0.1  # reduced from 0.3 — DP-SGD noise already regularises strongly

# ---------------------------------------------------------------------------
# Federated learning
# ---------------------------------------------------------------------------
N_CLIENTS: int = 50
N_ROUNDS: int = 50
LOCAL_EPOCHS: int = 5
CLIENTS_PER_ROUND: int = 10
FEDPROX_MU_DEFAULT: float = 0.01
N_EDGE_AGGREGATORS: int = 5  # cluster clients into edge aggregators
BYZANTINE_F: int = 1  # assumed number of Byzantine clients per cluster

# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------
CLIP_NORM: float = 1.0
NOISE_MULTIPLIER: float = 1.1
TARGET_EPSILON: float = 1.0
TARGET_DELTA: float = 1e-5
RDP_ORDERS: list[float] = [2, 4, 8, 16, 32, 64]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
LEARNING_RATE: float = 1e-3
BATCH_SIZE: int = 1024
WEIGHT_DECAY: float = 1e-4
SEED: int = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TON_IOT_PATH: str = os.path.join(_THIS_DIR, "data", "raw", "ton_iot")
CIC_IOT_PATH: str = os.path.join(_THIS_DIR, "data", "raw", "cic_iot_2023")
RESULTS_PATH: str = os.path.join(_THIS_DIR, "results")
CHECKPOINT_PATH: str = os.path.join(_THIS_DIR, "checkpoints")

os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
