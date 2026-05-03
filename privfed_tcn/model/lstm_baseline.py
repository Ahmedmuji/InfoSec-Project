"""FL-LSTM baseline used to benchmark PrivFed-TCN's communication cost.

Replicates a standard 2-layer stacked LSTM intrusion-detection model
(Anwar et al. 2025-style). Designed to land at roughly 300-400K params,
which is several times the size of PrivFed-TCN, so the communication
delta between the two becomes a meaningful comparison.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .. import config


class FLLSTMBaseline(nn.Module):
    """Two-layer LSTM + MLP head for intrusion detection."""

    def __init__(self,
                 input_dim: int = config.N_FEATURES,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 n_classes: int = config.N_CLASSES,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, n_classes)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, F)
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]              # (B, hidden_dim) — last layer's last hidden
        h = self.drop(self.act(self.fc1(h)))
        return self.fc2(h)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    m = FLLSTMBaseline()
    x = torch.randn(4, config.SEQUENCE_LEN, config.N_FEATURES)
    print(f"FL-LSTM params: {m.num_parameters():,}")
    print(f"Output shape  : {m(x).shape}")
