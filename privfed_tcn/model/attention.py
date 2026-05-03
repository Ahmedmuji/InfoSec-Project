"""Multi-head self-attention with sinusoidal positional encodings."""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal position encoding as in Vaswani et al. 2017."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, d_model)
        return x + self.pe[:, : x.size(1)]


class MultiHeadSelfAttention(nn.Module):
    """Wrapper around ``nn.MultiheadAttention`` with positional encoding + residual."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input shape (B, T, d_model)."""
        z = self.pos(x)
        attn_out, _ = self.mha(z, z, z, need_weights=False)
        return self.norm(x + self.drop(attn_out))
