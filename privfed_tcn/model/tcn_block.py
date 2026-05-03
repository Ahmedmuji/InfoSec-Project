"""Dilated causal 1-D convolutional block with residual, LayerNorm and GELU."""
from __future__ import annotations

import torch
import torch.nn as nn


class CausalConv1d(nn.Conv1d):
    """1-D causal convolution with left-padding so output_t depends only on inputs <= t."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        self.__left_pad = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size,
                         padding=0, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        x = nn.functional.pad(x, (self.__left_pad, 0))
        return super().forward(x)


class TCNBlock(nn.Module):
    """Single dilated causal conv block.

    Structure: CausalConv1D → GELU → LayerNorm → Dropout → residual add.
    A 1x1 projection matches input/output channels when they differ.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.act = nn.GELU()
        # LayerNorm is applied over the channel dimension per time-step
        self.norm = nn.LayerNorm(out_channels)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input/output shape: (B, C, T)."""
        residual = self.proj(x)
        y = self.conv(x)
        y = self.act(y)
        # LayerNorm expects (..., C) → move C to last, normalise, move back.
        y = y.transpose(1, 2)
        y = self.norm(y)
        y = y.transpose(1, 2)
        y = self.drop(y)
        return y + residual
