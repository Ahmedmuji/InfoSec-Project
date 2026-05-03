"""Full PrivFed-TCN local model.

Architecture:
    Input (T, F)
      → Categorical embedding for last N_CATEGORICAL columns (→ dense 16 each)
      → Concatenate with numerical features → (T, F_num + N_cat*emb)
      → Linear projection to TCN filter dim (64)
      → TCN block (dilation=1)
      → TCN block (dilation=2)
      → TCN block (dilation=4)
      → Multi-head self-attention (4 heads, d_model=64) + positional encoding
      → Global Average Pooling over time
      → FC(128) → ReLU → Dropout(0.3)
      → FC(64) → ReLU
      → FC(N_classes) → (logits; softmax is applied by the loss)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .. import config
from .tcn_block import TCNBlock
from .attention import MultiHeadSelfAttention


class PrivFedTCN(nn.Module):
    """Local model deployed at each federated client."""

    def __init__(self,
                 n_features: int = config.N_FEATURES,
                 n_categorical: int = config.N_CATEGORICAL,
                 cat_vocab: int = config.CATEGORICAL_VOCAB,
                 embedding_dim: int = config.EMBEDDING_DIM,
                 tcn_filters: int = config.TCN_FILTERS,
                 tcn_kernel: int = config.TCN_KERNEL,
                 tcn_dilations: list[int] | None = None,
                 n_heads: int = config.ATTENTION_HEADS,
                 d_model: int = config.D_MODEL,
                 attn_dropout: float = config.ATTENTION_DROPOUT,
                 fc1_dim: int = config.FC1_DIM,
                 fc2_dim: int = config.FC2_DIM,
                 dropout: float = config.DROPOUT,
                 n_classes: int = config.N_CLASSES,
                 sequence_len: int = config.SEQUENCE_LEN):
        super().__init__()
        tcn_dilations = tcn_dilations or config.TCN_DILATIONS

        self.n_features = n_features
        self.n_categorical = n_categorical
        self.n_numeric = n_features - n_categorical
        self.embedding_dim = embedding_dim

        # Shared embedding table per categorical field
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_vocab, embedding_dim) for _ in range(n_categorical)
        ])

        embedded_dim = self.n_numeric + n_categorical * embedding_dim  # 39 + 2*16 = 71 (placeholder)
        # Project concatenated features into TCN channel space
        self.input_proj = nn.Linear(embedded_dim, tcn_filters)

        # TCN blocks
        blocks = []
        in_c = tcn_filters
        for d in tcn_dilations:
            blocks.append(TCNBlock(in_c, tcn_filters, tcn_kernel, dilation=d, dropout=dropout * 0.5))
            in_c = tcn_filters
        self.tcn = nn.ModuleList(blocks)

        # Multi-head attention operates on d_model = tcn_filters
        assert tcn_filters == d_model, "TCN_FILTERS must equal D_MODEL for this architecture"
        self.attention = MultiHeadSelfAttention(d_model, n_heads,
                                                dropout=attn_dropout, max_len=sequence_len)

        # Classifier head
        self.fc1 = nn.Linear(d_model, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.out = nn.Linear(fc2_dim, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    # ------------------------------------------------------------------
    def _embed_input(self, x: torch.Tensor) -> torch.Tensor:
        """Apply categorical embeddings and concatenate with numerical features.

        Parameters
        ----------
        x : torch.Tensor, shape (B, T, F). Last ``n_categorical`` columns are
            integer category ids (stored as floats).
        """
        numeric = x[..., : self.n_numeric]
        embeds = []
        for i, emb in enumerate(self.embeddings):
            ids = x[..., self.n_numeric + i].long().clamp(min=0, max=emb.num_embeddings - 1)
            embeds.append(emb(ids))
        return torch.cat([numeric, *embeds], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits of shape (B, n_classes)."""
        # x : (B, T, F)
        x = self._embed_input(x)           # (B, T, embedded_dim)
        x = self.input_proj(x)             # (B, T, tcn_filters)
        x = x.transpose(1, 2)              # (B, C, T) for Conv1d
        for blk in self.tcn:
            x = blk(x)
        x = x.transpose(1, 2)              # (B, T, d_model)
        x = self.attention(x)              # (B, T, d_model)
        x = x.mean(dim=1)                  # GAP over time → (B, d_model)
        x = self.dropout(self.act(self.fc1(x)))
        x = self.act(self.fc2(x))
        return self.out(x)

    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
