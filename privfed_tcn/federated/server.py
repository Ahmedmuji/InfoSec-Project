"""Central FL server.

Maintains the global TCN+Attention model, aggregates cluster-level updates
from edge aggregators using FedProx with adaptive weighting, and tracks
cumulative Rényi DP privacy across rounds.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .. import config
from ..privacy.rdp_accountant import RDPAccountant


class FederatedServer:
    """Central FL coordinator."""

    def __init__(self, model: nn.Module, n_classes: int,
                 noise_multiplier: float = config.NOISE_MULTIPLIER,
                 sample_rate: float = config.CLIENTS_PER_ROUND / max(config.N_CLIENTS, 1),
                 target_delta: float = config.TARGET_DELTA,
                 rdp_orders: List[float] | None = None):
        self.model = model
        self.n_classes = n_classes
        self.global_hist = np.ones(n_classes, dtype=np.float64) / n_classes
        self.accountant = RDPAccountant(
            orders=rdp_orders or config.RDP_ORDERS,
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            target_delta=target_delta,
        )
        self.round = 0

    # ------------------------------------------------------------------
    def get_global_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    # ------------------------------------------------------------------
    def aggregate(self,
                  cluster_deltas: List[Dict[str, torch.Tensor]],
                  cluster_samples: List[int],
                  cluster_losses: List[float],
                  freshness: List[float] | None = None,
                  quality: List[float] | None = None) -> None:
        """FedProx global aggregation with adaptive weighting.

        Weight for cluster i = data_size · freshness · quality (normalised).
        """
        n = len(cluster_deltas)
        if n == 0:
            return
        freshness = freshness or [1.0] * n
        quality = quality or [1.0 / (l + 1e-6) for l in cluster_losses]
        raw = np.array([s * f * q for s, f, q in zip(cluster_samples, freshness, quality)],
                       dtype=np.float64)
        weights = raw / raw.sum()

        global_state = self.get_global_state()
        new_state: Dict[str, torch.Tensor] = {k: v.clone() for k, v in global_state.items()}
        for k in new_state:
            if not torch.is_floating_point(new_state[k]):
                continue
            update = torch.zeros_like(new_state[k], dtype=torch.float32)
            for w, delta in zip(weights, cluster_deltas):
                if k in delta:
                    update = update + float(w) * delta[k].to(update.dtype)
            new_state[k] = new_state[k] + update

        self.model.load_state_dict(new_state, strict=True)
        self.round += 1
        self.accountant.step(1)

    # ------------------------------------------------------------------
    def privacy_spent(self) -> Tuple[float, float]:
        """Return ``(ε, α*)`` cumulative across rounds."""
        return self.accountant.get_privacy_spent()

    # ------------------------------------------------------------------
    def update_global_histogram(self, client_hists: List[np.ndarray]) -> None:
        """Estimate the global label histogram from client samples.

        The server only sees aggregated histograms (a trusted coarse signal
        that is not as sensitive as gradients).
        """
        if not client_hists:
            return
        total = np.sum(client_hists, axis=0).astype(np.float64) + 1e-12
        self.global_hist = total / total.sum()
