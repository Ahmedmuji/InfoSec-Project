"""Edge aggregator: SecAgg+ weighted averaging with Krum Byzantine filtering."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .. import config
from ..privacy.secure_agg import SecAggPlus
from .client import ClientUpdate


# ---------------------------------------------------------------------------
# Multi-Krum Byzantine filter.
# ---------------------------------------------------------------------------
def _flatten(delta: Dict[str, torch.Tensor]) -> np.ndarray:
    return torch.cat([v.flatten() for v in delta.values()]).cpu().numpy().astype(np.float64)


def _unflatten(flat: np.ndarray, template: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    idx = 0
    for k, v in template.items():
        n = v.numel()
        out[k] = torch.from_numpy(flat[idx: idx + n]).view_as(v).to(v.dtype)
        idx += n
    return out


def krum_select(updates: Sequence[ClientUpdate], f: int = config.BYZANTINE_F,
                multi: int = 1) -> List[int]:
    """Multi-Krum: select ``multi`` updates minimizing Krum score.

    Krum score of update i = sum of squared distances to its n-f-2 nearest
    neighbours.  Returns indices (into ``updates``) of the selected items.
    Falls back to returning all indices when there are not enough updates
    to satisfy the Krum inequality (n > 2f + 2).
    """
    n = len(updates)
    if n <= 2 * f + 2:
        return list(range(n))

    flats = np.stack([_flatten(u.state_delta) for u in updates], axis=0)
    # Pairwise squared distances
    diffs = flats[:, None, :] - flats[None, :, :]
    dists = np.sum(diffs * diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)

    k = n - f - 2
    sorted_dists = np.sort(dists, axis=1)
    scores = sorted_dists[:, :k].sum(axis=1)
    order = np.argsort(scores)
    return order[:multi].tolist()


# ---------------------------------------------------------------------------
# Edge aggregator class.
# ---------------------------------------------------------------------------
class EdgeAggregator:
    """One edge aggregator serving a cluster of clients."""

    def __init__(self, cluster_id: int, byzantine_f: int = config.BYZANTINE_F):
        self.cluster_id = cluster_id
        self.f = byzantine_f

    # ------------------------------------------------------------------
    def aggregate(self, updates: List[ClientUpdate]
                  ) -> Tuple[Dict[str, torch.Tensor], int, float]:
        """Return ``(aggregated_delta, total_samples, mean_loss)``.

        Steps:
          1. Multi-Krum Byzantine filter (keep ``n − f`` robust updates).
          2. SecAgg+ weighted average over the surviving updates.
        """
        if not updates:
            raise ValueError("No updates provided to edge aggregator")

        n = len(updates)
        keep = max(1, n - self.f)
        keep_idx = krum_select(updates, f=self.f, multi=keep)
        kept = [updates[i] for i in keep_idx]

        flats = [_flatten(u.state_delta) for u in kept]
        total_samples = sum(u.num_samples for u in kept)
        weights = [u.num_samples / total_samples for u in kept]

        secagg = SecAggPlus(n_clients=len(kept), seed=self.cluster_id)
        agg_flat = secagg.aggregate(flats, weights=weights)

        agg_delta = _unflatten(agg_flat, kept[0].state_delta)
        mean_loss = float(np.mean([u.loss for u in kept]))
        return agg_delta, total_samples, mean_loss
