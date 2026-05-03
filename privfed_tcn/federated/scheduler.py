"""Resource-aware client scheduler.

Each client reports ``ClientResource`` telemetry (data freshness, battery,
local training speed). The scheduler combines these into a priority score
and selects the top-K clients per round.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from typing import List

from .. import config


@dataclass
class ClientResource:
    """Per-client resource snapshot."""
    client_id: int
    data_freshness: float = 1.0   # [0, 1] — 1 means freshest
    battery: float = 1.0          # [0, 1] — 1 means fully charged
    speed: float = 1.0            # samples per second (throughput proxy)
    reliability: float = 1.0      # [0, 1] — history of successful rounds


class ResourceAwareScheduler:
    """Selects clients to participate in each FL round."""

    def __init__(self, resources: List[ClientResource],
                 k: int = config.CLIENTS_PER_ROUND,
                 weights: tuple[float, float, float, float] = (0.35, 0.25, 0.25, 0.15),
                 seed: int = config.SEED):
        self.resources = {r.client_id: r for r in resources}
        self.k = k
        self.w = weights
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def score(self, r: ClientResource) -> float:
        wf, wb, ws, wr = self.w
        return wf * r.data_freshness + wb * r.battery + ws * min(1.0, r.speed) + wr * r.reliability

    # ------------------------------------------------------------------
    def select(self, available_ids: List[int] | None = None) -> List[int]:
        ids = available_ids if available_ids is not None else list(self.resources.keys())
        scored = [(cid, self.score(self.resources[cid])) for cid in ids]
        # Softmax-sample for fairness: higher-scored clients more likely but
        # every client has non-zero probability of being picked.
        scores = np.array([s for _, s in scored], dtype=np.float64)
        probs = np.exp(scores - scores.max())
        probs = probs / probs.sum()
        k = min(self.k, len(ids))
        chosen = self.rng.choice(len(ids), size=k, replace=False, p=probs)
        return [scored[i][0] for i in chosen]

    # ------------------------------------------------------------------
    def update(self, client_id: int, **kwargs) -> None:
        r = self.resources[client_id]
        for key, val in kwargs.items():
            if hasattr(r, key):
                setattr(r, key, val)
