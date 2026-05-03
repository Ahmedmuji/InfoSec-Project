"""Track communication cost (MB per FL round)."""
from __future__ import annotations

from typing import Dict, Iterable, List

import torch


def estimate_update_size_mb(state_dict: Dict[str, torch.Tensor]) -> float:
    """Return the size of a state-dict in megabytes (float32 assumed)."""
    total_bytes = 0
    for v in state_dict.values():
        total_bytes += v.numel() * v.element_size()
    return total_bytes / (1024 * 1024)


class CommunicationTracker:
    """Cumulative uplink + downlink tracker across FL rounds."""

    def __init__(self):
        self.rounds: List[Dict[str, float]] = []

    def log_round(self, round_idx: int, n_clients: int, update_mb: float,
                  downlink_mb: float) -> None:
        total_uplink = n_clients * update_mb
        total_downlink = n_clients * downlink_mb
        self.rounds.append({
            "round": round_idx,
            "n_clients": n_clients,
            "uplink_mb": total_uplink,
            "downlink_mb": total_downlink,
            "total_mb": total_uplink + total_downlink,
        })

    @property
    def total_mb(self) -> float:
        return sum(r["total_mb"] for r in self.rounds)

    def summary(self) -> Dict[str, float]:
        if not self.rounds:
            return {"total_mb": 0.0, "mean_round_mb": 0.0}
        total = self.total_mb
        return {
            "total_mb": total,
            "mean_round_mb": total / len(self.rounds),
            "n_rounds": len(self.rounds),
        }
