"""Privacy budget logger."""
from __future__ import annotations

from typing import Dict, List

from ..privacy.rdp_accountant import RDPAccountant


class PrivacyLogger:
    """Logs the cumulative (ε, δ) budget after each FL round."""

    def __init__(self, accountant: RDPAccountant):
        self.acc = accountant
        self.history: List[Dict[str, float]] = []

    def log(self, round_idx: int) -> Dict[str, float]:
        eps, alpha = self.acc.get_privacy_spent()
        record = {
            "round": round_idx,
            "epsilon": eps,
            "alpha_star": alpha,
            "delta": self.acc.target_delta,
        }
        self.history.append(record)
        return record

    def current(self) -> Dict[str, float]:
        return self.history[-1] if self.history else {"round": 0, "epsilon": 0.0,
                                                       "alpha_star": float("nan"),
                                                       "delta": self.acc.target_delta}
