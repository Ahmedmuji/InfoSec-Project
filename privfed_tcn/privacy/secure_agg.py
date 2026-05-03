"""SecAgg+ simulation using additive secret sharing.

Each client splits its update into N additive shares (one per peer). The
server only receives the element-wise sum of shares, which equals the sum
of client updates, without observing any individual update.

This is a simulation that preserves the protocol's information flow; in a
real deployment pairwise masks would be derived from Diffie-Hellman keys.
"""
from __future__ import annotations

import numpy as np
from typing import List


class SecAggPlus:
    """Toy additive-secret-sharing secure aggregator."""

    def __init__(self, n_clients: int, seed: int = 0):
        self.n = n_clients
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def _share(self, update: np.ndarray) -> List[np.ndarray]:
        """Split ``update`` into ``n`` random shares that sum to it."""
        shares = [self.rng.normal(0, 1.0, size=update.shape).astype(update.dtype)
                  for _ in range(self.n - 1)]
        last = update - sum(shares)
        shares.append(last)
        return shares

    # ------------------------------------------------------------------
    def aggregate(self, updates: List[np.ndarray], weights: List[float] | None = None
                  ) -> np.ndarray:
        """Privately aggregate a list of client updates.

        Returns the (weighted) mean. Each client's update is first shared; the
        simulated server only sums share totals per slot.
        """
        assert len(updates) == self.n, "SecAgg+ expects exactly n client updates"
        if weights is None:
            weights = [1.0 / self.n] * self.n
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()

        # Weighted share construction. Each client scales its update by w_i
        # before sharing, so the summed shares reconstruct sum(w_i · u_i).
        all_shares = []
        for u, wi in zip(updates, w):
            all_shares.append(self._share((u * wi).astype(u.dtype)))

        # Each peer j locally sums the share they received from every client.
        peer_sums = [np.zeros_like(updates[0]) for _ in range(self.n)]
        for client_shares in all_shares:
            for j, s in enumerate(client_shares):
                peer_sums[j] = peer_sums[j] + s

        # Server aggregates peer sums → recovers weighted sum of updates.
        return sum(peer_sums)
