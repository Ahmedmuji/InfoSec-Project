"""Rényi Differential Privacy moments accountant.

Implements the subsampled Gaussian mechanism RDP bounds from
Mironov 2017 / Wang et al. 2019 and the standard RDP → (ε, δ) conversion.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Tuple

from .. import config


def _log_add(a: float, b: float) -> float:
    """Stable computation of log(exp(a) + exp(b))."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _compute_rdp_sampled_gaussian(q: float, sigma: float, alpha: float) -> float:
    """RDP of a single step of subsampled-Gaussian mechanism.

    Uses the closed-form for integer α via binomial expansion (Mironov et al.).
    For non-integer α we round up to the next integer which yields a valid
    upper bound by monotonicity.
    """
    if q == 0:
        return 0.0
    if q == 1.0:
        return alpha / (2 * sigma ** 2)

    int_alpha = int(math.ceil(alpha))
    log_terms = []
    for i in range(int_alpha + 1):
        log_coef = _log_comb(int_alpha, i)
        log_q = i * math.log(q) if q > 0 else (-math.inf if i > 0 else 0.0)
        log_1mq = (int_alpha - i) * math.log(1 - q) if q < 1 else (-math.inf if int_alpha - i > 0 else 0.0)
        log_gauss = i * (i - 1) / (2 * sigma ** 2)
        log_terms.append(log_coef + log_q + log_1mq + log_gauss)

    log_sum = -math.inf
    for t in log_terms:
        log_sum = _log_add(log_sum, t)
    return float(log_sum) / (int_alpha - 1) if int_alpha > 1 else float(log_sum)


def compute_rdp(q: float, noise_multiplier: float, steps: int,
                orders: Iterable[float]) -> List[float]:
    """Compute RDP at each order after ``steps`` steps of subsampled Gaussian."""
    return [steps * _compute_rdp_sampled_gaussian(q, noise_multiplier, a) for a in orders]


def rdp_to_dp(orders: Iterable[float], rdp: Iterable[float], target_delta: float
              ) -> Tuple[float, float]:
    """Convert RDP values at several orders to the tightest (ε, α) for the
    given target δ using the standard conversion
        ε = rdp(α) + log(1/δ) / (α − 1).
    """
    eps_list = []
    for a, r in zip(orders, rdp):
        if a <= 1:
            continue
        eps = r + math.log(1.0 / target_delta) / (a - 1)
        eps_list.append((eps, a))
    eps_list.sort()
    return eps_list[0] if eps_list else (float("inf"), float("nan"))


class RDPAccountant:
    """Stateful tracker for RDP composition across FL rounds."""

    def __init__(self, orders: List[float] | None = None,
                 noise_multiplier: float = config.NOISE_MULTIPLIER,
                 sample_rate: float = 0.01,
                 target_delta: float = config.TARGET_DELTA):
        self.orders = list(orders or config.RDP_ORDERS)
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.target_delta = target_delta
        self.rdp = [0.0 for _ in self.orders]
        self.steps = 0

    # ------------------------------------------------------------------
    def step(self, n_steps: int = 1) -> None:
        """Account for ``n_steps`` additional subsampled-Gaussian applications."""
        add = compute_rdp(self.sample_rate, self.noise_multiplier, n_steps, self.orders)
        self.rdp = [r + a for r, a in zip(self.rdp, add)]
        self.steps += n_steps

    # ------------------------------------------------------------------
    def get_privacy_spent(self, steps: int | None = None,
                          noise_multiplier: float | None = None,
                          sample_rate: float | None = None,
                          target_delta: float | None = None
                          ) -> Tuple[float, float]:
        """Return ``(ε, α*)`` for cumulative composition.

        If explicit args are provided they override the accumulated state and
        compute a fresh ε for ``steps`` applications.
        """
        delta = target_delta if target_delta is not None else self.target_delta
        if steps is None:
            rdp = self.rdp
        else:
            sigma = noise_multiplier if noise_multiplier is not None else self.noise_multiplier
            q = sample_rate if sample_rate is not None else self.sample_rate
            rdp = compute_rdp(q, sigma, steps, self.orders)
        return rdp_to_dp(self.orders, rdp, delta)
