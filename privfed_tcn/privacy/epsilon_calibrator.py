"""Binary search calibrator for the DP-SGD noise multiplier.

Given a target privacy budget ``(epsilon, delta)`` and a set of FL training
hyper-parameters, this module computes the minimum noise multiplier
``sigma`` that satisfies the budget, using the Renyi-DP (Mironov 2017)
moments accountant for the subsampled-Gaussian mechanism.

The implementation deliberately re-derives the RDP bound here (rather than
re-using ``rdp_accountant.py``) so that the calibrator stays self-contained
and easy to audit in isolation.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Tuple


# ---------------------------------------------------------------------------
# Core RDP math
# ---------------------------------------------------------------------------
def _log_add(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _rdp_subsampled_gaussian(q: float, sigma: float, alpha: float) -> float:
    """RDP of one step of the subsampled-Gaussian mechanism at order alpha.

    Uses the closed-form binomial expansion of Mironov et al. for integer
    alpha. Non-integer alphas are rounded up (this is a valid upper bound).
    Edge cases:
      - q == 0: zero leakage.
      - q == 1: pure Gaussian RDP = alpha / (2 sigma^2).
    """
    if sigma <= 0:
        return float("inf")
    if q <= 0:
        return 0.0
    if q >= 1.0:
        return alpha / (2.0 * sigma * sigma)

    a = int(math.ceil(alpha))
    if a <= 1:
        return 0.0

    log_terms: List[float] = []
    for i in range(a + 1):
        log_coef = _log_comb(a, i)
        log_q = i * math.log(q)
        log_1mq = (a - i) * math.log(1.0 - q) if (a - i) > 0 else 0.0
        log_gauss = i * (i - 1) / (2.0 * sigma * sigma)
        log_terms.append(log_coef + log_q + log_1mq + log_gauss)

    log_sum = -math.inf
    for t in log_terms:
        log_sum = _log_add(log_sum, t)
    return float(log_sum) / (a - 1)


def _compute_epsilon(sigma: float, q: float, steps: int, target_delta: float,
                     orders: Iterable[float]) -> Tuple[float, float]:
    """Return ``(epsilon, best_alpha)`` after ``steps`` compositions."""
    best_eps = float("inf")
    best_alpha = float("nan")
    for alpha in orders:
        if alpha <= 1:
            continue
        rdp = steps * _rdp_subsampled_gaussian(q, sigma, alpha)
        eps = rdp + math.log(1.0 / target_delta) / (alpha - 1.0)
        if eps < best_eps:
            best_eps = eps
            best_alpha = alpha
    return best_eps, best_alpha


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
DEFAULT_ORDERS: List[float] = [2, 4, 8, 16, 32, 64, 128, 256]


def calibrate_noise_multiplier(target_epsilon: float = 1.0,
                                target_delta: float = 1e-5,
                                n_rounds: int = 50,
                                n_clients: int = 50,
                                clients_per_round: int = 10,
                                local_epochs: int = 5,
                                batch_size: int = 64,
                                dataset_size_per_client: int = 5000,
                                search_min: float = 0.5,
                                search_max: float = 10.0,
                                tolerance: float = 0.01,
                                orders: Iterable[float] | None = None,
                                verbose: bool = True) -> float:
    """Binary-search for the minimum sigma achieving ``target_epsilon``.

    Parameters mirror the FL configuration. The total number of DP-SGD
    composition steps is computed as::

        steps_per_round = local_epochs * (dataset_size_per_client / batch_size)
        total_steps     = n_rounds * (clients_per_round / n_clients) * steps_per_round

    where ``clients_per_round / n_clients`` accounts for per-round client
    sub-sampling (Poisson). The per-step sub-sampling rate ``q`` is the
    minibatch sample rate within a participating client::

        q = batch_size / dataset_size_per_client

    Returns
    -------
    The smallest ``sigma`` in ``[search_min, search_max]`` whose composed
    epsilon is ``<= target_epsilon`` (within ``tolerance``).
    """
    if orders is None:
        orders = DEFAULT_ORDERS

    steps_per_round = max(1, local_epochs * max(1, dataset_size_per_client // batch_size))
    client_participation = clients_per_round / max(1, n_clients)
    total_steps = int(round(n_rounds * client_participation * steps_per_round))
    q = batch_size / max(1, dataset_size_per_client)

    if verbose:
        print("=== Epsilon Calibrator ===")
        print(f"target_epsilon       = {target_epsilon}")
        print(f"target_delta         = {target_delta}")
        print(f"n_rounds             = {n_rounds}")
        print(f"clients_per_round/N  = {clients_per_round}/{n_clients}")
        print(f"local_epochs         = {local_epochs}")
        print(f"batch_size           = {batch_size}")
        print(f"dataset_size/client  = {dataset_size_per_client}")
        print(f"steps/round (per-cli)= {steps_per_round}")
        print(f"effective total steps= {total_steps}")
        print(f"sample rate q        = {q:.6f}")
        print()

        # Pretty-print sigma -> epsilon table
        header = f"{'sigma':>8} | {'epsilon':>10} | {'alpha*':>6}"
        print(header)
        print("-" * len(header))
        for s in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
            eps_s, a_s = _compute_epsilon(s, q, total_steps, target_delta, orders)
            print(f"{s:>8.3f} | {eps_s:>10.4f} | {a_s:>6.0f}")
        print()

    # Confirm the search interval brackets the target.
    eps_lo, _ = _compute_epsilon(search_min, q, total_steps, target_delta, orders)
    eps_hi, _ = _compute_epsilon(search_max, q, total_steps, target_delta, orders)
    if eps_hi > target_epsilon:
        if verbose:
            print(f"WARNING: even sigma={search_max} gives epsilon={eps_hi:.4f} > target. "
                  "Returning search_max.")
        return float(search_max)
    if eps_lo <= target_epsilon:
        if verbose:
            print(f"NOTE: even sigma={search_min} satisfies the target (eps={eps_lo:.4f}). "
                  "Returning search_min.")
        return float(search_min)

    # Binary search: monotone-decreasing eps as sigma grows.
    lo, hi = search_min, search_max
    while hi - lo > tolerance:
        mid = 0.5 * (lo + hi)
        eps_mid, _ = _compute_epsilon(mid, q, total_steps, target_delta, orders)
        if eps_mid > target_epsilon:
            lo = mid
        else:
            hi = mid

    final_sigma = hi
    final_eps, final_alpha = _compute_epsilon(final_sigma, q, total_steps, target_delta, orders)
    if verbose:
        print(f"Converged: sigma={final_sigma:.4f} -> epsilon={final_eps:.4f} "
              f"(alpha*={final_alpha:.0f})")
    return float(final_sigma)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sigma = calibrate_noise_multiplier(
        target_epsilon=1.0,
        target_delta=1e-5,
        n_rounds=50,
        n_clients=50,
        clients_per_round=10,
        local_epochs=5,
        batch_size=64,
        dataset_size_per_client=5000,
    )
    print(f"\nFinal calibrated sigma = {sigma:.4f}")
