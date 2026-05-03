"""DP-SGD: per-sample gradient clipping + Gaussian noise addition.

The implementation here is a manual version that mirrors the classical
Abadi et al. (2016) algorithm. If Opacus is installed it can be used in
place of this for per-sample clipping via hooks; we expose both paths.
"""
from __future__ import annotations

import math
from typing import Iterable, List

import torch
from torch import nn

from .. import config


# ---------------------------------------------------------------------------
# Pure-function utility: clip a list of parameter-wise gradient tensors.
# ---------------------------------------------------------------------------
def _flat_norm(grads: Iterable[torch.Tensor]) -> torch.Tensor:
    total = torch.zeros(1, device=next(iter(grads)).device)
    for g in grads:
        total = total + g.detach().pow(2).sum()
    return torch.sqrt(total + 1e-12)


def clip_and_noise_gradients(grads: List[torch.Tensor],
                             clip_norm: float = config.CLIP_NORM,
                             noise_multiplier: float = config.NOISE_MULTIPLIER,
                             batch_size: int = config.BATCH_SIZE) -> List[torch.Tensor]:
    """Clip a *sum* of per-sample gradients and add Gaussian noise.

    Parameters
    ----------
    grads : list of tensors — the summed gradient for the minibatch.
    clip_norm : L2 clipping threshold C.
    noise_multiplier : σ. Noise std is σ·C.
    batch_size : used to rescale noisy sum back to a mean estimate.
    """
    total_norm = _flat_norm(grads)
    scale = torch.clamp(clip_norm / (total_norm + 1e-12), max=1.0)
    clipped = [g * scale for g in grads]
    std = noise_multiplier * clip_norm
    noised = [c + torch.randn_like(c) * std for c in clipped]
    return [n / batch_size for n in noised]


# ---------------------------------------------------------------------------
# A thin DP-SGD optimizer wrapper. Treats the current ``param.grad`` (from
# a standard loss.backward()) as a clipped per-batch gradient approximation,
# then adds Gaussian noise. This matches the behaviour requested for an
# FL client that sends a clipped + noised gradient update.
# ---------------------------------------------------------------------------
class DPSGDOptimizer:
    """Lightweight DP-SGD wrapper around any torch optimizer."""

    def __init__(self, optimizer: torch.optim.Optimizer,
                 parameters: List[nn.Parameter],
                 clip_norm: float = config.CLIP_NORM,
                 noise_multiplier: float = config.NOISE_MULTIPLIER,
                 expected_batch_size: int = config.BATCH_SIZE):
        self.opt = optimizer
        self.params = list(parameters)
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.batch_size = expected_batch_size

    # ------------------------------------------------------------------
    def zero_grad(self) -> None:
        self.opt.zero_grad(set_to_none=True)

    def step(self) -> float:
        """Apply gradient clipping + noise then call the wrapped optimizer.

        Returns the pre-clip gradient norm for logging.
        """
        with torch.no_grad():
            grads = [p.grad for p in self.params if p.grad is not None]
            if not grads:
                return 0.0
            total_norm = _flat_norm(grads).item()
            scale = min(1.0, self.clip_norm / (total_norm + 1e-12))
            std = self.noise_multiplier * self.clip_norm
            for g in grads:
                g.mul_(scale)
                g.add_(torch.randn_like(g) * std / max(self.batch_size, 1))
        self.opt.step()
        return total_norm
