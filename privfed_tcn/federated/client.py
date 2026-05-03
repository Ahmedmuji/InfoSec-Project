"""Federated learning client.

Performs local SGD with a FedProx proximal term and a DP-SGD optimizer.
The adaptive μ coefficient is computed as the Jensen-Shannon distance
between the client's local label histogram and the global one.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .. import config
from ..privacy.dp_sgd import DPSGDOptimizer


# ---------------------------------------------------------------------------
# Utility: Jensen-Shannon distance between two probability vectors.
# ---------------------------------------------------------------------------
def js_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(math.sqrt(max(0.0, 0.5 * (kl_pm + kl_qm))))


@dataclass
class ClientUpdate:
    """Container for messages sent from client → edge aggregator."""
    client_id: int
    state_delta: Dict[str, torch.Tensor]  # w_local - w_global
    num_samples: int
    loss: float
    label_hist: np.ndarray


# ---------------------------------------------------------------------------
# Client implementation
# ---------------------------------------------------------------------------
class FLClient:
    """A federated learning client with DP-SGD + FedProx local training."""

    def __init__(self, client_id: int, model: nn.Module, loader: DataLoader,
                 device: torch.device, n_classes: int,
                 lr: float = config.LEARNING_RATE,
                 weight_decay: float = config.WEIGHT_DECAY,
                 local_epochs: int = config.LOCAL_EPOCHS,
                 clip_norm: float = config.CLIP_NORM,
                 noise_multiplier: float = config.NOISE_MULTIPLIER,
                 fedprox_mu: float = config.FEDPROX_MU_DEFAULT,
                 class_weights: "torch.Tensor | None" = None):
        self.id = client_id
        self.model = model.to(device)
        self.loader = loader
        self.device = device
        self.n_classes = n_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.local_epochs = local_epochs
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.fedprox_mu = fedprox_mu
        # Class weights (e.g. inverse-frequency) for handling imbalance.
        # Stored on the client device for fast loss computation.
        self.class_weights = (class_weights.to(device).float()
                              if class_weights is not None else None)

        # Cached label histogram
        self.label_hist = self._compute_label_hist()

    # ------------------------------------------------------------------
    def _compute_label_hist(self) -> np.ndarray:
        counts = np.zeros(self.n_classes, dtype=np.int64)
        for _, y in self.loader:
            for c in range(self.n_classes):
                counts[c] += int((y == c).sum().item())
        return counts

    def num_samples(self) -> int:
        return int(self.label_hist.sum())

    # ------------------------------------------------------------------
    def set_parameters(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    # ------------------------------------------------------------------
    def train_local(self, global_state: Dict[str, torch.Tensor],
                    global_hist: np.ndarray) -> ClientUpdate:
        """Run ``local_epochs`` of DP-SGD with FedProx + adaptive μ."""
        self.set_parameters(global_state)
        mu = self.fedprox_mu * (1.0 + js_distance(self.label_hist, global_hist))

        # Flatten global params once to massively speed up FedProx
        global_tensors = [global_state[name].to(self.device).detach() for name, _ in self.model.named_parameters()]
        global_vec = torch.cat([g.flatten() for g in global_tensors])

        base_opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay)
        dp_opt = DPSGDOptimizer(base_opt, list(self.model.parameters()),
                                clip_norm=self.clip_norm,
                                noise_multiplier=self.noise_multiplier,
                                expected_batch_size=self.loader.batch_size or config.BATCH_SIZE)
        # Label smoothing 0.05 stabilises training under DP-SGD noise and
        # softens the impact of class_weights on confident majority classes.
        criterion = nn.CrossEntropyLoss(weight=self.class_weights,
                                        label_smoothing=0.05)

        self.model.train()
        total_loss = 0.0
        n_batches = 0
        amp_dtype = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

        for _ in range(self.local_epochs):
            for xb, yb in self.loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                dp_opt.zero_grad()
                
                with torch.autocast(device_type=self.device.type, dtype=amp_dtype):
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                    # Vectorized FedProx proximal term
                    if mu > 0:
                        local_vec = torch.cat([p.flatten() for p in self.model.parameters()])
                        prox = torch.sum((local_vec - global_vec) ** 2)
                        loss = loss + 0.5 * mu * prox
                        
                loss.backward()
                dp_opt.step()
                total_loss += float(loss.item())
                n_batches += 1
        mean_loss = total_loss / max(n_batches, 1)

        # Compute delta w.r.t. global parameters (what we transmit)
        local_state = self.get_parameters()
        delta = {k: local_state[k] - global_state[k].cpu() for k in local_state}
        return ClientUpdate(
            client_id=self.id,
            state_delta=delta,
            num_samples=self.num_samples(),
            loss=mean_loss,
            label_hist=self.label_hist.copy(),
        )


# ---------------------------------------------------------------------------
# Functional entry point used by the Flower simulation.
# ---------------------------------------------------------------------------
def client_update(client: FLClient, global_state: Dict[str, torch.Tensor],
                  global_hist: np.ndarray) -> ClientUpdate:
    return client.train_local(global_state, global_hist)
