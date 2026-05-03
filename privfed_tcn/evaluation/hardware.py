"""Hardware-level benchmarks: latency, energy estimate, memory."""
from __future__ import annotations

import time
from typing import Dict

import torch
import torch.nn as nn


class HardwareBenchmark:
    """Profiles inference latency and estimates energy per inference.

    The energy estimate uses a simple model:
        energy_mJ ≈ latency_s · average_power_W · 1000
    With ``average_power_W`` defaulting to 3.5 W (Raspberry Pi 4 typical).
    """

    def __init__(self, average_power_w: float = 3.5):
        self.power = average_power_w

    # ------------------------------------------------------------------
    def param_count(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    def model_size_kb(self, model: nn.Module) -> float:
        b = sum(p.numel() * p.element_size() for p in model.parameters())
        return b / 1024.0

    # ------------------------------------------------------------------
    def benchmark(self, model: nn.Module, sample: torch.Tensor,
                  n_warmup: int = 10, n_iters: int = 100,
                  device: torch.device | None = None) -> Dict[str, float]:
        """Return latency (ms), throughput (samples/s), energy (mJ), memory (MB)."""
        device = device or torch.device("cpu")
        model = model.to(device).eval()
        sample = sample.to(device)

        with torch.no_grad():
            for _ in range(n_warmup):
                model(sample)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iters):
                model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

        latency_ms = ((t1 - t0) / n_iters) * 1000.0
        batch = sample.shape[0]
        throughput = batch * n_iters / (t1 - t0)
        energy_mj = (t1 - t0) / n_iters * self.power * 1000.0  # mJ per forward
        mem_mb = (torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                  if device.type == "cuda" else self.model_size_kb(model) / 1024)

        return {
            "device": str(device),
            "latency_ms": latency_ms,
            "throughput_sps": throughput,
            "energy_mj_per_inference": energy_mj,
            "memory_mb": float(mem_mb),
            "param_count": self.param_count(model),
            "model_size_kb": self.model_size_kb(model),
        }
