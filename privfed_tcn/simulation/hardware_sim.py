"""Hardware profiling hooks for Raspberry Pi 4 / Jetson Nano style devices."""
from __future__ import annotations

from typing import Dict, List

import torch

from ..evaluation.hardware import HardwareBenchmark
from ..model.privfed_tcn import PrivFedTCN
from .. import config


def profile_on_devices(n_classes: int = config.N_CLASSES,
                       batch_size: int = 1) -> List[Dict[str, float]]:
    """Profile on CPU (Raspberry Pi 4 proxy) and, if available, CUDA (Jetson proxy)."""
    results = []
    bench = HardwareBenchmark(average_power_w=3.5)
    sample = torch.randn(batch_size, config.SEQUENCE_LEN, config.N_FEATURES)

    # Raspberry Pi 4 proxy (CPU)
    cpu_model = PrivFedTCN(n_classes=n_classes)
    cpu_res = bench.benchmark(cpu_model, sample.clone(), device=torch.device("cpu"))
    cpu_res["simulated_device"] = "Raspberry Pi 4 (CPU)"
    results.append(cpu_res)

    # Jetson Nano proxy (CUDA if available)
    if torch.cuda.is_available():
        bench_gpu = HardwareBenchmark(average_power_w=5.0)
        gpu_model = PrivFedTCN(n_classes=n_classes)
        gpu_res = bench_gpu.benchmark(gpu_model, sample.clone(), device=torch.device("cuda"))
        gpu_res["simulated_device"] = "Jetson Nano (CUDA)"
        results.append(gpu_res)

    return results
