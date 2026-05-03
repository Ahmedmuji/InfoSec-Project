"""Evaluation utilities: classification metrics, communication, hardware, privacy."""
from .metrics import compute_classification_metrics, confusion_matrix_plot
from .communication import CommunicationTracker, estimate_update_size_mb
from .hardware import HardwareBenchmark
from .privacy_eval import PrivacyLogger

__all__ = [
    "compute_classification_metrics",
    "confusion_matrix_plot",
    "CommunicationTracker",
    "estimate_update_size_mb",
    "HardwareBenchmark",
    "PrivacyLogger",
]
