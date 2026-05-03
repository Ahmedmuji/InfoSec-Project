"""Local SHAP-based feature attribution.

SHAP values are computed *on-device* and never leave the client, preserving
privacy. If the SHAP library is unavailable we fall back to a gradient-input
saliency method which has the same signature and output format.
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List


class SHAPExplainer:
    """Wrapper around ``shap.GradientExplainer`` with a saliency fallback."""

    def __init__(self, model: torch.nn.Module, background: torch.Tensor,
                 feature_names: List[str] | None = None):
        self.model = model.eval()
        self.background = background
        self.feature_names = feature_names or [f"f{i}" for i in range(background.shape[-1])]
        try:
            import shap  # type: ignore
            self._shap = shap
            self._explainer = shap.GradientExplainer(model, background)
            self._mode = "shap"
        except Exception:
            self._shap = None
            self._explainer = None
            self._mode = "saliency"

    # ------------------------------------------------------------------
    def _saliency(self, x: torch.Tensor) -> np.ndarray:
        """Gradient × input saliency as a SHAP fallback."""
        x = x.clone().detach().requires_grad_(True)
        logits = self.model(x)
        top = logits.argmax(dim=-1)
        selected = logits.gather(1, top.unsqueeze(1)).sum()
        selected.backward()
        return (x.grad * x).detach().cpu().numpy()

    # ------------------------------------------------------------------
    def explain(self, x: torch.Tensor, top_k: int = 5) -> List[Dict[str, float]]:
        """Return a list of ``{feature_name: importance}`` dicts, one per sample.

        Feature importances are averaged across the time dimension so that
        each of the ``N_FEATURES`` positions receives a single scalar value.
        """
        if self._mode == "shap":
            # shap returns list (per class) of arrays of shape like x
            values = self._explainer.shap_values(x, nsamples=50)
            # Stack per-class absolute contributions
            if isinstance(values, list):
                vals = np.mean([np.abs(v) for v in values], axis=0)
            else:
                vals = np.abs(values)
        else:
            vals = np.abs(self._saliency(x))

        # vals shape (B, T, F) → aggregate over T
        agg = vals.mean(axis=1)  # (B, F)

        out: List[Dict[str, float]] = []
        for row in agg:
            idx = np.argsort(-row)[:top_k]
            out.append({self.feature_names[i]: float(row[i]) for i in idx})
        return out
