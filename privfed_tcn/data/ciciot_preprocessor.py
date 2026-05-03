"""Preprocessing helpers for CIC-IoT-2023 to match the PrivFed-TCN schema.

The PrivFed-TCN model expects a frame with ``N_FEATURES = N_NUMERIC +
N_CATEGORICAL`` columns followed by a final integer ``label`` column.
This module converts an arbitrary CIC-IoT-2023 dataframe into that form.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .. import config


def _pick_numeric_features(df: pd.DataFrame, target: int,
                           blocklist_substrings=("label", "timestamp",
                                                  "flow_id", "src_ip",
                                                  "dst_ip", "src_mac",
                                                  "dst_mac", "label_raw"),
                           ) -> List[str]:
    """Pick top-``target`` numeric columns by variance, excluding blocklist."""
    keep: List[str] = []
    for c in df.columns:
        cl = str(c).lower()
        if any(b in cl for b in blocklist_substrings):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and not df[c].isna().all():
            keep.append(c)
    if not keep:
        raise RuntimeError("No usable numeric feature columns in CIC-IoT dataframe.")
    if len(keep) > target:
        variances = df[keep].var(numeric_only=True).fillna(0.0)
        keep = variances.sort_values(ascending=False).head(target).index.tolist()
    return keep


def preprocess_ciciot(df: pd.DataFrame,
                      class_names: List[str],
                      n_features: int = config.N_FEATURES,
                      n_categorical: int = config.N_CATEGORICAL
                      ) -> Tuple[pd.DataFrame, List[str]]:
    """Coerce a CIC-IoT-2023 dataframe to the PrivFed-TCN schema.

    Steps:
      1. Pick up to ``n_features - n_categorical`` numeric columns by variance.
      2. Pad with zero-valued numeric columns if there are fewer than needed.
      3. Append ``n_categorical`` zero-valued integer columns (no real cats
         exist in CIC-IoT-2023 in a useful form, so we keep the embedding
         path active but neutral).
      4. Replace inf/-inf with the per-column finite max/min, fill NaN with 0.
      5. Encode ``label`` as integer codes ordered by ``class_names``.

    Returns
    -------
    (df_out, ordered_class_names)
    """
    n_numeric = n_features - n_categorical

    chosen = _pick_numeric_features(df, target=n_numeric)
    out = df[chosen].copy().astype(float)

    # Replace inf with column finite extremes, then fill NaN with 0.
    for c in out.columns:
        col = out[c]
        finite = col[np.isfinite(col)]
        cmax = finite.max() if len(finite) else 0.0
        cmin = finite.min() if len(finite) else 0.0
        out[c] = col.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out[c] = out[c].clip(lower=cmin, upper=cmax)

    # Pad numeric columns to n_numeric.
    while out.shape[1] < n_numeric:
        pad_name = f"_pad_{out.shape[1]}"
        out[pad_name] = 0.0

    # Append categorical placeholders.
    for i in range(n_categorical):
        out[f"_cat_{i}"] = 0  # neutral category id

    # Encode labels.
    cat = pd.Categorical(df["label"].astype(str), categories=class_names)
    out["label"] = cat.codes.astype(np.int64)
    # Drop rows where the class is outside the enumerated set (cat code = -1).
    out = out[out["label"] >= 0].reset_index(drop=True)
    return out, class_names


# ---------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(500, 60)),
                      columns=[f"f{i}" for i in range(60)])
    df["timestamp"] = np.arange(500)
    df["src_ip"] = "10.0.0.1"
    df["label"] = rng.choice(["DDoS", "Normal", "Reconnaissance"], size=500)
    out, names = preprocess_ciciot(df, class_names=sorted(df["label"].unique()))
    assert out.shape[1] == config.N_FEATURES + 1, out.shape
    assert out["label"].max() >= 0
    print(f"Self-test OK: {out.shape}, classes={names}")


if __name__ == "__main__":
    _self_test()
