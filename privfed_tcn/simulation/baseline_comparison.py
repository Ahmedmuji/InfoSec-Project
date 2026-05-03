"""Side-by-side communication-cost comparison: PrivFed-TCN vs FL-LSTM.

Trains the FL-LSTM baseline on the same dataset/partition the user already
ran for PrivFed-TCN, then writes ``baseline_comparison.json`` summarising
both runs (parameter count, model size, MB/round, accuracy, F1).

The TCN result is supplied directly so we don't pay for a redundant run.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .. import config
from ..data.loader import load_dataset
from ..data.preprocessor import Preprocessor, WindowDataset
from ..data.partitioner import partition_iid, partition_dirichlet, make_client_loaders
from ..model.lstm_baseline import FLLSTMBaseline
from ..evaluation.metrics import compute_classification_metrics
from ..privacy.dp_sgd import DPSGDOptimizer


# ---------------------------------------------------------------------------
def _model_size_kb(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024.0


def _comm_mb_per_round(model: nn.Module, clients_per_round: int) -> float:
    """MB transferred per round = clients_per_round * (uplink + downlink)."""
    bytes_per_param = 4  # float32
    n_params = sum(p.numel() for p in model.parameters())
    one_way_mb = n_params * bytes_per_param / (1024 * 1024)
    return clients_per_round * 2 * one_way_mb


# ---------------------------------------------------------------------------
def _train_lstm_round(model: nn.Module, loaders: List[DataLoader],
                      device: torch.device, local_epochs: int,
                      lr: float = 1e-3,
                      noise_multiplier: float = 0.0,
                      clip_norm: float = config.CLIP_NORM,
                      class_weights: torch.Tensor | None = None) -> float:
    """Run one FL round and return mean loss.

    If ``noise_multiplier > 0`` the local optimizer is wrapped with DP-SGD
    so the baseline is evaluated under the *same* privacy budget as
    PrivFed-TCN (a fair side-by-side comparison).
    """
    global_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    deltas: List[Dict[str, torch.Tensor]] = []
    weights: List[float] = []
    losses: List[float] = []

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    cw = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=cw)

    for loader in loaders:
        local = copy.deepcopy(model).to(device)
        local.load_state_dict(global_state)
        base_opt = torch.optim.Adam(local.parameters(), lr=lr)
        if noise_multiplier > 0:
            opt = DPSGDOptimizer(base_opt, list(local.parameters()),
                                 clip_norm=clip_norm,
                                 noise_multiplier=noise_multiplier,
                                 expected_batch_size=loader.batch_size or config.BATCH_SIZE)
        else:
            opt = base_opt
        local.train()
        n_seen = 0
        loss_sum = 0.0
        for _ in range(local_epochs):
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad()
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    logits = local(xb)
                    loss = criterion(logits, yb)
                loss.backward()
                opt.step()
                loss_sum += float(loss.item()) * yb.size(0)
                n_seen += yb.size(0)
        if n_seen == 0:
            continue
        losses.append(loss_sum / n_seen)
        weights.append(float(n_seen))
        local_state = {k: v.detach().cpu() for k, v in local.state_dict().items()}
        deltas.append({k: local_state[k] - global_state[k].cpu() for k in local_state})

    if not deltas:
        return float("nan")

    w = np.array(weights, dtype=np.float64)
    w /= w.sum()
    # Aggregate on CPU to avoid cross-device tensor ops.
    new_state = {k: v.detach().cpu().clone() for k, v in global_state.items()}
    for k in new_state:
        if not torch.is_floating_point(new_state[k]):
            continue
        agg = torch.zeros_like(new_state[k], dtype=torch.float32)
        for wi, d in zip(w, deltas):
            if k in d:
                agg = agg + float(wi) * d[k].to(agg.dtype)
        new_state[k] = (new_state[k] + agg).to(new_state[k].dtype)
    model.load_state_dict(new_state)
    model.to(device)
    return float(np.average(losses, weights=weights))


# ---------------------------------------------------------------------------
def _build_loaders(args, device):
    df, class_names = load_dataset(
        args.dataset,
        custom_path=getattr(args, "custom_data_path", None),
        limit_samples=getattr(args, "limit_samples", None),
        ciciot_data_dir=getattr(args, "ciciot_data_dir", None),
        limit_files=getattr(args, "limit_files", None),
        limit_rows=getattr(args, "limit_rows", None),
    )
    prep = Preprocessor()
    X, y = prep.fit_transform(df)
    Xtr, ytr, _Xv, _yv, Xte, yte = prep.split(X, y)
    train_ds = WindowDataset(Xtr, ytr)
    test_ds = WindowDataset(Xte, yte)
    if args.partition == "iid":
        parts = partition_iid(ytr, args.clients)
    elif args.partition == "dirichlet_05":
        parts = partition_dirichlet(ytr, args.clients, alpha=0.5)
    else:
        parts = partition_dirichlet(ytr, args.clients, alpha=0.1)
    client_loaders = make_client_loaders(train_ds, parts)
    pin = torch.cuda.is_available()
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False, pin_memory=pin)
    # Inverse-frequency class weights for fair comparison vs PrivFed-TCN.
    n_classes = len(class_names)
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (n_classes * counts)
    w = np.clip(w, 0.5, 50.0)
    class_weights = torch.tensor(w, dtype=torch.float32)
    return client_loaders, test_loader, class_names, class_weights


# ---------------------------------------------------------------------------
def run_baseline_comparison(args, device, tcn_result: Dict[str, Any]) -> Dict[str, Any]:
    """Run the FL-LSTM baseline and emit ``baseline_comparison.json``.

    Parameters
    ----------
    args : argparse.Namespace from train.py — used for dataset, partition,
           clients, rounds, local_epochs, output_dir.
    tcn_result : the dict returned by ``run_simulation`` for PrivFed-TCN.
    """
    print("\n=== Running FL-LSTM Baseline Comparison ===")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client_loaders, test_loader, class_names, class_weights = _build_loaders(args, device)
    # Use the same noise multiplier as the PrivFed-TCN run so privacy
    # budgets match.  Falls back to 0 (non-private upper bound) if the
    # caller didn't pass one.
    noise_multiplier = float(getattr(args, "noise_multiplier", 0.0) or 0.0)
    print(f"[LSTM] DP-SGD noise_multiplier = {noise_multiplier:.4f} "
          f"(0 = non-private upper bound)")

    # -------- PrivFed-TCN summary (taken from existing result) ------------
    from ..model.privfed_tcn import PrivFedTCN
    tcn_template = PrivFedTCN(n_classes=len(class_names))
    tcn_params = tcn_template.num_parameters()
    tcn_size_kb = _model_size_kb(tcn_template)
    tcn_comm_mb = _comm_mb_per_round(tcn_template, args.clients_per_round)
    tcn_per_round = [r.get("val_accuracy", 0.0) for r in tcn_result.get("rounds", [])]
    tcn_final_acc = tcn_result["final_test_metrics"]["accuracy"]
    tcn_final_f1 = tcn_result["final_test_metrics"]["f1_macro"]

    # -------- Run FL-LSTM training -----------------------------------------
    lstm = FLLSTMBaseline(input_dim=config.N_FEATURES,
                          n_classes=len(class_names)).to(device)
    lstm_params = lstm.num_parameters()
    lstm_size_kb = _model_size_kb(lstm)
    lstm_comm_mb = _comm_mb_per_round(lstm, args.clients_per_round)

    n_rounds = args.rounds
    rng = np.random.default_rng(config.SEED)
    per_round_acc: List[float] = []
    per_round_comm: List[float] = []
    per_round_time: List[float] = []
    n_clients = len(client_loaders)
    k = min(args.clients_per_round, n_clients)

    for r in range(1, n_rounds + 1):
        t0 = time.perf_counter()
        chosen = rng.choice(n_clients, size=k, replace=False).tolist()
        loaders = [client_loaders[i] for i in chosen]
        loss = _train_lstm_round(lstm, loaders, device, args.local_epochs,
                                 noise_multiplier=noise_multiplier,
                                 class_weights=class_weights)
        metrics = compute_classification_metrics(lstm, test_loader, device, len(class_names))
        elapsed = time.perf_counter() - t0
        per_round_acc.append(metrics["accuracy"])
        per_round_comm.append(lstm_comm_mb)
        per_round_time.append(elapsed)
        print(f"[LSTM R{r:02d}] acc={metrics['accuracy']:.4f} "
              f"f1={metrics['f1_macro']:.4f} loss={loss:.4f} t={elapsed:.1f}s")

    final_metrics = compute_classification_metrics(lstm, test_loader, device, len(class_names))

    # -------- Build comparison JSON ----------------------------------------
    reduction = (lstm_comm_mb - tcn_comm_mb) / lstm_comm_mb * 100 if lstm_comm_mb > 0 else 0.0
    payload = {
        "privfed_tcn": {
            "param_count": int(tcn_params),
            "model_size_kb": float(tcn_size_kb),
            "comm_mb_per_round": float(tcn_comm_mb),
            "final_accuracy": float(tcn_final_acc),
            "final_f1": float(tcn_final_f1),
            "per_round_accuracy": [float(a) for a in tcn_per_round],
            "per_round_comm_mb": [float(tcn_comm_mb)] * len(tcn_per_round),
        },
        "fl_lstm": {
            "param_count": int(lstm_params),
            "model_size_kb": float(lstm_size_kb),
            "comm_mb_per_round": float(lstm_comm_mb),
            "final_accuracy": float(final_metrics["accuracy"]),
            "final_f1": float(final_metrics["f1_macro"]),
            "per_round_accuracy": [float(a) for a in per_round_acc],
            "per_round_comm_mb": [float(c) for c in per_round_comm],
            "per_round_time_sec": [float(t) for t in per_round_time],
        },
        "comm_reduction_pct": float(reduction),
        "n_rounds": int(n_rounds),
        "n_classes": len(class_names),
        "class_names": class_names,
        "shared_noise_multiplier": float(noise_multiplier),
        "fair_dp_comparison": noise_multiplier > 0,
    }
    out_path = out_dir / "baseline_comparison.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nBaseline comparison saved -> {out_path}")
    print(f"Communication reduction (LSTM -> TCN): {reduction:.2f}%")
    return payload


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # A minimal smoke test that exercises the LSTM training round end-to-end
    # on synthetic data.  Real comparison runs come from train.py.
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1)
    a = parser.parse_args()

    class _A:
        dataset = "synthetic"
        partition = "iid"
        clients = 4
        clients_per_round = 2
        rounds = a.rounds
        local_epochs = 1
        output_dir = "results"
        custom_data_path = None
        limit_samples = None
        ciciot_data_dir = None
        limit_files = None
        limit_rows = None

    fake_tcn = {
        "final_test_metrics": {"accuracy": 0.99, "f1_macro": 0.99},
        "rounds": [{"val_accuracy": 0.5}, {"val_accuracy": 0.99}],
    }
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_baseline_comparison(_A, dev, fake_tcn)
