"""Federated simulation loop.

Orchestrates 50 clients across multiple edge aggregators, runs 50 FL
rounds, and logs per-round metrics. If the Flower framework (``flwr``) is
installed the simulation can also be launched via ``flwr.simulation`` —
we expose a ``NumPyClient``-style wrapper for that path, but by default
use a lightweight in-process loop for reproducibility and minimal deps.
"""
from __future__ import annotations

import copy
import os
import time
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .. import config
from ..data.loader import load_dataset
from ..data.preprocessor import Preprocessor, WindowDataset
from ..data.partitioner import partition_iid, partition_dirichlet, make_client_loaders
from ..model.privfed_tcn import PrivFedTCN
from ..federated.client import FLClient, ClientUpdate
from ..federated.edge_aggregator import EdgeAggregator
from ..federated.server import FederatedServer
from ..federated.scheduler import ResourceAwareScheduler, ClientResource
from ..evaluation.metrics import compute_classification_metrics
from ..evaluation.communication import CommunicationTracker, estimate_update_size_mb
from ..evaluation.privacy_eval import PrivacyLogger


# ---------------------------------------------------------------------------
def _build_data(dataset: str, partition: str, n_clients: int, device: torch.device,
                custom_path: str = None, limit_samples: int = None,
                ciciot_data_dir: str = None, limit_files: int = None,
                limit_rows: int = None
                ) -> Tuple[List[DataLoader], DataLoader, DataLoader, List[str], int]:
    df, class_names = load_dataset(
        dataset,
        custom_path=custom_path,
        limit_samples=limit_samples,
        ciciot_data_dir=ciciot_data_dir,
        limit_files=limit_files,
        limit_rows=limit_rows,
    )
    n_classes = len(class_names)

    prep = Preprocessor()
    X, y = prep.fit_transform(df)
    Xtr, ytr, Xval, yval, Xte, yte = prep.split(X, y)

    train_ds = WindowDataset(Xtr, ytr)
    val_ds = WindowDataset(Xval, yval)
    test_ds = WindowDataset(Xte, yte)
    
    pin_mem = torch.cuda.is_available()
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, pin_memory=pin_mem)
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False, pin_memory=pin_mem)

    if partition == "iid":
        parts = partition_iid(ytr, n_clients)
    elif partition == "dirichlet_05":
        parts = partition_dirichlet(ytr, n_clients, alpha=0.5)
    elif partition == "dirichlet_01":
        parts = partition_dirichlet(ytr, n_clients, alpha=0.1)
    else:
        raise ValueError(f"Unknown partition: {partition}")

    client_loaders = make_client_loaders(train_ds, parts, batch_size=config.BATCH_SIZE)

    # Inverse-frequency class weights (capped) computed on the training set
    # so DP-SGD updates don't ignore minority attack classes.
    class_counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    class_counts = np.maximum(class_counts, 1.0)
    weights = class_counts.sum() / (n_classes * class_counts)
    weights = np.clip(weights, 0.5, 50.0)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    return client_loaders, val_loader, test_loader, class_names, n_classes, class_weights


# ---------------------------------------------------------------------------
def _assign_clusters(n_clients: int, n_clusters: int) -> List[List[int]]:
    clusters: List[List[int]] = [[] for _ in range(n_clusters)]
    for cid in range(n_clients):
        clusters[cid % n_clusters].append(cid)
    return clusters


# ---------------------------------------------------------------------------
def run_simulation(dataset: str = "synthetic",
                   partition: str = "iid",
                   n_clients: int = config.N_CLIENTS,
                   n_rounds: int = config.N_ROUNDS,
                   clients_per_round: int = config.CLIENTS_PER_ROUND,
                   local_epochs: int = config.LOCAL_EPOCHS,
                   noise_multiplier: float = config.NOISE_MULTIPLIER,
                   device: torch.device | None = None,
                   output_dir: str = config.RESULTS_PATH,
                   save_model: bool = True,
                   verbose: bool = True,
                   custom_path: str = None,
                   limit_samples: int = None,
                   ciciot_data_dir: str = None,
                   limit_files: int = None,
                   limit_rows: int = None) -> Dict[str, object]:
    """Run a full PrivFed-TCN simulation and return the collected logs."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    client_loaders, val_loader, test_loader, class_names, n_classes, class_weights = _build_data(
        dataset, partition, n_clients, device,
        custom_path=custom_path, limit_samples=limit_samples,
        ciciot_data_dir=ciciot_data_dir, limit_files=limit_files, limit_rows=limit_rows)

    # ------------------------------------------------------------------ model
    global_model = PrivFedTCN(n_classes=n_classes).to(device)

    # ------------------------------------------------------------------ FL entities
    clients: List[FLClient] = []
    for cid, loader in enumerate(client_loaders):
        local_model = PrivFedTCN(n_classes=n_classes)
        local_model.load_state_dict(global_model.state_dict())
        clients.append(FLClient(cid, local_model, loader, device, n_classes,
                                 local_epochs=local_epochs,
                                 noise_multiplier=noise_multiplier,
                                 class_weights=class_weights))

    clusters = _assign_clusters(n_clients, config.N_EDGE_AGGREGATORS)
    edges = [EdgeAggregator(i) for i in range(config.N_EDGE_AGGREGATORS)]

    # Seed resources with small random variation
    rng = np.random.default_rng(config.SEED)
    resources = [ClientResource(
        client_id=i,
        data_freshness=float(rng.uniform(0.6, 1.0)),
        battery=float(rng.uniform(0.4, 1.0)),
        speed=float(rng.uniform(0.5, 1.5)),
        reliability=float(rng.uniform(0.7, 1.0)),
    ) for i in range(n_clients)]
    scheduler = ResourceAwareScheduler(resources, k=clients_per_round)

    server = FederatedServer(global_model, n_classes,
                             noise_multiplier=noise_multiplier,
                             sample_rate=clients_per_round / max(n_clients, 1))
    server.update_global_histogram([c.label_hist for c in clients])

    comm = CommunicationTracker()
    priv_log = PrivacyLogger(server.accountant)

    # ------------------------------------------------------------------ loop
    round_log: List[Dict[str, float]] = []
    update_mb_sample = estimate_update_size_mb(server.get_global_state())

    for rnd in range(1, n_rounds + 1):
        t_round = time.perf_counter()
        selected_ids = scheduler.select()
        global_state = server.get_global_state()

        # --------------------------------------------------- local training
        cluster_updates: Dict[int, List[ClientUpdate]] = {i: [] for i in range(len(clusters))}
        for cid in selected_ids:
            update = clients[cid].train_local(global_state, server.global_hist)
            # Find which cluster this client belongs to
            for k, members in enumerate(clusters):
                if cid in members:
                    cluster_updates[k].append(update)
                    break

        # --------------------------------------------------- edge aggregation
        cluster_deltas, cluster_samples, cluster_losses = [], [], []
        for k, updates in cluster_updates.items():
            if not updates:
                continue
            delta, n_samp, mean_loss = edges[k].aggregate(updates)
            cluster_deltas.append(delta)
            cluster_samples.append(n_samp)
            cluster_losses.append(mean_loss)

        # --------------------------------------------------- server aggregation
        freshness = [float(np.mean([resources[cid].data_freshness
                                     for cid in clusters[k] if cid in selected_ids] or [1.0]))
                     for k, updates in cluster_updates.items() if updates]
        server.aggregate(cluster_deltas, cluster_samples, cluster_losses,
                         freshness=freshness)

        # --------------------------------------------------- metrics
        metrics = compute_classification_metrics(server.model, val_loader, device, n_classes)
        eps_rec = priv_log.log(rnd)
        comm.log_round(rnd, len(selected_ids), update_mb_sample, update_mb_sample)
        elapsed = time.perf_counter() - t_round

        row = {
            "round": rnd,
            "val_accuracy": metrics["accuracy"],
            "val_f1_macro": metrics["f1_macro"],
            "val_fpr": metrics["fpr"],
            "val_fnr": metrics["fnr"],
            "val_auc": metrics["auc_roc"],
            "epsilon": eps_rec["epsilon"],
            "round_comm_mb": comm.rounds[-1]["total_mb"],
            "time_sec": elapsed,
            "n_selected": len(selected_ids),
        }
        round_log.append(row)
        if verbose:
            print(f"[R{rnd:02d}] acc={row['val_accuracy']:.4f} "
                  f"f1={row['val_f1_macro']:.4f} fpr={row['val_fpr']:.4f} "
                  f"eps={row['epsilon']:.3f} commMB={row['round_comm_mb']:.2f} "
                  f"t={elapsed:.1f}s")

    # ------------------------------------------------------------------ final eval
    final_metrics = compute_classification_metrics(server.model, test_loader, device, n_classes)

    # Persist CSV + summary
    df_log = pd.DataFrame(round_log)
    df_log.to_csv(os.path.join(output_dir, f"rounds_{partition}.csv"), index=False)

    if save_model:
        torch.save({
            "model_state_dict": server.model.state_dict(),
            "class_names": class_names,
            "n_classes": n_classes,
            "partition": partition,
            "dataset": dataset,
        }, os.path.join(config.CHECKPOINT_PATH, f"privfed_tcn_{partition}.pt"))

    return {
        "rounds": round_log,
        "final_test_metrics": final_metrics,
        "communication": comm.summary(),
        "privacy": priv_log.current(),
        "class_names": class_names,
    }
