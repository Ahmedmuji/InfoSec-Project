"""End-to-end federated training entry point for PrivFed-TCN."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch

from . import config
from .simulation.flower_sim import run_simulation
from .simulation.hardware_sim import profile_on_devices
from .privacy.epsilon_calibrator import calibrate_noise_multiplier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PrivFed-TCN federated IDS")
    p.add_argument("--dataset", choices=["ton_iot", "cic_iot", "ciciot", "synthetic", "custom"], default="synthetic")
    p.add_argument("--custom_data_path", type=str, default=None, help="Path to custom CSV if dataset=custom")
    p.add_argument("--limit_samples", type=int, default=None, help="Limit rows loaded from custom dataset")
    p.add_argument("--ciciot_data_dir", type=str, default=None, help="Directory with CIC-IoT-2023 CSVs (dataset=ciciot)")
    p.add_argument("--limit_files", type=int, default=None, help="Limit CIC-IoT CSV files loaded")
    p.add_argument("--limit_rows", type=int, default=None, help="Limit CIC-IoT rows per file")
    p.add_argument("--partition", choices=["iid", "dirichlet_05", "dirichlet_01"], default="iid")
    p.add_argument("--rounds", type=int, default=config.N_ROUNDS)
    p.add_argument("--clients", type=int, default=config.N_CLIENTS)
    p.add_argument("--clients_per_round", type=int, default=config.CLIENTS_PER_ROUND)
    p.add_argument("--local_epochs", type=int, default=config.LOCAL_EPOCHS)
    p.add_argument("--noise_multiplier", type=float, default=config.NOISE_MULTIPLIER)
    p.add_argument("--epsilon_target", type=float, default=config.TARGET_EPSILON)
    p.add_argument("--hardware_benchmark", action="store_true")
    p.add_argument("--output_dir", default=config.RESULTS_PATH)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    p.add_argument("--auto_calibrate_epsilon", action="store_true",
                   help="Auto-calibrate noise_multiplier so cumulative epsilon <= epsilon_target")
    p.add_argument("--noise_multiplier_sweep", action="store_true",
                   help="Sweep sigma in [1.1, 2.0, 3.0, 4.0] and dump privacy_accuracy_tradeoff.csv")
    p.add_argument("--run_baseline_comparison", action="store_true",
                   help="After PrivFed-TCN training, run FL-LSTM baseline and emit baseline_comparison.json")
    return p.parse_args()


def _run_one(args, device, noise_multiplier, summary_filename):
    """Helper to run a single FL simulation and dump a summary JSON.

    Returns the simulation ``result`` dict (also persisted to disk).
    """
    result = run_simulation(
        dataset=args.dataset,
        partition=args.partition,
        n_clients=args.clients,
        n_rounds=args.rounds,
        clients_per_round=args.clients_per_round,
        local_epochs=args.local_epochs,
        noise_multiplier=noise_multiplier,
        device=device,
        output_dir=args.output_dir,
        custom_path=args.custom_data_path,
        limit_samples=args.limit_samples,
        ciciot_data_dir=args.ciciot_data_dir,
        limit_files=args.limit_files,
        limit_rows=args.limit_rows,
    )
    summary_path = Path(args.output_dir) / summary_filename
    payload = {
        "final_test_metrics": result["final_test_metrics"],
        "communication": result["communication"],
        "privacy": result["privacy"],
        "class_names": result["class_names"],
        "per_round": result.get("rounds", []),
        "args": {**vars(args), "effective_noise_multiplier": noise_multiplier},
    }
    with summary_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Saved summary -> {summary_path}")
    return result, payload


def _do_sweep(args, device):
    """Sweep noise_multiplier and write privacy_accuracy_tradeoff.csv.

    Saves one summary JSON per sigma so partial progress survives crashes.
    """
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "privacy_accuracy_tradeoff.csv"

    fieldnames = ["noise_multiplier", "epsilon", "accuracy", "f1_macro", "fpr"]
    if not csv_path.exists():
        with csv_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for sigma in [1.1, 2.0, 3.0, 4.0]:
        print(f"\n=== Sweep sigma={sigma} ===")
        _, payload = _run_one(args, device, sigma,
                              f"summary_sweep_sigma_{sigma:.2f}.json")
        row = {
            "noise_multiplier": sigma,
            "epsilon": payload["privacy"].get("epsilon", float("nan")),
            "accuracy": payload["final_test_metrics"]["accuracy"],
            "f1_macro": payload["final_test_metrics"]["f1_macro"],
            "fpr": payload["final_test_metrics"]["fpr"],
        }
        # Append the row immediately so we never lose progress.
        with csv_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
        print(f"  -> recorded {row}")
    print(f"\nSweep complete. CSV: {csv_path}")


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")

    # Optional: auto-calibrate noise_multiplier to satisfy epsilon_target.
    if args.auto_calibrate_epsilon:
        # Estimate per-client dataset size from actual data parameters.
        # With limit_rows * limit_files spread across n_clients, each client
        # gets roughly total_rows * 0.7 (train split) / n_clients samples.
        est_total = (args.limit_rows or 50_000) * max(1, args.limit_files or 10)
        est_per_client = max(1000, int(est_total * 0.7 / args.clients))
        sigma = calibrate_noise_multiplier(
            target_epsilon=args.epsilon_target,
            target_delta=config.TARGET_DELTA,
            n_rounds=args.rounds,
            n_clients=args.clients,
            clients_per_round=args.clients_per_round,
            local_epochs=args.local_epochs,
            batch_size=config.BATCH_SIZE,
            dataset_size_per_client=est_per_client,
        )
        # Warn if sigma is very high — accuracy will suffer.
        if sigma > 5.0:
            print(f"WARNING: calibrated sigma={sigma:.2f} is very high. "
                  f"Consider relaxing --epsilon_target (currently {args.epsilon_target}) "
                  f"or reducing --rounds to improve accuracy.")
        print(f"Auto-calibrated noise_multiplier = {sigma:.4f} "
              f"for target epsilon = {args.epsilon_target} "
              f"(est. {est_per_client} samples/client)")
        args.noise_multiplier = sigma

    print("=== PrivFed-TCN ===")
    print(f"Dataset       : {args.dataset}")
    print(f"Partition     : {args.partition}")
    print(f"Clients       : {args.clients} (K={args.clients_per_round}/round)")
    print(f"Rounds        : {args.rounds}")
    print(f"Noise sigma   : {args.noise_multiplier}")
    print(f"Target epsilon: {args.epsilon_target}")
    print(f"Device        : {device}")

    # Sweep mode: run several sigmas and emit a CSV, then exit.
    if args.noise_multiplier_sweep:
        _do_sweep(args, device)
        return

    # Pick the appropriate summary filename for the dataset.
    if args.dataset == "ciciot":
        summary_filename = "ciciot_summary.json"
    else:
        summary_filename = f"summary_{args.partition}.json"

    result, _payload = _run_one(args, device, args.noise_multiplier, summary_filename)

    if args.hardware_benchmark:
        hw = profile_on_devices(n_classes=len(result["class_names"]))
        with (Path(args.output_dir) / "hardware.json").open("w") as f:
            json.dump(hw, f, indent=2)
        for r in hw:
            print(f"[HW] {r['simulated_device']}: "
                  f"latency={r['latency_ms']:.2f}ms "
                  f"energy={r['energy_mj_per_inference']:.2f}mJ "
                  f"params={r['param_count']} size={r['model_size_kb']:.1f}KB")

    if args.run_baseline_comparison:
        # Lazy import so the rest of the file works even if the baseline
        # module has not been generated yet.
        from .simulation.baseline_comparison import run_baseline_comparison
        run_baseline_comparison(
            args=args,
            device=device,
            tcn_result=result,
        )


if __name__ == "__main__":
    main()
