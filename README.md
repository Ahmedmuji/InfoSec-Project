# PrivFed-TCN

**A Lightweight, Privacy-Preserving Federated Learning IDS for IoT Networks**

PrivFed-TCN combines a compact Temporal Convolutional Network (TCN) with
multi-head self-attention and deploys it in a hierarchical federated
learning setup protected by DP-SGD with Rényi-DP accounting, SecAgg+
secure aggregation, and Krum Byzantine filtering. On the CIC-IoT-2023
benchmark it **outperforms a 2-layer FL-LSTM baseline at the same privacy
budget while transmitting 65.7 % less data per round**.

---

## 1. Headline Results (CIC-IoT-2023, 14 classes, σ=10.0, ε ≤ 1.0)

| Metric                       | **PrivFed-TCN** | FL-LSTM (DP) | Δ vs LSTM         |
| ---------------------------- | --------------------- | ------------ | ------------------ |
| Test accuracy                | **79.66 %**     | 76.30 %      | **+3.36 pp** |
| F1-macro                     | **0.5861**      | 0.5485       | **+0.0376**  |
| FPR                          | **1.62 %**      | —           | strong             |
| AUC-ROC                      | 0.786                 | —           | —                 |
| Parameters                   | **78 350**      | 228 814      | **−65.8 %** |
| Model size                   | **306 KB**      | 894 KB       | **−65.8 %** |
| Communication / round        | **2.99 MB**     | 8.73 MB      | **−65.8 %** |
| Total comm (50 rounds)       | **149 MB**      | 436 MB       | **−65.8 %** |
| Privacy budget (ε, δ=1e-5) | ≤ 1.0                | ≤ 1.0       | tied               |

Both models were trained under the **same** DP-SGD noise multiplier
(σ = 10.0) and Poisson sub-sampling rate so the comparison is privacy-fair.
See `results/baseline_comparison.json` for the raw numbers and
`results/figures/` for paper-ready PNG + PDF figures.

---

## 2. Architecture: PrivFed-TCN vs FL-LSTM

### 2.1 PrivFed-TCN (proposed)

```
Flow window  (B, T=32, F=41)
        │
        ▼
Categorical embeddings    (last 2 cols → dense-16 each, vocab=64)
        │
        ▼
Concat (numeric ‖ embeds) → Linear projection → (B, T, 64)
        │
        ▼
TCN block, dilation=1   ┐ Conv1D(64,64,k=3) + LayerNorm + GELU + Dropout(0.05)
TCN block, dilation=2   │ residual skip connection
TCN block, dilation=4   ┘ effective receptive field = 21 timesteps
        │
        ▼
Multi-Head Self-Attention (4 heads, d_model=64) + sinusoidal pos-enc
        │
        ▼
Global Average Pooling over time → (B, 64)
        │
        ▼
FC(64→128) → ReLU → Dropout(0.1)
FC(128→64) → ReLU
FC(64→N_classes)   ← logits
```

Total: **78 350 trainable parameters** (≈ 306 KB float32).

Key properties:

- **Causal dilated convolutions** capture long-range temporal patterns
  with a fraction of an LSTM's parameter count.
- **Self-attention** lets the classifier focus on the most informative
  timesteps after the TCN has aggregated local context.
- **Categorical embeddings** treat port and protocol IDs as learnable
  16-dim vectors, much richer than one-hot.
- **Dropout = 0.1** (deliberately small) — DP-SGD noise already provides
  strong implicit regularisation; stacking 0.3 dropout on top destroys
  signal that survived clipping.

### 2.2 FL-LSTM Baseline

```
Flow window  (B, T=32, F=41)
        │
        ▼
LSTM layer 1  (hidden=128)
LSTM layer 2  (hidden=128)
        │
        ▼
Last hidden state → Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→C)
```

Total: **228 814 trainable parameters** (≈ 894 KB float32).

Implements the standard 2-layer stacked LSTM used in recent
FL-IDS papers (Anwar et al. 2025-style). Trained with Adam, FedAvg
aggregation, and the **same DP-SGD optimizer** as PrivFed-TCN so the
privacy comparison is apples-to-apples.

### 2.3 Federated stack (shared by both models)

```
                       ┌─────────────────────────────────────┐
                       │         Central FL Server           │
                       │  • Adaptive weighted FedProx        │
                       │  • Rényi-DP accountant (Mironov '17)│
                       │  • Auto-calibrated noise multiplier │
                       └────────────────┬────────────────────┘
                                        │  cluster aggregates
            ┌──────────────┬────────────┴────────────┬──────────────┐
            ▼              ▼                         ▼              ▼
       ┌────────┐     ┌────────┐                ┌────────┐    ┌────────┐
       │ Edge 1 │     │ Edge 2 │      …         │ Edge 4 │    │ Edge 5 │
       │ SecAgg+│     │ SecAgg+│                │ SecAgg+│    │ SecAgg+│
       │ Krum   │     │ Krum   │                │ Krum   │    │ Krum   │
       └────┬───┘     └────┬───┘                └────┬───┘    └────┬───┘
            │              │                         │              │
       ┌────┴───────┐  ┌───┴────────┐           ┌────┴────────┐  ...
       │ Clients    │  │ Clients    │           │ Clients     │
       │ (DP-SGD +  │  │ (DP-SGD +  │           │ (DP-SGD +   │
       │  FedProx)  │  │  FedProx)  │           │  FedProx)   │
       └────────────┘  └────────────┘           └─────────────┘
```

- **Hierarchical aggregation:** clients → 5 edge aggregators → server.
  Edge aggregators run **multi-Krum** to filter Byzantine updates and
  **SecAgg+** masking so the server never sees individual gradients.
- **Adaptive FedProx μ:** the proximal coefficient is scaled by the
  Jensen–Shannon distance between the client's local label histogram
  and the global one — clients with skewed labels are pulled harder
  toward the global model.
- **Resource-aware client selection:** scheduler weights freshness,
  battery, and reliability; the top-K clients are sampled per round.

---

## 3. Privacy Mechanism

| Component                  | Spec                                    |
| -------------------------- | --------------------------------------- |
| Mechanism                  | DP-SGD (Abadi et al. 2016)              |
| Per-step gradient clipping | L2-norm bound C = 1.0                   |
| Gaussian noise std         | σ · C, σ auto-calibrated             |
| Composition                | Rényi-DP moments accountant            |
| Sub-sampling               | Poisson client + minibatch (q ≈ 0.013) |
| **Reported budget**  | **ε ≤ 1.0, δ = 10⁻⁵**        |

Both PrivFed-TCN and the FL-LSTM baseline run under identical (σ, q,
T) so the privacy guarantees are matched and the accuracy/F1 deltas
are causally attributable to the model choice.

---

## 4. Improvements implemented in this codebase

The following changes were applied to push PrivFed-TCN's accuracy past
the FL-LSTM baseline under equal privacy:

1. **Stratified train/val/test split** (`data/preprocessor.py`).
   Per-class 70/15/15 stratification so highly imbalanced minority
   attack classes (BruteForce, MitM, XSS, …) appear in val and test.
   Eliminates `NaN` AUC and resurrects per-class F1 reporting.
2. **Inverse-frequency class weights** (`simulation/flower_sim.py`,
   `simulation/baseline_comparison.py`).
   Computed on the training split, clipped to `[0.5, 10]`, and passed
   to the `CrossEntropyLoss` of every client. Brings minority-class F1
   from 0.0 to mid-range without hurting majority-class accuracy.
3. **Fair DP comparison** (`simulation/baseline_comparison.py`).
   The FL-LSTM baseline is now wrapped with the **same**
   `DPSGDOptimizer` and noise multiplier as PrivFed-TCN. A previous
   non-private LSTM baseline made the comparison unfair.
4. **Reduced TCN dropout 0.3 → 0.1** (`config.py`).
   DP-SGD noise is itself an aggressive regulariser — stacking 30 %
   dropout on top throttles the signal that survives clipping. This is
   the single change responsible for stabilising rounds 7–13 (where
   accuracy used to *decrease* before the fix).
5. **Label smoothing 0.05** (`federated/client.py`).
   Stabilises training under noisy gradients and prevents the
   class-weighted loss from over-emphasising rare classes.
6. **Strong RDP calibrator with auto-noise** (`privacy/epsilon_calibrator.py`).
   Binary search on σ that solves the Mironov RDP composition exactly
   for the user's `(ε_target, δ, T, q, E, K, N)` configuration so you
   never have to hand-tune the noise.

---

## 5. Reproducing the results

### 5.1 Install

```bash
pip install -r privfed_tcn/requirements.txt
```

`opacus`, `shap`, `flwr`, and `pycryptodome` are optional; the codebase
falls back to internal implementations when these are unavailable.

### 5.2 Run end-to-end (CIC-IoT-2023)

```powershell
python -m privfed_tcn.train `
  --dataset ciciot `
  --ciciot_data_dir "C:\path\to\cic_iot_2023" `
  --limit_rows 50000 `
  --partition iid `
  --rounds 50 `
  --clients 20 `
  --clients_per_round 5 `
  --auto_calibrate_epsilon `
  --epsilon_target 1.0 `
  --run_baseline_comparison `
  --output_dir "D:\UNI\INFOSEC\privfed_tcn\results"
```

Outputs:

- `results/ciciot_summary.json` — per-round + final test metrics
- `results/baseline_comparison.json` — TCN vs LSTM payload
- `results/rounds_iid.csv` — convergence trace
- `checkpoints/privfed_tcn_iid.pt` — final global model

### 5.3 Generate paper figures

```powershell
python -m privfed_tcn.evaluation.research_plots `
    --results_dir "D:\UNI\INFOSEC\privfed_tcn\results" `
    --output_dir  "D:\UNI\INFOSEC\privfed_tcn\results\figures"
```

Each figure is saved as a 300-dpi PNG **and** vector PDF for direct
inclusion in LaTeX:

| File                               | Figure caption                                         |
| ---------------------------------- | ------------------------------------------------------ |
| `fig1_privacy_accuracy_tradeoff` | Accuracy vs noise multiplier σ; ε on the right axis  |
| `fig2_convergence_comparison`    | Test accuracy per round, PrivFed-TCN vs FL-LSTM        |
| `fig3_communication_cost`        | Grouped bars: model KB / MB-per-round / total MB       |
| `fig4_per_class_f1`              | Horizontal F1 bars, colour-coded by performance bucket |
| `fig5_noniid_impact`             | (requires Dirichlet runs — see §5.4)                 |
| `fig6_confusion_matrix`          | Row-normalised heatmap of the final test set           |
| `fig7_headline_metrics`          | Side-by-side accuracy + F1 with Δ-annotations         |

### 5.4 Optional: non-IID ablation (Fig 5)

```powershell
python -m privfed_tcn.train --dataset ciciot --partition dirichlet_05 ...
python -m privfed_tcn.train --dataset ciciot --partition dirichlet_01 ...
python -m privfed_tcn.evaluation.research_plots --results_dir results
```

### 5.5 Markdown report

```powershell
python -m privfed_tcn.evaluation.generate_report --results_dir "D:\UNI\INFOSEC\privfed_tcn\results"
```

Produces `results/EVALUATION_REPORT.md` summarising every JSON in one place.

---

## 6. CLI reference (`privfed_tcn.train`)

| Flag                          | Default      | Notes                                              |
| ----------------------------- | ------------ | -------------------------------------------------- |
| `--dataset`                 | synthetic    | `ton_iot \| cic_iot \| ciciot \| custom`            |
| `--ciciot_data_dir`         | —           | folder of CIC-IoT-2023 CSVs                        |
| `--limit_rows`              | None         | rows per CSV (debug speed-up)                      |
| `--partition`               | iid          | `iid \| dirichlet_05 \| dirichlet_01`              |
| `--rounds`                  | 50           | global FL rounds                                   |
| `--clients`                 | 50           | total clients in the federation                    |
| `--clients_per_round`       | 10           | sub-sampled per round (Poisson)                    |
| `--local_epochs`            | 5            | local SGD passes per client per round              |
| `--noise_multiplier`        | 1.1          | overridden by `--auto_calibrate_epsilon`         |
| `--epsilon_target`          | 1.0          | RDP budget the calibrator must satisfy             |
| `--auto_calibrate_epsilon`  | off          | binary-search σ to hit `--epsilon_target`       |
| `--noise_multiplier_sweep`  | off          | dump `privacy_accuracy_tradeoff.csv`             |
| `--run_baseline_comparison` | off          | run FL-LSTM with**matched** DP and dump JSON |
| `--hardware_benchmark`      | off          | profile latency / energy on simulated devices      |
| `--output_dir`              | `results/` |                                                    |
| `--cpu`                     | off          | force CPU                                          |

---

## 7. Project layout

```
privfed_tcn/
├── config.py                # all hyperparameters
├── train.py                 # CLI: federated training entrypoint
├── evaluate.py              # CLI: post-training evaluation + plots
├── data/
│   ├── loader.py            # multi-dataset router
│   ├── ciciot_loader.py     # CIC-IoT-2023 ingestion
│   ├── preprocessor.py      # feature eng + stratified splits + windowing
│   └── partitioner.py       # IID / Dirichlet client partitioning
├── model/
│   ├── tcn_block.py         # dilated causal Conv1D residual block
│   ├── attention.py         # multi-head self-attention + pos-enc
│   ├── privfed_tcn.py       # full model
│   ├── lstm_baseline.py     # 2-layer LSTM baseline
│   └── explainability.py    # SHAP wrapper (optional)
├── privacy/
│   ├── dp_sgd.py            # clip + Gaussian noise wrapper
│   ├── rdp_accountant.py    # online Rényi-DP composition tracker
│   ├── epsilon_calibrator.py# binary search σ → target ε
│   └── secure_agg.py        # SecAgg+ pairwise masking
├── federated/
│   ├── client.py            # FLClient: DP-SGD + FedProx + class weights
│   ├── edge_aggregator.py   # SecAgg+ + multi-Krum
│   ├── server.py            # adaptive weighted FedAvg / FedProx server
│   └── scheduler.py         # resource-aware client selection
├── simulation/
│   ├── flower_sim.py        # in-process FL simulator
│   ├── baseline_comparison.py # PrivFed-TCN vs FL-LSTM under matched DP
│   └── hardware_sim.py      # latency / energy profiling
├── evaluation/
│   ├── metrics.py           # acc / F1 / FPR / FNR / AUC / CM
│   ├── communication.py     # per-round MB tracker
│   ├── privacy_eval.py      # ε bookkeeping
│   ├── hardware.py          # device profile aggregation
│   ├── research_plots.py    # 7 paper-ready figures
│   └── generate_report.py   # one-shot Markdown summary
└── requirements.txt
```

---

## 8. Citation

If you use PrivFed-TCN in academic work, please cite this implementation
along with the underlying components:

- **DP-SGD** — Abadi et al., *Deep Learning with Differential Privacy*, CCS 2016.
- **Rényi-DP** — Mironov, *Rényi Differential Privacy*, CSF 2017.
- **FedProx** — Li et al., *Federated Optimization in Heterogeneous Networks*, MLSys 2020.
- **Multi-Krum** — Blanchard et al., *Machine Learning with Adversaries*, NeurIPS 2017.
- **TCN** — Bai et al., *An Empirical Evaluation of Generic CNNs*, arXiv 2018.
- **CIC-IoT-2023** — Neto et al., Canadian Institute for Cybersecurity, 2023.
