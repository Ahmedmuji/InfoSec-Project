"""Classification metrics for intrusion detection."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)


def compute_classification_metrics(model: torch.nn.Module, loader, device: torch.device,
                                   n_classes: int) -> Dict[str, float]:
    """Run the model on ``loader`` and return a metrics dict."""
    model.eval()
    ys, yhats, probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            probs.append(p)
            yhats.append(p.argmax(axis=1))
            ys.append(yb.numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(yhats)
    y_prob = np.concatenate(probs, axis=0)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # FPR / FNR averaged one-vs-rest
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    fpr_list, fnr_list = [], []
    total = cm.sum()
    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        fpr_list.append(fp / max(fp + tn, 1))
        fnr_list.append(fn / max(fn + tp, 1))
    fpr = float(np.mean(fpr_list))
    fnr = float(np.mean(fnr_list))

    # AUC-ROC (one-vs-rest, macro)
    try:
        if n_classes == 2:
            auc = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            # Guard against classes missing from y_true in this batch
            present = np.unique(y_true)
            if len(present) < 2:
                auc = float("nan")
            else:
                auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")

    per_class_f1 = f1_score(y_true, y_pred, average=None,
                             labels=list(range(n_classes)), zero_division=0).tolist()

    return {
        "accuracy": acc,
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "fpr": fpr,
        "fnr": fnr,
        "auc_roc": auc,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist()
    }


def confusion_matrix_plot(cm: np.ndarray, class_names: List[str], save_path: str) -> None:
    """Save a heatmap PNG of the confusion matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.7),
                                     max(5, len(class_names) * 0.6)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
