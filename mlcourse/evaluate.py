"""Metric helpers shared by the lessons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    """Metrics for a binary classifier given predicted probabilities of class 1.

    Threshold-free (rank quality): roc_auc, pr_auc
    Threshold-dependent (decisions): accuracy, precision, recall, f1
    """
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def best_f1_threshold(y_true, proba) -> float:
    """Pick the probability cut-off that maximises F1 (use VALIDATION data, never test)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # precision/recall have one more entry than thresholds; drop the last point.
    return float(thresholds[np.argmax(f1[:-1])])


def show_table(rows: dict[str, dict[str, float]], title: str = "") -> pd.DataFrame:
    table = pd.DataFrame(rows).T.round(3)
    if title:
        print(f"\n{title}")
    print(table.to_string())
    return table


def header(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)
