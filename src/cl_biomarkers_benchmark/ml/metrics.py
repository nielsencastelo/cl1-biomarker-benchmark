from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss
from sklearn.preprocessing import LabelBinarizer


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    labels = np.unique(y_true)
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)
    if y_true_bin.ndim == 1:
        y_true_bin = np.c_[1 - y_true_bin, y_true_bin]

    brier = 0.0
    try:
        brier = float(
            np.mean(
                [
                    brier_score_loss(y_true_bin[:, i], y_proba[:, i])
                    for i in range(y_true_bin.shape[1])
                ]
            )
        )
    except Exception:
        brier = float("nan")

    try:
        ll = float(log_loss(y_true, y_proba, labels=labels))
    except Exception:
        ll = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "log_loss": ll,
        "brier_score": brier,
    }
