from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cl_biomarkers_benchmark.ml.metrics import classification_metrics


@dataclass
class ModelRunResult:
    name: str
    metrics: dict[str, float]
    oof_pred: np.ndarray
    oof_proba: np.ndarray
    fit_time_sec: float
    predict_time_sec: float


def build_models(seed: int = 42) -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=2000, random_state=seed),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        "mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        max_iter=1000,
                        random_state=seed,
                        early_stopping=True,
                    ),
                ),
            ]
        ),
    }


def run_cv(df: pd.DataFrame, cfg: dict[str, Any]) -> list[ModelRunResult]:
    label_col = "label"
    feature_cols = [c for c in df.columns if c not in {"sample_id", label_col}]
    X = df[feature_cols].fillna(0.0).to_numpy(dtype=float)
    y_raw = df[label_col].to_numpy()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    seed = int(cfg.get("seed", 42))
    n_splits = int(cfg["ml"]["n_splits"])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    models = build_models(seed=seed)
    enabled = {k for k, v in cfg["ml"]["models"].items() if v}

    results: list[ModelRunResult] = []
    for name, model in models.items():
        if name not in enabled:
            continue
        oof_pred = np.empty(shape=(len(df),), dtype=object)
        oof_proba = np.zeros((len(df), len(np.unique(y))), dtype=float)
        fit_time = 0.0
        pred_time = 0.0

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]

            t0 = perf_counter()
            model.fit(X_train, y_train)
            fit_time += perf_counter() - t0

            t0 = perf_counter()
            oof_pred[test_idx] = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                oof_proba[test_idx] = model.predict_proba(X_test)
            pred_time += perf_counter() - t0

        metrics = classification_metrics(y_true=y, y_pred=oof_pred.astype(int), y_proba=oof_proba)
        results.append(
            ModelRunResult(
                name=name,
                metrics=metrics,
                oof_pred=oof_pred,
                oof_proba=oof_proba,
                fit_time_sec=fit_time,
                predict_time_sec=pred_time / max(1, len(df)),
            )
        )
    return results
