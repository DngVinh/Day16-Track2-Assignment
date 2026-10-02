#!/usr/bin/env python3
"""CPU LightGBM benchmark for the Kaggle credit-card fraud dataset."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="~/ml-benchmark/creditcard.csv",
        help="Path to creditcard.csv",
    )
    parser.add_argument(
        "--output",
        default="~/ml-benchmark/benchmark_result.json",
        help="Path for the JSON result",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=0,
        help=(
            "Enable validation-based early stopping. Zero preserves a fixed-"
            "estimator run and reports the configured iteration limit."
        ),
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.20,
        help="Validation fraction of the training split when early stopping is enabled",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1")
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")
    if args.n_estimators < 1:
        raise ValueError("--n-estimators must be positive")
    if args.n_jobs == 0:
        raise ValueError("--n-jobs cannot be zero")
    if args.early_stopping_rounds < 0:
        raise ValueError("--early-stopping-rounds cannot be negative")
    if args.early_stopping_rounds > 0 and not 0 < args.validation_size < 1:
        raise ValueError("--validation-size must be between 0 and 1 when early stopping is enabled")

    data_path = Path(args.data).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    load_started = time.perf_counter()
    dataframe = pd.read_csv(data_path)
    load_seconds = time.perf_counter() - load_started

    if "Class" not in dataframe.columns:
        raise ValueError("Expected target column 'Class' was not found")

    features = dataframe.drop(columns=["Class"])
    target = dataframe["Class"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=target,
    )

    use_early_stopping = args.early_stopping_rounds > 0
    if use_early_stopping:
        x_fit, x_validation, y_fit, y_validation = train_test_split(
            x_train,
            y_train,
            test_size=args.validation_size,
            random_state=args.random_state,
            stratify=y_train,
        )
    else:
        x_fit, y_fit = x_train, y_train
        x_validation, y_validation = None, None

    negative_count = int((y_fit == 0).sum())
    positive_count = int((y_fit == 1).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)

    model = LGBMClassifier(
        objective="binary",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=31,
        max_depth=-1,
        scale_pos_weight=scale_pos_weight,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbosity=-1,
    )

    train_started = time.perf_counter()
    fit_kwargs = {}
    if use_early_stopping:
        fit_kwargs = {
            "eval_set": [(x_validation, y_validation)],
            "eval_metric": "auc",
            "callbacks": [early_stopping(args.early_stopping_rounds, verbose=False)],
        }
    model.fit(x_fit, y_fit, **fit_kwargs)
    train_seconds = time.perf_counter() - train_started

    native_best_iteration = getattr(model, "best_iteration_", None)
    best_iteration = int(native_best_iteration) if native_best_iteration else args.n_estimators
    probabilities = model.predict_proba(x_test, num_iteration=best_iteration)[:, 1]
    predictions = (probabilities >= args.threshold).astype(int)

    metrics = {
        "load_time_seconds": float(load_seconds),
        "training_time_seconds": float(train_seconds),
        # With early stopping disabled, this is the configured iteration limit,
        # not a validation-tuned best iteration. The JSON records that fact
        # explicitly below so the metric is not overstated.
        "best_iteration": best_iteration,
        "best_iteration_is_tuned": bool(native_best_iteration),
        "n_estimators_used": best_iteration,
        "auc_roc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
    }

    one_row = x_test.iloc[[0]]
    batch = x_test.iloc[: min(1000, len(x_test))]

    for _ in range(5):
        model.predict_proba(one_row, num_iteration=best_iteration)
        model.predict_proba(batch, num_iteration=best_iteration)

    latency_samples_ms = []
    for _ in range(25):
        started = time.perf_counter()
        model.predict_proba(one_row, num_iteration=best_iteration)
        latency_samples_ms.append((time.perf_counter() - started) * 1000)

    throughput_samples = []
    for _ in range(10):
        started = time.perf_counter()
        model.predict_proba(batch, num_iteration=best_iteration)
        elapsed = time.perf_counter() - started
        throughput_samples.append(len(batch) / elapsed)

    metrics.update(
        {
            "inference_latency_1_row_ms": float(np.median(latency_samples_ms)),
            "inference_latency_1_row_mean_ms": float(np.mean(latency_samples_ms)),
            "inference_throughput_1000_rows_per_second": float(np.median(throughput_samples)),
        }
    )

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "dataset": {
            # Keep the artifact portable and avoid leaking the VM's OS Login
            # username through an absolute /home/<user>/ path.
            "path": data_path.name,
            "rows": int(len(dataframe)),
            "features": int(features.shape[1]),
            "fraud_rows": int(target.sum()),
        },
        "split": {
            "test_size": args.test_size,
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "validation_rows": int(len(x_validation)) if use_early_stopping else 0,
            "random_state": args.random_state,
        },
        "model": {
            "name": "LGBMClassifier",
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "scale_pos_weight": float(scale_pos_weight),
            "n_jobs": args.n_jobs,
            "threshold": args.threshold,
            "n_estimators_used": best_iteration,
            "early_stopping_rounds": args.early_stopping_rounds,
            "validation_size": args.validation_size if use_early_stopping else None,
        },
        "metrics": metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"Wrote benchmark result to {output_path}")


if __name__ == "__main__":
    main()
