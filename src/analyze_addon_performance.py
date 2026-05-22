from __future__ import annotations

import csv
import json
import math
import signal
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from datasets import df1, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19
from datasets import df2, df3, df4, df5, df6, df7, df8, df9
from logistic_regression import CAGD, F1Threshold, Gaar, LogisticRegression
from logistic_regression import WeightedErrors
from metrics import BinaryClassImbalanceMetrics, safe_division


DATASET_LOADERS: dict[str, Callable[[], pd.DataFrame]] = {
    "df1": df1,
    "df2": df2,
    "df3": df3,
    "df4": df4,
    "df5": df5,
    "df6": df6,
    "df7": df7,
    "df8": df8,
    "df9": df9,
    "df10": df10,
    "df11": df11,
    "df12": df12,
    "df13": df13,
    "df14": df14,
    "df15": df15,
    "df16": df16,
    "df17": df17,
    "df18": df18,
    "df19": df19,
}

DEFAULT_SEEDS = tuple(range(20))
DEFAULT_OUTPUT_DIR = Path("outputs") / "addon_performance"
RESULT_COLUMNS = [
    "dataset",
    "seed",
    "model",
    "status",
    "rows",
    "features",
    "target_0_count",
    "target_1_count",
    "train_rows",
    "validation_rows",
    "test_rows",
    "train_positive_count",
    "validation_positive_count",
    "test_positive_count",
    "learning_rate",
    "iterations",
    "threshold",
    "threshold_score",
    "training_time_seconds",
    "prediction_time_seconds",
    "accuracy",
    "majority_baseline_accuracy",
    "accuracy_gain_over_majority",
    "balanced_accuracy",
    "majority_recall",
    "minority_recall",
    "minority_precision",
    "minority_f1",
    "minority_error_rate",
    "false_positive_rate",
    "mcc",
    "roc_auc",
    "average_precision",
    "tn",
    "fp",
    "fn",
    "tp",
]
ERROR_COLUMNS = [
    "dataset",
    "seed",
    "model",
    "stage",
    "error_type",
    "error_message",
    "traceback",
]
PRIMARY_METRICS = [
    "balanced_accuracy",
    "minority_f1",
    "minority_recall",
    "average_precision",
    "roc_auc",
    "accuracy_gain_over_majority",
]


@dataclass(frozen=True)
class AnalysisConfig:
    learning_rate: float
    iterations: int
    gaar_lambda: float
    gaar_c_minority: float
    focal_gamma: float
    focal_alpha: float
    cagd_alpha: float
    cagd_epsilon: float
    cagd_approximation: str


class Timeout:
    def __init__(self, seconds: int | None):
        self.seconds = seconds
        self.previous_handler = None

    def __enter__(self):
        if not self.seconds:
            return self

        self.previous_handler = signal.signal(signal.SIGALRM, self._raise_timeout)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.seconds:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.previous_handler)
        return False

    def _raise_timeout(self, signum, frame):
        raise TimeoutError(f"Run exceeded {self.seconds} seconds")


def build_models(config: AnalysisConfig) -> dict[str, Callable[[], LogisticRegression]]:
    def model(addons=None):
        return LogisticRegression(
            learning_rate=config.learning_rate,
            iterations=config.iterations,
            addons=addons,
        )

    return {
        "default": lambda: model(),
        "gaar": lambda: model(
            [Gaar(lambda_=config.gaar_lambda, c_minority=config.gaar_c_minority)]
        ),
        "focal_loss": lambda: model(
            [WeightedErrors(gamma=config.focal_gamma, alpha=config.focal_alpha)]
        ),
        "f1_threshold": lambda: model([F1Threshold()]),
        "gaar_focal_loss": lambda: model(
            [
                WeightedErrors(gamma=config.focal_gamma, alpha=config.focal_alpha),
                Gaar(lambda_=config.gaar_lambda, c_minority=config.gaar_c_minority),
            ]
        ),
        "gaar_f1_threshold": lambda: model(
            [
                Gaar(lambda_=config.gaar_lambda, c_minority=config.gaar_c_minority),
                F1Threshold(),
            ]
        ),
        "focal_loss_f1_threshold": lambda: model(
            [
                WeightedErrors(gamma=config.focal_gamma, alpha=config.focal_alpha),
                F1Threshold(),
            ]
        ),
        "gaar_focal_loss_f1_threshold": lambda: model(
            [
                WeightedErrors(gamma=config.focal_gamma, alpha=config.focal_alpha),
                Gaar(lambda_=config.gaar_lambda, c_minority=config.gaar_c_minority),
                F1Threshold(),
            ]
        ),
        "cagd": lambda: model(
            [
                CAGD(
                    alpha=config.cagd_alpha,
                    epsilon=config.cagd_epsilon,
                    approximation=config.cagd_approximation,
                )
            ]
        ),
        "gaar_cagd": lambda: model(
            [
                Gaar(lambda_=config.gaar_lambda, c_minority=config.gaar_c_minority),
                CAGD(
                    alpha=config.cagd_alpha,
                    epsilon=config.cagd_epsilon,
                    approximation=config.cagd_approximation,
                ),
            ]
        ),
        "focal_loss_cagd": lambda: model(
            [
                WeightedErrors(gamma=config.focal_gamma, alpha=config.focal_alpha),
                CAGD(
                    alpha=config.cagd_alpha,
                    epsilon=config.cagd_epsilon,
                    approximation=config.cagd_approximation,
                ),
            ]
        ),
        "cagd_f1_threshold": lambda: model(
            [
                CAGD(
                    alpha=config.cagd_alpha,
                    epsilon=config.cagd_epsilon,
                    approximation=config.cagd_approximation,
                ),
                F1Threshold(),
            ]
        ),
        "gaar_focal_loss_cagd": lambda: model(
            [
                WeightedErrors(gamma=config.focal_gamma, alpha=config.focal_alpha),
                Gaar(lambda_=config.gaar_lambda, c_minority=config.gaar_c_minority),
                CAGD(
                    alpha=config.cagd_alpha,
                    epsilon=config.cagd_epsilon,
                    approximation=config.cagd_approximation,
                ),
            ]
        ),
        "gaar_cagd_f1_threshold": lambda: model(
            [
                Gaar(lambda_=config.gaar_lambda, c_minority=config.gaar_c_minority),
                CAGD(
                    alpha=config.cagd_alpha,
                    epsilon=config.cagd_epsilon,
                    approximation=config.cagd_approximation,
                ),
                F1Threshold(),
            ]
        ),
        "focal_loss_cagd_f1_threshold": lambda: model(
            [
                WeightedErrors(gamma=config.focal_gamma, alpha=config.focal_alpha),
                CAGD(
                    alpha=config.cagd_alpha,
                    epsilon=config.cagd_epsilon,
                    approximation=config.cagd_approximation,
                ),
                F1Threshold(),
            ]
        ),
        "gaar_focal_loss_cagd_f1_threshold": lambda: model(
            [
                WeightedErrors(gamma=config.focal_gamma, alpha=config.focal_alpha),
                Gaar(lambda_=config.gaar_lambda, c_minority=config.gaar_c_minority),
                CAGD(
                    alpha=config.cagd_alpha,
                    epsilon=config.cagd_epsilon,
                    approximation=config.cagd_approximation,
                ),
                F1Threshold(),
            ]
        ),
    }


def split_dataset(df: pd.DataFrame, seed: int) -> dict[str, np.ndarray | pd.DataFrame]:
    train = df.sample(frac=0.8, random_state=seed)
    test = df.drop(train.index)
    validation = train.sample(frac=0.2, random_state=seed)
    train = train.drop(validation.index)

    return {
        "train": train,
        "validation": validation,
        "test": test,
        "X_train": train.drop(columns=["target"]).values.astype(float),
        "y_train": train["target"].values.astype(float),
        "X_val": validation.drop(columns=["target"]).values.astype(float),
        "y_val": validation["target"].values.astype(float),
        "X_test": test.drop(columns=["target"]).values.astype(float),
        "y_test": test["target"].values.astype(float),
    }


def extra_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    binary_metrics = BinaryClassImbalanceMetrics(y_true, y_pred)
    tn, fp, fn, tp = (
        binary_metrics.tn,
        binary_metrics.fp,
        binary_metrics.fn,
        binary_metrics.tp,
    )
    summary = binary_metrics.summary()
    summary.update(
        {
            "majority_recall": binary_metrics.majority_recall(),
            "minority_precision": safe_division(tp, tp + fp),
            "minority_f1": safe_division(2 * tp, 2 * tp + fp + fn),
            "false_positive_rate": safe_division(fp, fp + tn),
            "mcc": matthews_correlation(tn, fp, fn, tp),
            "roc_auc": roc_auc_score(y_true, y_score),
            "average_precision": average_precision_score(y_true, y_score),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
    )
    return summary


def matthews_correlation(tn: int, fp: int, fn: int, tp: int) -> float:
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return safe_division(tp * tn - fp * fn, denominator)


def roc_auc_score(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    positive_count = int(np.sum(y_true == 1))
    negative_count = int(np.sum(y_true == 0))
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=float)
    index = 0
    while index < len(y_score):
        end = index + 1
        while end < len(y_score) and sorted_scores[end] == sorted_scores[index]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2
        index = end

    positive_rank_sum = float(np.sum(ranks[y_true == 1]))
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def average_precision_score(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    positive_count = int(np.sum(y_true == 1))
    if positive_count == 0:
        return float("nan")

    order = np.argsort(-y_score)
    sorted_true = y_true[order]
    true_positives = np.cumsum(sorted_true == 1)
    precision_at_k = true_positives / (np.arange(len(sorted_true)) + 1)
    return float(np.sum(precision_at_k[sorted_true == 1]) / positive_count)


def append_csv(path: Path, row: dict, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow({column: row.get(column) for column in columns})
        file.flush()


def load_completed_keys(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()

    results = pd.read_csv(path)
    if results.empty:
        return set()

    return {
        (str(row.dataset), int(row.seed), str(row.model))
        for row in results.itertuples()
        if row.status == "ok"
    }


def run_single_model(
    dataset_name: str,
    seed: int,
    model_name: str,
    model_factory: Callable[[], LogisticRegression],
    df: pd.DataFrame,
    splits: dict[str, np.ndarray | pd.DataFrame],
    config: AnalysisConfig,
    timeout_seconds: int | None,
) -> dict:
    with Timeout(timeout_seconds):
        model = model_factory()
        start = time.perf_counter()
        model.fit(
            splits["X_train"],
            splits["y_train"],
            splits["X_val"],
            splits["y_val"],
        )
        training_time = time.perf_counter() - start

        start = time.perf_counter()
        y_score = model.predict_proba(splits["X_test"])
        y_pred = model.predict(splits["X_test"])
        prediction_time = time.perf_counter() - start

    if not np.all(np.isfinite(y_score)):
        raise FloatingPointError("Model produced non-finite prediction probabilities")

    metrics = extra_metrics(splits["y_test"], y_pred, y_score)
    target_counts = df["target"].value_counts().to_dict()
    return {
        "dataset": dataset_name,
        "seed": seed,
        "model": model_name,
        "status": "ok",
        "rows": len(df),
        "features": df.shape[1] - 1,
        "target_0_count": int(target_counts.get(0, 0)),
        "target_1_count": int(target_counts.get(1, 0)),
        "train_rows": len(splits["train"]),
        "validation_rows": len(splits["validation"]),
        "test_rows": len(splits["test"]),
        "train_positive_count": int(np.sum(splits["y_train"] == 1)),
        "validation_positive_count": int(np.sum(splits["y_val"] == 1)),
        "test_positive_count": int(np.sum(splits["y_test"] == 1)),
        "learning_rate": config.learning_rate,
        "iterations": config.iterations,
        "threshold": model.threshold,
        "threshold_score": model.threshold_score,
        "training_time_seconds": training_time,
        "prediction_time_seconds": prediction_time,
        **metrics,
    }


def run_experiment(
    output_dir: Path,
    config: AnalysisConfig,
    datasets: list[str],
    models: list[str],
    seeds: list[int],
    timeout_seconds: int | None = None,
    resume: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "raw_results.csv"
    errors_path = output_dir / "errors.csv"
    config_path = output_dir / "config.json"
    if not resume:
        reset_output_files(output_dir)

    config_path.write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "datasets": datasets,
                "models": models,
                "seeds": seeds,
                "timeout_seconds": timeout_seconds,
                "resume": resume,
                "analysis_config": config.__dict__,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    model_factories = build_models(config)
    completed = load_completed_keys(results_path) if resume else set()

    total_runs = len(datasets) * len(seeds) * len(models)
    run_index = 0
    for dataset_name in datasets:
        try:
            df = DATASET_LOADERS[dataset_name]()
            validate_dataset(df, dataset_name)
        except Exception as exc:
            append_error(errors_path, dataset_name, None, None, "load_dataset", exc)
            continue

        for seed in seeds:
            try:
                splits = split_dataset(df, seed)
            except Exception as exc:
                append_error(
                    errors_path, dataset_name, seed, None, "split_dataset", exc
                )
                continue

            for model_name in models:
                run_index += 1
                key = (dataset_name, seed, model_name)
                if key in completed:
                    print(f"[{run_index}/{total_runs}] skip {key}")
                    continue

                print(f"[{run_index}/{total_runs}] run {key}", flush=True)
                try:
                    result = run_single_model(
                        dataset_name,
                        seed,
                        model_name,
                        model_factories[model_name],
                        df,
                        splits,
                        config,
                        timeout_seconds,
                    )
                    append_csv(results_path, result, RESULT_COLUMNS)
                except Exception as exc:
                    append_error(
                        errors_path, dataset_name, seed, model_name, "fit", exc
                    )

    create_analysis_outputs(output_dir, results_path, config)


def reset_output_files(output_dir: Path) -> None:
    filenames = [
        "raw_results.csv",
        "errors.csv",
        "config.json",
        "model_summary.csv",
        "dataset_summary.csv",
        "paired_tests_vs_default.csv",
        "analysis_artifact.pkl",
        "analysis_report.md",
    ]
    for filename in filenames:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def validate_dataset(df: pd.DataFrame, dataset_name: str) -> None:
    if "target" not in df.columns:
        raise ValueError(f"{dataset_name} is missing a target column")

    target_values = set(df["target"].dropna().unique())
    if target_values != {0, 1}:
        raise ValueError(f"{dataset_name} target must contain exactly 0 and 1")

    non_numeric = [
        column
        for column in df.columns
        if column != "target" and not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        raise ValueError(f"{dataset_name} has non-numeric features: {non_numeric}")


def append_error(
    errors_path: Path,
    dataset_name: str,
    seed: int | None,
    model_name: str | None,
    stage: str,
    exc: Exception,
) -> None:
    row = {
        "dataset": dataset_name,
        "seed": seed,
        "model": model_name,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }
    append_csv(errors_path, row, ERROR_COLUMNS)
    print(
        f"ERROR dataset={dataset_name} seed={seed} model={model_name}: {exc}",
        flush=True,
    )


def create_analysis_outputs(
    output_dir: Path,
    results_path: Path,
    config: AnalysisConfig,
) -> None:
    if not results_path.exists():
        return

    results = pd.read_csv(results_path)
    if results.empty:
        return

    model_summary = summarize_models(results)
    dataset_summary = summarize_datasets(results)

    model_summary.to_csv(output_dir / "model_summary.csv", index=False)
    dataset_summary.to_csv(output_dir / "dataset_summary.csv", index=False)


def summarize_models(results: pd.DataFrame) -> pd.DataFrame:
    aggregations = {}
    for metric in PRIMARY_METRICS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_median"] = (metric, "median")
        aggregations[f"{metric}_std"] = (metric, "std")

    aggregations["mean_training_time_seconds"] = ("training_time_seconds", "mean")
    aggregations["runs"] = ("status", "count")
    return (
        results.groupby("model", as_index=False)
        .agg(**aggregations)
        .sort_values("balanced_accuracy_mean", ascending=False)
    )


def summarize_datasets(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_name, dataset_results in results.groupby("dataset"):
        row = {"dataset": dataset_name}
        for metric in PRIMARY_METRICS:
            means = dataset_results.groupby("model")[metric].mean()
            row[f"best_{metric}_model"] = means.idxmax()
            row[f"best_{metric}"] = means.max()
            row[f"default_{metric}"] = means.get("default", float("nan"))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("dataset")


def default_config() -> AnalysisConfig:
    return AnalysisConfig(
        learning_rate=0.01,
        iterations=1000,
        gaar_lambda=0.01,
        gaar_c_minority=25.0,
        focal_gamma=2.0,
        focal_alpha=0.5,
        cagd_alpha=5.0,
        cagd_epsilon=1e-2,
        cagd_approximation="diagonal",
    )


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR
    config = default_config()
    datasets = list(DATASET_LOADERS)
    models = list(build_models(config))
    seeds = list(DEFAULT_SEEDS)

    run_experiment(
        output_dir=output_dir,
        config=config,
        datasets=datasets,
        models=models,
        seeds=seeds,
        timeout_seconds=None,
        resume=False,
    )


if __name__ == "__main__":
    main()
