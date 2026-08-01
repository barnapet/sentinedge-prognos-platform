"""MLflow instrumentation for the M3 LOEO comparisons (Issue #74).

`docs/PRD.md` Section 11 defines M3 as "trained, tracked in MLflow, evaluated honestly",
and Section 10's acceptance criteria require at least one visible MLflow run showing a
trained model's metrics and parameters. Neither #21 (`compare_imbalance.py`) nor #72
(`train_baseline.py`) logged to MLflow when they ran -- this module adds that instrumentation
without changing either harness. It imports and re-runs the exact functions those modules'
own "Reproducing" sections already name (`python -m src.training.compare_imbalance` /
`python -m src.training.train_baseline`), so what gets logged is what produced
`docs/class_imbalance_decision.md` and `docs/model_training_decision.md`'s numbers, not a
re-derivation of them.

Two MLflow experiments, matching Issue #74's Task 2 scope exactly:

- `m3-imbalance-comparison` (#21) -- one run per strategy in `imbalance.STRATEGIES`, all on
  the full M2 feature set. Reproduces every row of
  `docs/class_imbalance_decision.md` Section 3.
- `m3-baseline-ablation` (#72) -- one run per feature-set configuration, restricted to
  `full` and `no_rms_ratio` -- Issue #74's Task 2 names these two explicitly as "the two
  #72 configurations (baseline, ablation)". `no_raw_rms` and `kurtosis_skewness_only` are
  #72's own diagnostic side-runs, not part of what #74 asks to be tracked; they stay
  reproducible via `python -m src.training.train_baseline` directly, unlogged.

See `docs/mlflow_tracking.md` for the dependency choice (`mlflow-skinny` + `sqlalchemy` +
`alembic` over the full `mlflow` package) and how to inspect the results.
"""
from __future__ import annotations

from pathlib import Path

import mlflow

from src.labeling import LABELS
from src.training import compare_imbalance, train_baseline
from src.training.evaluation import FEATURE_MATRIX_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[2]

# SQLite rather than the plain `file:./mlruns` backend -- see `docs/mlflow_tracking.md`
# for why. Both `mlflow.db` and `mlartifacts/` are gitignored (Issue #74 Task 4).
TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
ARTIFACT_ROOT = (REPO_ROOT / "mlartifacts").as_uri()

EXPERIMENT_IMBALANCE = "m3-imbalance-comparison"
EXPERIMENT_ABLATION = "m3-baseline-ablation"

# `docs/class_imbalance_decision.md` Section 3's published numbers are on the unablated
# feature set -- `compare_imbalance.run_comparison()`'s default.
IMBALANCE_FEATURE_SET = "full"
ABLATION_CONFIGS = ("full", "no_rms_ratio")
# #21's adopted default, reused unchanged by #72 (`train_baseline.BASELINE_STRATEGY`).
IMBALANCE_STRATEGY_FOR_ABLATION = "class_weight_balanced"


def configure_tracking() -> None:
    """Point MLflow at this repo's local, file-backed store (`docs/PRD.md` Section 8)."""
    mlflow.set_tracking_uri(TRACKING_URI)


def _ensure_experiment(name: str) -> str:
    """Get or create an experiment, with its artifacts under `mlartifacts/`."""
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(name, artifact_location=ARTIFACT_ROOT)


def _log_fold_metrics(fold: dict) -> None:
    """Log one fold's per-class recall/precision/support and macro-F1, suffixed by the
    held-out experiment.

    Suffixing by experiment name, rather than using MLflow's `step` axis, because the
    three folds are not steps of one training run -- they are three independent held-out
    experiments (`docs/evaluation_protocol.md` Section 1), and this way each metric name
    is directly comparable across runs in the MLflow UI's table/chart view.
    """
    held_out = fold["held_out"]
    for label in LABELS:
        mlflow.log_metric(f"recall_{label}_{held_out}", fold["recall"][label])
        mlflow.log_metric(f"precision_{label}_{held_out}", fold["precision"][label])
        mlflow.log_metric(f"support_{label}_{held_out}", fold["support"][label])
    mlflow.log_metric(f"macro_f1_{held_out}", fold["macro_f1"])


def _log_aggregate_metrics(agg: dict, name: str) -> None:
    """Log one metric's mean/min/max/range, matching `evaluation.aggregate`'s four fields
    (`docs/evaluation_protocol.md` Section 5 -- no standard deviation)."""
    for stat, value in agg.items():
        mlflow.log_metric(f"{name}_{stat}", value)


def _log_confusion_matrices(per_fold: list[dict]) -> None:
    mlflow.log_dict(
        {fold["held_out"]: fold["confusion_matrix"] for fold in per_fold},
        "confusion_matrices.json",
    )


def log_imbalance_comparison(results: dict | None = None) -> dict[str, str]:
    """Log #21's five-arm comparison to MLflow, one run per strategy.

    `results` is injectable so tests can log a small synthetic comparison without needing
    `data/processed/training_dataset.parquet`; the default recomputes the real thing via
    `compare_imbalance.run_comparison()`, unchanged.

    Logs to whatever `mlflow.set_tracking_uri` currently points at -- callers decide that
    (`main` below calls `configure_tracking()` first); this function does not, so tests can
    point it at an isolated store without its own default silently overriding them.
    """
    results = compare_imbalance.run_comparison() if results is None else results
    experiment_id = _ensure_experiment(EXPERIMENT_IMBALANCE)

    run_ids = {}
    for strategy_name, result in results.items():
        with mlflow.start_run(experiment_id=experiment_id, run_name=strategy_name) as run:
            mlflow.set_tags({"issue": "21", "component": "imbalance_comparison"})
            mlflow.log_params(
                {
                    "strategy": strategy_name,
                    "description": result["description"],
                    "feature_set": IMBALANCE_FEATURE_SET,
                    "feature_columns": ",".join(FEATURE_MATRIX_COLUMNS),
                    "n_folds": len(result["per_fold"]),
                }
            )
            for fold in result["per_fold"]:
                _log_fold_metrics(fold)
            _log_aggregate_metrics(result["aggregate"]["critical_recall"], "critical_recall")
            _log_aggregate_metrics(result["aggregate"]["critical_precision"], "critical_precision")
            _log_aggregate_metrics(result["aggregate"]["macro_f1"], "macro_f1")
            _log_confusion_matrices(result["per_fold"])
            run_ids[strategy_name] = run.info.run_id
    return run_ids


def log_baseline_ablation(
    configs: tuple[str, ...] = ABLATION_CONFIGS,
    results: dict | None = None,
) -> dict[str, str]:
    """Log #72's baseline + `rms_ratio` ablation to MLflow, one run per feature-set config.

    `results` is injectable for the same reason as `log_imbalance_comparison`. The default
    loads the real dataset once and scores only `configs` (default: `full`, `no_rms_ratio`)
    via `train_baseline.run_all_feature_sets`, unchanged. Logs to whatever
    `mlflow.set_tracking_uri` currently points at -- see `log_imbalance_comparison`.
    """
    if results is None:
        feature_sets = {name: train_baseline.FEATURE_SETS[name] for name in configs}
        results = train_baseline.run_all_feature_sets(feature_sets=feature_sets)
    experiment_id = _ensure_experiment(EXPERIMENT_ABLATION)

    run_ids = {}
    for config_name, result in results.items():
        with mlflow.start_run(experiment_id=experiment_id, run_name=config_name) as run:
            mlflow.set_tags({"issue": "72", "component": "baseline_ablation"})
            mlflow.log_params(
                {
                    "feature_set": config_name,
                    "feature_columns": ",".join(result["feature_columns"]),
                    "imbalance_strategy": IMBALANCE_STRATEGY_FOR_ABLATION,
                    "n_folds": len(result["per_fold"]),
                }
            )
            for fold in result["per_fold"]:
                _log_fold_metrics(fold)
            for metric_name in ("critical_recall", "critical_precision", "macro_f1", "normal_recall"):
                _log_aggregate_metrics(result["aggregate"][metric_name], metric_name)
            _log_confusion_matrices(result["per_fold"])
            run_ids[config_name] = run.info.run_id
    return run_ids


def main() -> None:
    configure_tracking()
    imbalance_runs = log_imbalance_comparison()
    ablation_runs = log_baseline_ablation()

    print(f"Logged #21 imbalance-comparison runs (experiment '{EXPERIMENT_IMBALANCE}'):")
    for name, run_id in imbalance_runs.items():
        print(f"  {name}: {run_id}")

    print(f"\nLogged #72 baseline-ablation runs (experiment '{EXPERIMENT_ABLATION}'):")
    for name, run_id in ablation_runs.items():
        print(f"  {name}: {run_id}")

    print(f"\nTracking store: {TRACKING_URI}")
    print(f"Inspect with: mlflow ui --backend-store-uri {TRACKING_URI}")
    print("Or query: mlflow.search_runs(experiment_names=[...]) -- see docs/mlflow_tracking.md")


if __name__ == "__main__":
    main()
