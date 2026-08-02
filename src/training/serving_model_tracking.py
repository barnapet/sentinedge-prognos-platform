"""MLflow tracking for the pooled M4 serving model (Issue #80).

Follows `src/training/mlflow_tracking.py` (#74)'s instrumentation pattern -- same local
SQLite store, same `mlartifacts/` artifact root, same injectable-arguments shape so tests
can log without the real dataset -- in a separate module because the run it logs is a
different *kind* of run, and Issue #80 asks for that difference to be visible.

**What makes this run distinguishable**, which is the requirement:

- Its own experiment, `m4-serving-model`, separate from `m3-imbalance-comparison` (#21)
  and `m3-baseline-ablation` (#72).
- Tag `run_purpose=serving_artifact`. The #21/#72 runs are evaluation-only: they measure
  generalization under LOEO and persist nothing. This one produces the binary the serving
  layer actually loads, and logs that binary as a run artifact so the MLflow run and the
  file on disk are tied together rather than merely adjacent.
- Every metric is prefixed `insample_`, and `metrics_scope` is logged as a parameter. The
  pooled model has no held-out rows (`docs/serving_design.md` Section 4), so its metrics
  measure fit, not capability -- the opposite of what the #21/#72 runs' metrics mean.
  Prefixing them makes the two impossible to confuse in a UI that lists runs side by side.

The model is logged with `mlflow.log_artifact`, not `mlflow.sklearn.log_model`: the
sklearn flavor is unavailable under `mlflow-skinny` (verified -- it raises
`ModuleNotFoundError: No module named 'skops'`), and `docs/mlflow_tracking.md` Section 1
already decided not to install the full `mlflow` package, which would downgrade this
project's pinned `pandas==3.0.5`. See `docs/serving_model_artifact.md` Section 4.
"""
from __future__ import annotations

from pathlib import Path

import mlflow

from src.labeling import LABELS
from src.training.mlflow_tracking import configure_tracking, ensure_experiment

EXPERIMENT_SERVING = "m4-serving-model"

# Logged as a parameter so a reader of the run does not have to know that a pooled model
# has no held-out rows to understand what its metrics do and do not mean.
METRICS_SCOPE = "in_sample_training_fit"

# Where the LOEO evidence for this model class lives, per `docs/model_training_decision.md`
# Section 6 -- logged so the serving run points at its own generalization evidence rather
# than appearing to be its own.
LOEO_EVALUATION_EXPERIMENT = "m3-baseline-ablation"


def _log_insample_metrics(metrics: dict) -> None:
    """Log per-class recall/precision/support and macro-F1, all `insample_`-prefixed."""
    for label in LABELS:
        mlflow.log_metric(f"insample_recall_{label}", metrics["recall"][label])
        mlflow.log_metric(f"insample_precision_{label}", metrics["precision"][label])
        mlflow.log_metric(f"insample_support_{label}", metrics["support"][label])
    mlflow.log_metric("insample_macro_f1", metrics["macro_f1"])


def _serving_run_params(manifest: dict) -> dict:
    """The parameters that identify which artifact this run produced, and from what."""
    return {
        "split": manifest["split"],
        "trained_on": ",".join(manifest["trained_on"]),
        "feature_columns": ",".join(manifest["feature_columns"]),
        "class_weight": manifest["class_weight"],
        "max_iter": manifest["model_params"]["max_iter"],
        "random_state": manifest["model_params"]["random_state"],
        "n_training_rows": manifest["n_training_rows"],
        "metrics_scope": METRICS_SCOPE,
        "loeo_evaluation_experiment": LOEO_EVALUATION_EXPERIMENT,
        "combined_hash": manifest["combined_hash"],
        "model_sha256": manifest["model_sha256"],
        "training_dataset_version": manifest["training_dataset_version"],
    }


def log_serving_model_run(
    manifest: dict,
    metrics: dict,
    model_path: Path | None = None,
    manifest_path: Path | None = None,
) -> str:
    """Log one run recording the pooled training that produced the served artifact.

    `manifest` and `metrics` are passed in rather than recomputed, so the run describes
    the artifact that was actually just written -- the same injectable shape
    `mlflow_tracking.log_imbalance_comparison` uses, and the reason a test can exercise
    this without the real dataset.

    Logs to whatever `mlflow.set_tracking_uri` currently points at -- callers decide that
    by calling `configure_tracking()` first; this function does not, so tests can point it
    at an isolated store without a default silently overriding them. Same contract as
    `mlflow_tracking.log_imbalance_comparison`.

    Args:
        manifest: `train_serving_model.build_serving_manifest`'s output.
        metrics: `train_serving_model.insample_metrics`'s output.
        model_path: The persisted `.joblib`, logged as a run artifact when given.
        manifest_path: The persisted manifest JSON, logged alongside it when given.

    Returns:
        The MLflow run ID.
    """
    experiment_id = ensure_experiment(EXPERIMENT_SERVING)

    with mlflow.start_run(experiment_id=experiment_id, run_name="pooled_serving_model") as run:
        mlflow.set_tags(
            {
                "issue": "80",
                "component": "serving_model",
                # The distinguishing tag: #21/#72's runs are evaluation-only.
                "run_purpose": "serving_artifact",
            }
        )
        mlflow.log_params(_serving_run_params(manifest))
        _log_insample_metrics(metrics)
        mlflow.log_dict({"in_sample": metrics["confusion_matrix"]}, "confusion_matrix.json")

        if model_path is not None:
            mlflow.log_artifact(str(model_path))
        if manifest_path is not None:
            mlflow.log_artifact(str(manifest_path))

        return run.info.run_id


def main() -> None:
    """Log a serving run for the artifact already on disk, without retraining it."""
    import json

    from src.training.train_serving_model import (
        MANIFEST_PATH,
        MODEL_PATH,
        insample_metrics,
        load_serving_model,
    )
    from src.training.evaluation import load_training_dataset

    configure_tracking()
    manifest = json.loads(MANIFEST_PATH.read_text())
    metrics = insample_metrics(load_serving_model(), load_training_dataset())

    run_id = log_serving_model_run(
        manifest=manifest,
        metrics=metrics,
        model_path=MODEL_PATH,
        manifest_path=MANIFEST_PATH,
    )
    print(f"logged MLflow run {run_id} (experiment '{EXPERIMENT_SERVING}')")


if __name__ == "__main__":
    main()
