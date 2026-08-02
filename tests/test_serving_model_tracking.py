"""Tests for the MLflow run recording the pooled serving model (Issue #80).

The property under test is *distinguishability*: Issue #80 requires that the run which
produced the served artifact be separable, in the MLflow UI and API, from #21's and #72's
evaluation-only LOEO runs. Several tests below therefore log a serving run alongside real
#72-style ablation runs and assert that a query can still pick out exactly one.

Every test points MLflow at an isolated, per-test SQLite store (`tmp_path`) rather than the
repo's own `mlflow.db`, so running the suite never mixes synthetic runs into the real
tracking history -- the same fixture `tests/test_mlflow_tracking.py` uses.
"""
from __future__ import annotations

import json

import mlflow
import pytest

from src.labeling import LABELS
from src.training.mlflow_tracking import EXPERIMENT_ABLATION, log_baseline_ablation
from src.training.serving_model_tracking import (
    EXPERIMENT_SERVING,
    METRICS_SCOPE,
    log_serving_model_run,
)
from src.training.train_baseline import FEATURE_SETS, run_all_feature_sets
from src.training.train_serving_model import (
    insample_metrics,
    persist_serving_model,
    train_pooled_model,
)
from tests.test_train_baseline import make_dataset

FAKE_DATASET_VERSION = "0" * 64


@pytest.fixture(autouse=True)
def isolated_tracking_store(tmp_path):
    """Every test gets its own SQLite store, never the repo's `mlflow.db`."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'test_mlflow.db'}")


@pytest.fixture
def serving_run(tmp_path):
    """Train, persist, and log one serving run against the synthetic dataset."""
    df = make_dataset()
    model = train_pooled_model(df)
    model_path = tmp_path / "serving_model.joblib"
    manifest_path = tmp_path / "serving_model_manifest.json"
    manifest = persist_serving_model(
        model,
        df,
        model_path=model_path,
        manifest_path=manifest_path,
        training_dataset_version=FAKE_DATASET_VERSION,
    )
    metrics = insample_metrics(model, df)
    run_id = log_serving_model_run(
        manifest=manifest,
        metrics=metrics,
        model_path=model_path,
        manifest_path=manifest_path,
    )
    return {"run_id": run_id, "manifest": manifest, "metrics": metrics}


def _serving_run_row():
    return mlflow.search_runs(experiment_names=[EXPERIMENT_SERVING]).iloc[0]


# --- distinguishable from the #21/#72 evaluation runs ------------------------------------


def test_serving_run_lands_in_its_own_experiment(serving_run):
    assert len(mlflow.search_runs(experiment_names=[EXPERIMENT_SERVING])) == 1
    assert EXPERIMENT_SERVING not in {EXPERIMENT_ABLATION, "m3-imbalance-comparison"}


def test_serving_run_is_tagged_as_the_artifact_producing_run(serving_run):
    row = _serving_run_row()

    assert row["tags.run_purpose"] == "serving_artifact"
    assert row["tags.issue"] == "80"
    assert row["tags.component"] == "serving_model"


def test_a_single_query_separates_the_served_artifact_from_evaluation_runs(serving_run):
    """Issue #80's actual requirement, tested against real #72 ablation runs rather than
    against the serving run alone -- otherwise 'distinguishable' would be untested."""
    results = run_all_feature_sets(
        feature_sets={name: FEATURE_SETS[name] for name in ("full", "no_rms_ratio")},
        df=make_dataset(),
    )
    log_baseline_ablation(configs=("full", "no_rms_ratio"), results=results)

    experiments = [EXPERIMENT_SERVING, EXPERIMENT_ABLATION]
    served = mlflow.search_runs(
        experiment_names=experiments, filter_string="tags.run_purpose = 'serving_artifact'"
    )

    assert len(mlflow.search_runs(experiment_names=experiments)) == 3
    assert len(served) == 1
    assert served.iloc[0]["run_id"] == serving_run["run_id"]


# --- metrics are unmistakably in-sample ---------------------------------------------------


def test_every_logged_metric_is_prefixed_insample(serving_run):
    """A pooled model has no held-out rows, so its metrics measure fit. The prefix is what
    stops them being read as capability next to #72's LOEO numbers."""
    row = _serving_run_row()

    metric_columns = [c for c in row.index if c.startswith("metrics.")]

    assert metric_columns
    for column in metric_columns:
        assert column.startswith("metrics.insample_")


def test_metrics_scope_is_logged_as_a_parameter(serving_run):
    assert _serving_run_row()["params.metrics_scope"] == METRICS_SCOPE
    assert "in_sample" in METRICS_SCOPE


def test_run_points_at_where_its_generalization_evidence_lives(serving_run):
    """The serving run must not look like its own evidence of capability."""
    assert _serving_run_row()["params.loeo_evaluation_experiment"] == EXPERIMENT_ABLATION


def test_logged_metrics_match_the_computed_in_sample_metrics(serving_run):
    """Logging is a faithful pass-through, not a re-derivation."""
    row = _serving_run_row()

    for label in LABELS:
        assert row[f"metrics.insample_recall_{label}"] == pytest.approx(
            serving_run["metrics"]["recall"][label]
        )
        assert row[f"metrics.insample_support_{label}"] == pytest.approx(
            serving_run["metrics"]["support"][label]
        )
    assert row["metrics.insample_macro_f1"] == pytest.approx(serving_run["metrics"]["macro_f1"])


# --- the run is tied to the artifact on disk ----------------------------------------------


def test_run_logs_the_model_binary_and_its_manifest_as_artifacts(serving_run):
    """What makes this run the record of a *served* artifact rather than of a training
    event: the bytes themselves are attached to it."""
    logged = {f.path for f in mlflow.MlflowClient().list_artifacts(serving_run["run_id"])}

    assert {"serving_model.joblib", "serving_model_manifest.json"} <= logged


def test_run_records_the_artifact_hash_so_it_can_be_matched_to_a_file_on_disk(serving_run):
    assert _serving_run_row()["params.model_sha256"] == serving_run["manifest"]["model_sha256"]
    assert _serving_run_row()["params.combined_hash"] == serving_run["manifest"]["combined_hash"]


def test_run_records_the_adopted_pipeline_configuration(serving_run):
    """The configuration `docs/model_training_decision.md` fixed, visible in the run itself
    rather than only in the manifest."""
    row = _serving_run_row()

    assert row["params.class_weight"] == "balanced"
    assert row["params.split"] == "pooled_all_experiments_no_holdout"
    assert row["params.trained_on"] == "1st_test,2nd_test,3rd_test"
    assert row["params.feature_columns"] == "rms,rms_ratio,kurtosis,skewness,skewness_smoothed"


def test_confusion_matrix_is_logged_as_a_retrievable_artifact(serving_run):
    path = mlflow.artifacts.download_artifacts(
        run_id=serving_run["run_id"], artifact_path="confusion_matrix.json"
    )

    assert json.load(open(path)) == {"in_sample": serving_run["metrics"]["confusion_matrix"]}
