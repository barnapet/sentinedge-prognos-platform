"""Tests for MLflow instrumentation of the #21/#72 LOEO comparisons (Issue #74).

Uses the same synthetic training-dataset-shaped frame as `tests/test_train_baseline.py`
(`make_dataset`) rather than the real `data/processed/training_dataset.parquet`, which is
gitignored and not present when CI's unit-test step runs (before the notebook/data step)
-- same rationale `tests/test_train_baseline.py` and `tests/test_training.py` already give.

Every test points MLflow at an isolated, per-test SQLite store (`tmp_path`) rather than the
repo's own `mlflow.db`, so running the suite never mixes synthetic test runs into the real
tracking history `python -m src.training.mlflow_tracking` produces.
"""
from __future__ import annotations

import mlflow
import pytest

from src.labeling import LABELS
from src.training.compare_imbalance import run_strategy_on_fold
from src.training.evaluation import aggregate, loeo_folds
from src.training.imbalance import STRATEGIES
from src.training.mlflow_tracking import (
    EXPERIMENT_ABLATION,
    EXPERIMENT_IMBALANCE,
    log_baseline_ablation,
    log_imbalance_comparison,
)
from src.training.train_baseline import FEATURE_SETS, run_all_feature_sets
from tests.test_train_baseline import make_dataset


@pytest.fixture(autouse=True)
def isolated_tracking_store(tmp_path):
    """Every test in this module gets its own SQLite store, never the repo's `mlflow.db`."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'test_mlflow.db'}")


def _small_comparison_results():
    """One real run of #21's five-arm comparison on the synthetic dataset.

    `compare_imbalance.run_comparison()` itself always reads the real
    `training_dataset.parquet` (no way to inject a frame), so this mirrors its loop body
    over the synthetic frame instead -- same `run_strategy_on_fold`/`aggregate` calls
    `run_comparison` uses, just pointed at `make_dataset()`.
    """
    folds = loeo_folds(make_dataset())
    results = {}
    for strategy in STRATEGIES:
        per_fold = [run_strategy_on_fold(strategy, fold) for fold in folds]
        results[strategy.name] = {
            "description": strategy.description,
            "per_fold": per_fold,
            "aggregate": {
                "critical_recall": aggregate([f["recall"]["Critical"] for f in per_fold]),
                "critical_precision": aggregate([f["precision"]["Critical"] for f in per_fold]),
                "macro_f1": aggregate([f["macro_f1"] for f in per_fold]),
            },
        }
    return results


# --- #21 imbalance-comparison logging ------------------------------------------------


def test_logs_one_run_per_strategy():
    results = _small_comparison_results()

    run_ids = log_imbalance_comparison(results=results)

    assert set(run_ids) == {s.name for s in STRATEGIES}


def test_imbalance_runs_are_tagged_and_searchable_per_strategy():
    """The distinguishing requirement from Issue #74: each of the five arms must be
    identifiable and separately queryable in the MLflow UI/API, not just logged."""
    results = _small_comparison_results()
    log_imbalance_comparison(results=results)

    found = mlflow.search_runs(
        experiment_names=[EXPERIMENT_IMBALANCE],
        filter_string="params.strategy = 'class_weight_balanced'",
    )

    assert len(found) == 1
    assert found.iloc[0]["tags.issue"] == "21"
    assert found.iloc[0]["params.feature_set"] == "full"


def test_imbalance_run_metrics_match_the_computed_result_exactly():
    """Pins that logging is a faithful pass-through: what MLflow reports back must equal
    what `run_comparison` computed, not a re-derived or rounded copy of it."""
    results = _small_comparison_results()
    log_imbalance_comparison(results=results)

    balanced = results["class_weight_balanced"]
    found = mlflow.search_runs(
        experiment_names=[EXPERIMENT_IMBALANCE],
        filter_string="params.strategy = 'class_weight_balanced'",
    ).iloc[0]

    for fold in balanced["per_fold"]:
        held_out = fold["held_out"]
        for label in LABELS:
            assert found[f"metrics.recall_{label}_{held_out}"] == pytest.approx(
                fold["recall"][label]
            )
    assert found["metrics.critical_recall_mean"] == pytest.approx(
        balanced["aggregate"]["critical_recall"]["mean"]
    )


def test_imbalance_run_logs_confusion_matrices_as_a_retrievable_artifact():
    import json

    results = _small_comparison_results()
    run_ids = log_imbalance_comparison(results=results)

    path = mlflow.artifacts.download_artifacts(
        run_id=run_ids["none"], artifact_path="confusion_matrices.json"
    )
    logged = json.load(open(path))

    expected = {f["held_out"]: f["confusion_matrix"] for f in results["none"]["per_fold"]}
    assert logged == expected


# --- #72 baseline-ablation logging ----------------------------------------------------


def test_logs_only_the_two_configurations_issue_74_asks_for():
    """Issue #74 Task 2 names exactly `full` and `no_rms_ratio` -- not #72's other two
    diagnostic feature sets (`no_raw_rms`, `kurtosis_skewness_only`)."""
    df = make_dataset()
    feature_sets = {name: FEATURE_SETS[name] for name in ("full", "no_rms_ratio")}
    results = run_all_feature_sets(feature_sets=feature_sets, df=df)

    run_ids = log_baseline_ablation(configs=("full", "no_rms_ratio"), results=results)

    assert set(run_ids) == {"full", "no_rms_ratio"}


def test_ablation_runs_are_tagged_and_searchable_per_configuration():
    df = make_dataset()
    feature_sets = {name: FEATURE_SETS[name] for name in ("full", "no_rms_ratio")}
    results = run_all_feature_sets(feature_sets=feature_sets, df=df)
    log_baseline_ablation(configs=("full", "no_rms_ratio"), results=results)

    found = mlflow.search_runs(
        experiment_names=[EXPERIMENT_ABLATION],
        filter_string="params.feature_set = 'no_rms_ratio'",
    )

    assert len(found) == 1
    assert found.iloc[0]["tags.issue"] == "72"
    assert found.iloc[0]["params.imbalance_strategy"] == "class_weight_balanced"


def test_ablation_run_metrics_match_the_computed_result_exactly():
    df = make_dataset()
    feature_sets = {name: FEATURE_SETS[name] for name in ("full", "no_rms_ratio")}
    results = run_all_feature_sets(feature_sets=feature_sets, df=df)
    log_baseline_ablation(configs=("full", "no_rms_ratio"), results=results)

    full = results["full"]
    found = mlflow.search_runs(
        experiment_names=[EXPERIMENT_ABLATION],
        filter_string="params.feature_set = 'full'",
    ).iloc[0]

    assert found["metrics.normal_recall_mean"] == pytest.approx(
        full["aggregate"]["normal_recall"]["mean"]
    )
    assert found["metrics.macro_f1_range"] == pytest.approx(full["aggregate"]["macro_f1"]["range"])
