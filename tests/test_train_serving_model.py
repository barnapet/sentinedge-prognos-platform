"""Tests for the pooled M4 serving model and its persisted artifact (Issue #80).

Two properties carry most of the weight here, because they are what the serving layer will
depend on and what could silently rot:

- **No drift** between "how #72 trains a pipeline" and "how this issue persists one". The
  training module imports its configuration rather than re-declaring it, and the tests
  below re-derive the expected model *independently* (constructing a pipeline from
  `imbalance.make_baseline_model` in-test) and require the persisted artifact to predict
  identically. A copy-pasted configuration that drifted would fail these.
- **Determinism**, asserted at the byte level rather than the metric level, since
  `docs/serving_model_artifact.md` Section 3 uses reproducibility as the justification for
  committing the artifact to the repo at all.

Uses the same synthetic training-dataset-shaped frame as `tests/test_train_baseline.py`
rather than the real `data/processed/training_dataset.parquet`, which is gitignored and
absent when CI's unit-test step runs -- the rationale `tests/test_training.py`,
`tests/test_train_baseline.py`, and `tests/test_mlflow_tracking.py` already share.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from src.labeling import LABELS
from src.training.evaluation import FEATURE_MATRIX_COLUMNS
from src.training.imbalance import BASELINE_MODEL_PARAMS, make_baseline_model
from src.training.train_baseline import FEATURE_SETS
from src.training.train_serving_model import (
    SERVING_CODE_FILES,
    build_serving_manifest,
    compute_serving_code_hash,
    feature_matrix,
    insample_metrics,
    label_vector,
    load_serving_model,
    persist_serving_model,
    serialise_model,
    train_pooled_model,
    verify_artifact_integrity,
)
from tests.test_train_baseline import ROWS_PER_EXPERIMENT, make_dataset

# Stand-in for Issue #67's `training_dataset_manifest.json` `combined_hash`, which is not
# on disk in CI. Any fixed string works -- what is under test is that it is recorded and
# chained into `combined_hash`, not its value.
FAKE_DATASET_VERSION = "0" * 64


@pytest.fixture
def persisted(tmp_path):
    """Train and persist a serving model into `tmp_path`, never the repo's `models/`."""
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
    return {
        "df": df,
        "model": model,
        "manifest": manifest,
        "model_path": model_path,
        "manifest_path": manifest_path,
    }


# --- the required no-drift test --------------------------------------------------------


def test_persisted_artifact_predicts_identically_to_an_in_test_fitted_pipeline(persisted):
    """Issue #80's headline acceptance criterion.

    The reference pipeline here is built from `make_baseline_model` directly, with the
    configuration `docs/model_training_decision.md` records, rather than by calling the
    training module again -- otherwise both sides would drift together and the test would
    pass while proving nothing.
    """
    df = persisted["df"]
    reference = make_baseline_model(class_weight="balanced")
    reference.fit(df[FEATURE_MATRIX_COLUMNS].to_numpy(), df["label"].astype(str).to_numpy())

    loaded = load_serving_model(persisted["model_path"])

    np.testing.assert_array_equal(
        loaded.predict(feature_matrix(df)), reference.predict(feature_matrix(df))
    )
    np.testing.assert_allclose(
        loaded.predict_proba(feature_matrix(df)), reference.predict_proba(feature_matrix(df))
    )


def test_persisted_artifact_carries_the_fitted_scaler_not_just_the_classifier(persisted):
    """The whole `Pipeline` is persisted. Serving the bare classifier would feed unscaled
    features to a model fitted on scaled ones -- silently, with no error."""
    loaded = load_serving_model(persisted["model_path"])

    assert [name for name, _ in loaded.steps] == ["scaler", "clf"]
    np.testing.assert_allclose(
        loaded.named_steps["scaler"].mean_, feature_matrix(persisted["df"]).mean(axis=0)
    )


# --- configuration matches the adopted M3 baseline exactly -----------------------------


def test_pooled_model_uses_issue_21s_adopted_configuration(persisted):
    """`docs/model_training_decision.md`'s adopted configuration, asserted on the artifact
    itself rather than on the code that wrote it."""
    params = load_serving_model(persisted["model_path"]).get_params()

    assert params["clf__class_weight"] == "balanced"
    assert params["clf__max_iter"] == BASELINE_MODEL_PARAMS["max_iter"]
    assert params["clf__random_state"] == BASELINE_MODEL_PARAMS["random_state"]


def test_feature_columns_are_the_adopted_full_set_in_order():
    """The five columns `docs/model_training_decision.md` adopted -- the `full`
    configuration, not the `no_rms_ratio` ablation that over-alarms (its Section 2)."""
    assert FEATURE_MATRIX_COLUMNS == ["rms", "rms_ratio", "kurtosis", "skewness", "skewness_smoothed"]
    assert FEATURE_SETS["full"] == FEATURE_MATRIX_COLUMNS


def test_serving_model_sees_exactly_five_features(persisted):
    assert load_serving_model(persisted["model_path"]).named_steps["scaler"].n_features_in_ == 5


# --- pooled, not LOEO -------------------------------------------------------------------


def test_training_uses_every_row_of_every_experiment(persisted):
    """`docs/serving_design.md` Section 4's decision: no experiment is held out, because
    nothing is being estimated from a held-out fold at serving time."""
    manifest, df = persisted["manifest"], persisted["df"]

    assert manifest["n_training_rows"] == len(df)
    assert manifest["trained_on"] == sorted(ROWS_PER_EXPERIMENT)
    assert manifest["n_rows_per_experiment"] == dict(ROWS_PER_EXPERIMENT)
    assert manifest["split"] == "pooled_all_experiments_no_holdout"


def test_the_model_can_predict_every_label_it_was_trained_on(persisted):
    assert sorted(load_serving_model(persisted["model_path"]).classes_) == sorted(LABELS)


# --- determinism ------------------------------------------------------------------------


def test_refitting_produces_a_byte_identical_artifact():
    """`docs/serving_model_artifact.md` Section 3's reproducibility claim, which is the
    justification for committing the artifact rather than gitignoring it. Asserted on the
    serialised bytes, not on metrics: two models can score identically and still differ."""
    df = make_dataset()

    first = serialise_model(train_pooled_model(df))
    second = serialise_model(train_pooled_model(df))

    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


def test_manifest_records_the_hash_of_the_bytes_actually_written(persisted):
    on_disk = hashlib.sha256(persisted["model_path"].read_bytes()).hexdigest()

    assert persisted["manifest"]["model_sha256"] == on_disk


def test_verify_artifact_integrity_detects_a_modified_artifact(persisted):
    """The check that makes a committed binary auditable rather than trusted."""
    assert verify_artifact_integrity(persisted["model_path"], persisted["manifest_path"])

    persisted["model_path"].write_bytes(persisted["model_path"].read_bytes() + b"tampered")

    assert not verify_artifact_integrity(persisted["model_path"], persisted["manifest_path"])


# --- provenance manifest ----------------------------------------------------------------


def test_manifest_chains_code_and_dataset_versions_into_one_combined_hash(persisted):
    """Same provenance pattern as `docs/training_dataset_versioning.md` Section 2, one link
    further down the pipeline: this model, from this training code, on that dataset."""
    manifest = persisted["manifest"]

    assert manifest["serving_code_hash"] == compute_serving_code_hash()
    assert manifest["training_dataset_version"] == FAKE_DATASET_VERSION
    expected = hashlib.sha256(
        f"{manifest['serving_code_hash']}:{FAKE_DATASET_VERSION}".encode()
    ).hexdigest()
    assert manifest["combined_hash"] == expected


def test_serving_code_hash_covers_the_files_that_define_the_model():
    """If the imbalance treatment, the feature matrix, or the training entrypoint changes,
    the artifact's recorded provenance must change with it."""
    names = {path.name for path in SERVING_CODE_FILES}

    assert names == {"imbalance.py", "evaluation.py", "train_serving_model.py"}
    for path in SERVING_CODE_FILES:
        assert path.exists()


def test_serving_code_hash_is_order_independent():
    reversed_order = tuple(reversed(SERVING_CODE_FILES))

    assert compute_serving_code_hash(reversed_order) == compute_serving_code_hash()


def test_manifest_records_library_versions_needed_to_load_the_pickle(persisted):
    """A joblib-serialised estimator is only guaranteed to load under the versions that
    wrote it, so the committed artifact records them next to itself."""
    versions = persisted["manifest"]["library_versions"]

    assert {"python", "scikit-learn", "numpy", "joblib"} <= set(versions)
    assert all(versions.values())


def test_manifest_is_written_as_readable_json(persisted):
    """The manifest is the human-auditable half of a committed binary -- it has to be
    diffable text, not another opaque blob."""
    loaded = json.loads(persisted["manifest_path"].read_text())

    assert loaded == persisted["manifest"]


# --- in-sample metrics are labelled as such ---------------------------------------------


def test_insample_metrics_report_the_committed_metric_shape(persisted):
    """Same metric definitions as `docs/evaluation_protocol.md` Section 4 (reused via
    `fold_metrics`), scored on the training rows."""
    metrics = insample_metrics(persisted["model"], persisted["df"])

    assert set(metrics["recall"]) == set(LABELS)
    assert set(metrics["precision"]) == set(LABELS)
    assert np.array(metrics["confusion_matrix"]).shape == (3, 3)


def test_insample_support_equals_the_full_pooled_class_counts(persisted):
    """The tell that these metrics are in-sample: their support is the *training* support,
    not a held-out fold's."""
    metrics = insample_metrics(persisted["model"], persisted["df"])
    y = label_vector(persisted["df"])

    for label in LABELS:
        assert metrics["support"][label] == int((y == label).sum())
    assert sum(metrics["support"].values()) == len(persisted["df"])


def test_manifest_class_support_matches_the_pooled_dataset(persisted):
    y = label_vector(persisted["df"])

    for label, count in persisted["manifest"]["class_support"].items():
        assert count == int((y == label).sum())


# --- the artifact committed to the repo --------------------------------------------------


def test_committed_artifact_matches_its_manifest_if_present():
    """The repo's own `models/serving_model.joblib` is committed
    (`docs/serving_model_artifact.md` Section 2). When it is present, it must hash to what
    its committed manifest says -- this is the check a reviewer would otherwise have to run
    by hand. Skipped rather than failed where the artifact has not been generated, so the
    suite still runs on a clone that has not trained one."""
    from src.training.train_serving_model import MANIFEST_PATH, MODEL_PATH

    if not (MODEL_PATH.exists() and MANIFEST_PATH.exists()):
        pytest.skip("no serving artifact on disk")

    assert verify_artifact_integrity()


def test_committed_artifact_still_loads_and_predicts_on_this_interpreter():
    """The version-coupling risk `docs/serving_model_artifact.md` Section 2 accepts, turned
    into a check rather than left as prose.

    A joblib-pickled estimator is only guaranteed to load under the libraries that wrote it,
    and the committed artifact is generated on a maintainer's machine while CI runs a
    different Python minor version -- so CI loading it here is the thing that would catch a
    stale or incompatible commit of this binary."""
    from src.training.train_serving_model import MODEL_PATH

    if not MODEL_PATH.exists():
        pytest.skip("no serving artifact on disk")

    model = load_serving_model()
    predictions = model.predict(feature_matrix(make_dataset()))

    assert len(predictions) == sum(ROWS_PER_EXPERIMENT.values())
    assert set(predictions) <= set(LABELS)
