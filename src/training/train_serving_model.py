"""Train and persist the single pooled model the M4 serving layer loads (Issue #80).

`docs/serving_design.md` Section 4 decided what this module produces: **one** model,
trained on all three experiments pooled, using the exact configuration
`docs/model_training_decision.md` already adopted -- and persisted, which no M3 module
does. #21's and #72's harnesses deliberately train and discard three throwaway models per
run (`docs/model_training_decision.md` Section 5): correct for *measuring* generalization
under LOEO, but it leaves nothing to serve.

The anti-drift property this module is built around: it does not re-declare the model
configuration, it **imports** it. `BASELINE_STRATEGY` is the same object #72 used
(`class_weight='balanced'`, `BASELINE_MODEL_PARAMS`), and `FEATURE_MATRIX_COLUMNS` is the
same five-column list `evaluation.py` defines. Nothing here can drift from the adopted
baseline without also changing what #72's LOEO numbers were measured on, and
`tests/test_train_serving_model.py` pins that.

**What changes versus LOEO, and what does not.** The only difference is the split: every
row trains, nothing is held out. That is deliberate and is *not* a second evaluation --
`insample_metrics` below scores the model on its own training rows, which measures fit,
not generalization. The generalization evidence for this model class stays what
`docs/model_training_decision.md` Section 6 reports, including the `1st_test` failure
`docs/serving_design.md` Section 4 requires the service to disclose.

Artifact location, gitignore status, and the byte-level reproducibility claim are decided
and evidenced in `docs/serving_model_artifact.md`.

Reproducing:

    python -m src.features.build_training_dataset   # Issue #67, if not already built
    python -m src.training.train_serving_model
"""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from src.features.versioning import compute_combined_hash, write_manifest
from src.labeling import LABELS
from src.training.evaluation import (
    EXPERIMENTS,
    FEATURE_MATRIX_COLUMNS,
    TRAINING_DATASET_PATH,
    fold_metrics,
    load_training_dataset,
)
# The adopted configuration, imported rather than re-declared -- see the module docstring.
from src.training.imbalance import BASELINE_MODEL_PARAMS
from src.training.train_baseline import BASELINE_STRATEGY

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
MODEL_PATH = MODELS_DIR / "serving_model.joblib"
MANIFEST_PATH = MODELS_DIR / "serving_model_manifest.json"
TRAINING_DATASET_MANIFEST_PATH = TRAINING_DATASET_PATH.parent / "training_dataset_manifest.json"

# The source files whose content determines what this artifact *is*. Same scheme and
# reasoning as `build_training_dataset.LABELING_CODE_FILES` (`docs/training_dataset_versioning.md`
# Section 2): its own hash chain, deliberately not merged into
# `src/features/versioning.py`'s `GENERATING_CODE_FILES`, which stays scoped to the
# feature parquets so this issue cannot invalidate them.
SERVING_CODE_FILES: tuple[Path, ...] = (
    REPO_ROOT / "src" / "training" / "imbalance.py",
    REPO_ROOT / "src" / "training" / "evaluation.py",
    REPO_ROOT / "src" / "training" / "train_serving_model.py",
)


def train_pooled_model(df: pd.DataFrame) -> Pipeline:
    """Fit the adopted baseline pipeline on every row, with no held-out fold.

    `docs/serving_design.md` Section 4's decision. The feature matrix and label vector are
    built exactly as `evaluation.loeo_folds` builds a fold's -- same columns, same
    `astype(str)` on the label -- so the only difference from a LOEO fold is that nothing
    is excluded.
    """
    model = BASELINE_STRATEGY.build_model()
    model.fit(feature_matrix(df), label_vector(df))
    return model


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """The five adopted feature columns, in `FEATURE_MATRIX_COLUMNS` order."""
    return df[FEATURE_MATRIX_COLUMNS].to_numpy()


def label_vector(df: pd.DataFrame) -> np.ndarray:
    """String labels, matching `evaluation.loeo_folds`'s `astype(str)` handling."""
    return df["label"].astype(str).to_numpy()


def insample_metrics(model: Pipeline, df: pd.DataFrame) -> dict:
    """Score the pooled model on its own training rows.

    Reuses `evaluation.fold_metrics` so the metric *definitions* are identical to the ones
    `docs/evaluation_protocol.md` Section 4 committed to. The **scope** is not: these rows
    were all trained on, so this measures fit, not generalization, and every consumer here
    labels it `insample_` for that reason. `docs/model_training_decision.md` Section 6
    remains the honest statement of what this model class generalizes to.
    """
    return fold_metrics(label_vector(df), model.predict(feature_matrix(df)))


def compute_serving_code_hash(code_files: tuple[Path, ...] = SERVING_CODE_FILES) -> str:
    """SHA-256 over the sorted, concatenated bytes of the training source files.

    Same scheme as `src.features.versioning.compute_code_hash` and
    `build_training_dataset.compute_labeling_code_hash`, over a third, independent file
    set -- sorted first so the result does not depend on argument order.
    """
    hasher = hashlib.sha256()
    for path in sorted(code_files):
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def read_training_dataset_version(manifest_path: Path = TRAINING_DATASET_MANIFEST_PATH) -> str:
    """The `combined_hash` of the training dataset this model was fitted on (Issue #67).

    Reused rather than recomputed, exactly as `build_training_dataset` reuses the upstream
    feature manifests' hashes: it already answers "this labeling code, against these
    feature-parquet versions", so chaining it here extends the same provenance chain one
    link further without re-reading any parquet.
    """
    return json.loads(manifest_path.read_text())["combined_hash"]


def build_serving_manifest(
    model: Pipeline,
    df: pd.DataFrame,
    model_sha256: str,
    training_dataset_version: str | None = None,
    generated_at: datetime | None = None,
) -> dict:
    """Record what produced this artifact, and what is needed to load it back.

    Follows `docs/training_dataset_versioning.md`'s manifest pattern (code hash + upstream
    data version + combined hash), with two additions specific to a *pickled* artifact:
    `model_sha256`, which makes the committed binary auditable rather than opaque, and
    `library_versions`, since a joblib-serialised sklearn estimator is only guaranteed to
    load cleanly under the versions that wrote it.

    `training_dataset_version` defaults to reading Issue #67's manifest, and is injectable
    so tests can build a manifest without `data/processed/`, which is gitignored and absent
    when CI's unit-test step runs -- the same constraint `tests/test_train_baseline.py`
    works around with a synthetic frame.
    """
    serving_code_hash = compute_serving_code_hash()
    if training_dataset_version is None:
        training_dataset_version = read_training_dataset_version()
    generated_at = generated_at or datetime.now(timezone.utc)
    return {
        "generated_at": generated_at.isoformat(),
        "serving_code_hash": serving_code_hash,
        "training_dataset_version": training_dataset_version,
        "combined_hash": compute_combined_hash(serving_code_hash, training_dataset_version),
        "model_sha256": model_sha256,
        "trained_on": sorted(df["experiment"].unique().tolist()),
        "split": "pooled_all_experiments_no_holdout",
        "feature_columns": list(FEATURE_MATRIX_COLUMNS),
        "labels": LABELS,
        "class_weight": BASELINE_STRATEGY.build_model().get_params()["clf__class_weight"],
        "model_params": dict(BASELINE_MODEL_PARAMS),
        "pipeline_steps": [name for name, _ in model.steps],
        "n_training_rows": int(len(df)),
        "n_rows_per_experiment": {
            str(name): int(n) for name, n in df["experiment"].value_counts().items()
        },
        "class_support": {label: int((label_vector(df) == label).sum()) for label in LABELS},
        "library_versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }


def serialise_model(model: Pipeline) -> bytes:
    """`joblib.dump` the fitted pipeline to bytes.

    Bytes rather than straight to disk so the artifact can be hashed before it is written,
    letting the manifest record the SHA-256 of exactly the file that lands on disk.
    """
    import io

    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.getvalue()


def persist_serving_model(
    model: Pipeline,
    df: pd.DataFrame,
    model_path: Path = MODEL_PATH,
    manifest_path: Path = MANIFEST_PATH,
    training_dataset_version: str | None = None,
) -> dict:
    """Write the fitted pipeline and its manifest, returning the manifest.

    The whole `Pipeline` is persisted, not just the classifier: `StandardScaler` is a
    fitted component (it carries the pooled training rows' means and scales), so serving
    the classifier without it would silently feed unscaled features to a model fitted on
    scaled ones. `docs/evaluation_protocol.md` Section 1's reason for keeping the scaler
    inside the pipeline applies verbatim at serving time.
    """
    payload = serialise_model(model)
    model_sha256 = hashlib.sha256(payload).hexdigest()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(payload)

    manifest = build_serving_manifest(
        model,
        df,
        model_sha256=model_sha256,
        training_dataset_version=training_dataset_version,
    )
    write_manifest(manifest, manifest_path)
    return manifest


def load_serving_model(model_path: Path = MODEL_PATH) -> Pipeline:
    """Load the persisted pipeline. The serving layer's entrypoint to this artifact."""
    return joblib.load(model_path)


def verify_artifact_integrity(
    model_path: Path = MODEL_PATH, manifest_path: Path = MANIFEST_PATH
) -> bool:
    """Does the artifact on disk still hash to what its manifest recorded?

    Exists because this artifact is committed rather than gitignored
    (`docs/serving_model_artifact.md`): a binary in version control is only as trustworthy
    as the ability to check it, and this is that check in one call.
    """
    recorded = json.loads(manifest_path.read_text())["model_sha256"]
    return hashlib.sha256(model_path.read_bytes()).hexdigest() == recorded


def format_report(manifest: dict, metrics: dict) -> str:
    """Human-readable summary, with the in-sample caveat attached to the numbers."""
    lines = [
        f"Trained on {manifest['n_training_rows']} rows, "
        f"pooled across {', '.join(manifest['trained_on'])} (no held-out fold)",
        f"  class support: {manifest['class_support']}",
        f"  combined_hash: {manifest['combined_hash'][:16]}...",
        f"  model_sha256:  {manifest['model_sha256'][:16]}...",
        "",
        "In-sample fit (training rows -- NOT a generalization estimate):",
    ]
    for label in LABELS:
        lines.append(
            f"  {label:<10} recall={metrics['recall'][label]:.3f}  "
            f"precision={metrics['precision'][label]:.3f}"
        )
    lines += [
        f"  macro-F1: {metrics['macro_f1']:.3f}",
        "",
        "This model's LOEO-validated capability -- the honest number -- is unchanged:",
        "  docs/model_training_decision.md Section 6. Critical recall 0.913 / 1.000 on",
        "  2nd_test / 3rd_test, and 0.059 on 1st_test. The in-sample figures above",
        "  measure fit, not generalization, and must not be quoted as performance.",
    ]
    return "\n".join(lines)


def main() -> None:
    df = load_training_dataset()
    model = train_pooled_model(df)
    manifest = persist_serving_model(model, df)
    metrics = insample_metrics(model, df)

    print(format_report(manifest, metrics))
    print(f"\nwrote {MODEL_PATH} ({MODEL_PATH.stat().st_size} bytes)")
    print(f"wrote {MANIFEST_PATH}")

    # Imported here rather than at module scope so training and persistence do not depend
    # on MLflow being importable -- the artifact is the deliverable, tracking is
    # instrumentation on it.
    from src.training.mlflow_tracking import configure_tracking
    from src.training.serving_model_tracking import EXPERIMENT_SERVING, log_serving_model_run

    configure_tracking()
    run_id = log_serving_model_run(
        manifest=manifest,
        metrics=metrics,
        model_path=MODEL_PATH,
        manifest_path=MANIFEST_PATH,
    )
    print(f"\nlogged MLflow run {run_id} (experiment '{EXPERIMENT_SERVING}')")


if __name__ == "__main__":
    main()
