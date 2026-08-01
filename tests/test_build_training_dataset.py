import json
import shutil

import pandas as pd
import pytest

from src.features import build_training_dataset as build_training_dataset_module
from src.features.build_training_dataset import (
    LABELING_CODE_FILES,
    PROCESSED_DIR,
    TRAINING_DATASET_COLUMNS,
    build_training_dataset,
    compute_labeling_code_hash,
    compute_upstream_feature_version,
    main,
)
from src.features.extraction import FEATURE_COLUMNS
from src.features.versioning import compute_combined_hash
from src.labeling import LABELS, assign_labels, derive_critical_multiple

# Synthetic per-experiment fixtures, not the real NASA data: `data/processed/` is
# gitignored and, in CI, the "Run unit tests" step runs *before* "Execute notebooks"
# (the step that actually populates `<name>_features.parquet` via `build_dataset.py`),
# so a test that required the real parquet files to exist would silently skip in a
# from-scratch CI run -- exactly where a regression would matter most. Same rationale
# `tests/test_labeling.py` used for #65's exactness tests, and the same one
# `tests/test_features.py` already applies to `build_experiment` (synthetic raw
# snapshot dirs rather than the real `data/raw/`).
EXPERIMENT_ROWS = {
    "1st_test": {"rms": [0.10, 0.10, 0.32, 0.40], "rms_ratio": [1.0, 1.05, 2.0, 2.87]},
    "2nd_test": {"rms": [0.20, 0.20, 0.55, 0.70, 0.90, 1.20], "rms_ratio": [1.0, 1.1, 1.6, 2.9, 4.5, 6.32]},
    "3rd_test": {"rms": [0.05] * 5 + [0.20, 0.35], "rms_ratio": [1.0, 1.02, 0.98, 1.01, 1.29, 3.0, 7.15]},
}


def make_feature_df(name: str, rms: list[float], rms_ratio: list[float]) -> pd.DataFrame:
    n = len(rms)
    return pd.DataFrame(
        {
            "experiment": [name] * n,
            "file_index": list(range(n)),
            "timestamp": pd.date_range("2003-10-22", periods=n, freq="10min"),
            "rms": rms,
            "rms_ratio": rms_ratio,
            "kurtosis": [3.0] * n,
            "skewness": [0.0] * n,
            "skewness_smoothed": [0.0] * n,
        }
    )[FEATURE_COLUMNS]


def write_feature_parquet_and_manifest(processed_dir, name: str, df: pd.DataFrame, combined_hash: str) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_dir / f"{name}_features.parquet", index=False)
    manifest = {
        "experiment": name,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "code_hash": "irrelevant-to-this-module",
        "raw_dataset_version": "irrelevant-to-this-module",
        "combined_hash": combined_hash,
        "feature_columns": FEATURE_COLUMNS,
        "n_files": len(df),
    }
    (processed_dir / f"{name}_features_manifest.json").write_text(json.dumps(manifest))


def write_all_experiments(processed_dir, rows=EXPERIMENT_ROWS):
    """Writes the three synthetic experiments this file's tests share."""
    for name, cols in rows.items():
        df = make_feature_df(name, cols["rms"], cols["rms_ratio"])
        write_feature_parquet_and_manifest(processed_dir, name, df, combined_hash=f"{name}-combined-hash")


# --- build_training_dataset: shape and columns -----------------------------------

def test_writes_parquet_with_expected_columns(tmp_path):
    write_all_experiments(tmp_path)

    path = build_training_dataset(processed_dir=tmp_path)
    written = pd.read_parquet(path)

    assert path == tmp_path / "training_dataset.parquet"
    assert list(written.columns) == TRAINING_DATASET_COLUMNS


def test_row_counts_match_the_three_input_parquets_with_none_dropped_or_duplicated(tmp_path):
    """The join must be lossless: every row from every experiment's feature parquet
    appears exactly once in the concatenated output."""
    write_all_experiments(tmp_path)

    written = pd.read_parquet(build_training_dataset(processed_dir=tmp_path))

    counts = written["experiment"].value_counts().to_dict()
    assert counts == {name: len(cols["rms"]) for name, cols in EXPERIMENT_ROWS.items()}
    assert len(written) == sum(len(cols["rms"]) for cols in EXPERIMENT_ROWS.values())


def test_experiment_column_values_are_exactly_the_three_known_experiments(tmp_path):
    write_all_experiments(tmp_path)

    written = pd.read_parquet(build_training_dataset(processed_dir=tmp_path))

    assert set(written["experiment"].unique()) == set(EXPERIMENT_ROWS)


def test_rms_ratio_is_present_unchanged_no_leakage_mitigation_applied(tmp_path):
    """Issue #67 Task 3: the join must not exclude, transform, or flag `rms_ratio` --
    that decision belongs to M3's model-training/ablation step, not here."""
    write_all_experiments(tmp_path)

    written = pd.read_parquet(build_training_dataset(processed_dir=tmp_path))

    for name, cols in EXPERIMENT_ROWS.items():
        subset = written.loc[written["experiment"] == name].sort_values("file_index")
        assert subset["rms_ratio"].tolist() == pytest.approx(cols["rms_ratio"])


# --- label correctness: no drift introduced by the join ---------------------------

@pytest.mark.parametrize("name", list(EXPERIMENT_ROWS))
def test_labels_match_calling_assign_labels_and_derive_critical_multiple_directly(tmp_path, name):
    """The substantive correctness check: for each experiment, the label/
    label_pre_override/override_applied columns the join produces must be bit-for-bit
    identical to what a caller gets by calling derive_critical_multiple + assign_labels
    directly on that experiment's own feature parquet -- i.e. the join introduces no
    silent relabeling, reordering, or dtype drift."""
    write_all_experiments(tmp_path)

    written = pd.read_parquet(build_training_dataset(processed_dir=tmp_path))
    from_join = written.loc[written["experiment"] == name].reset_index(drop=True)

    cols = EXPERIMENT_ROWS[name]
    input_df = make_feature_df(name, cols["rms"], cols["rms_ratio"])
    critical_multiple = derive_critical_multiple(input_df["rms_ratio"].max())
    expected = assign_labels(input_df, critical_multiple)

    pd.testing.assert_frame_equal(
        from_join[FEATURE_COLUMNS + ["label", "label_pre_override", "override_applied"]],
        expected[FEATURE_COLUMNS + ["label", "label_pre_override", "override_applied"]],
    )


def test_all_three_labels_appear_somewhere_in_the_synthetic_fixture(tmp_path):
    """Sanity check on the fixture itself: every one of `EXPERIMENT_ROWS`' rms_ratio
    sequences was chosen to cross both the onset and Critical boundary, so this isn't
    silently testing an all-Normal edge case."""
    write_all_experiments(tmp_path)

    written = pd.read_parquet(build_training_dataset(processed_dir=tmp_path))

    assert set(written["label"].unique()) == set(LABELS)


# --- manifest: fields, reproducibility, drift detection ---------------------------

def test_manifest_written_alongside_parquet_with_expected_fields(tmp_path):
    write_all_experiments(tmp_path)

    build_training_dataset(processed_dir=tmp_path)
    manifest = json.loads((tmp_path / "training_dataset_manifest.json").read_text())

    assert manifest["labeling_code_hash"] == compute_labeling_code_hash()
    assert manifest["combined_hash"] == compute_combined_hash(
        manifest["labeling_code_hash"], manifest["upstream_feature_version"]
    )
    assert manifest["labels"] == LABELS
    assert manifest["columns"] == TRAINING_DATASET_COLUMNS
    assert manifest["n_files"] == {name: len(cols["rms"]) for name, cols in EXPERIMENT_ROWS.items()}
    assert manifest["n_files_total"] == sum(len(cols["rms"]) for cols in EXPERIMENT_ROWS.values())
    assert set(manifest["critical_multiple"]) == set(EXPERIMENT_ROWS)
    generated_at = pd.Timestamp(manifest["generated_at"])
    assert generated_at.tz is not None


def test_critical_multiple_in_manifest_matches_deriving_it_directly(tmp_path):
    write_all_experiments(tmp_path)

    build_training_dataset(processed_dir=tmp_path)
    manifest = json.loads((tmp_path / "training_dataset_manifest.json").read_text())

    for name, cols in EXPERIMENT_ROWS.items():
        expected = derive_critical_multiple(max(cols["rms_ratio"]))
        assert manifest["critical_multiple"][name] == expected


def test_reproducible_for_unchanged_inputs(tmp_path):
    """Re-running against unchanged feature parquets and unchanged labeling code
    reproduces the same combined_hash -- the same reproducibility promise
    docs/training_dataset_versioning.md Section 2 documents, and the same property
    `test_build_experiment_is_reproducible_for_unchanged_code_and_raw_data` in
    tests/test_features.py checks for the upstream feature parquets."""
    write_all_experiments(tmp_path / "run_a")
    write_all_experiments(tmp_path / "run_b")

    build_training_dataset(processed_dir=tmp_path / "run_a")
    build_training_dataset(processed_dir=tmp_path / "run_b")

    manifest_a = json.loads((tmp_path / "run_a" / "training_dataset_manifest.json").read_text())
    manifest_b = json.loads((tmp_path / "run_b" / "training_dataset_manifest.json").read_text())

    assert manifest_a["combined_hash"] == manifest_b["combined_hash"]
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / "run_a" / "training_dataset.parquet"),
        pd.read_parquet(tmp_path / "run_b" / "training_dataset.parquet"),
    )


def test_combined_hash_changes_when_an_upstream_feature_manifest_changes(tmp_path):
    """If build_dataset.py (#41) is re-run and produces a different feature parquet
    (changed extraction code or changed raw data), that experiment's combined_hash
    changes -- and this join's combined_hash must change too, without this module
    re-reading raw data or re-hashing parquet content itself."""
    write_all_experiments(tmp_path)
    build_training_dataset(processed_dir=tmp_path)
    before = json.loads((tmp_path / "training_dataset_manifest.json").read_text())

    manifest_path = tmp_path / "2nd_test_features_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["combined_hash"] = "a-different-upstream-hash"
    manifest_path.write_text(json.dumps(manifest))

    build_training_dataset(processed_dir=tmp_path)
    after = json.loads((tmp_path / "training_dataset_manifest.json").read_text())

    assert before["combined_hash"] != after["combined_hash"]
    assert before["upstream_feature_version"] != after["upstream_feature_version"]


def test_generating_code_files_in_versioning_is_untouched_by_this_module():
    """Issue #65's constraint restated for #67: this module must not add
    src/labeling.py (or itself) to src/features/versioning.py's GENERATING_CODE_FILES
    -- doing so would invalidate every existing feature-parquet manifest's
    combined_hash and force an unnecessary raw-dataset re-extraction. This module
    defines its own, separate LABELING_CODE_FILES instead
    (docs/training_dataset_versioning.md Section 2)."""
    from src.features.versioning import GENERATING_CODE_FILES

    assert all(f not in GENERATING_CODE_FILES for f in LABELING_CODE_FILES)
    assert {p.name for p in GENERATING_CODE_FILES} == {"extraction.py", "versioning.py"}


# --- compute_labeling_code_hash / compute_upstream_feature_version, in isolation --

def test_compute_labeling_code_hash_changes_with_content(tmp_path):
    f = tmp_path / "module.py"
    f.write_text("x = 1\n")
    original = compute_labeling_code_hash((f,))

    f.write_text("x = 2\n")
    changed = compute_labeling_code_hash((f,))

    assert original != changed


def test_compute_upstream_feature_version_is_order_independent_and_sensitive_to_hashes():
    manifests_ab = {"1st_test": {"combined_hash": "a"}, "2nd_test": {"combined_hash": "b"}}
    manifests_ba = {"2nd_test": {"combined_hash": "b"}, "1st_test": {"combined_hash": "a"}}
    manifests_changed = {"1st_test": {"combined_hash": "a"}, "2nd_test": {"combined_hash": "c"}}

    assert compute_upstream_feature_version(manifests_ab) == compute_upstream_feature_version(manifests_ba)
    assert compute_upstream_feature_version(manifests_ab) != compute_upstream_feature_version(manifests_changed)


# --- main() ------------------------------------------------------------------------

def test_main_prints_a_per_experiment_label_summary(tmp_path, monkeypatch, capsys):
    """build_training_dataset's default processed_dir is bound to the repo-absolute
    PROCESSED_DIR at import time, and main() calls it without arguments -- same
    situation test_features.py's test_main_builds_every_experiment_in_the_registry
    documents for build_dataset.main(). Substituting the function (not its default
    argument) is what keeps this test off the real data/processed/ tree."""
    write_all_experiments(tmp_path)

    def fake_build_training_dataset():
        return build_training_dataset(processed_dir=tmp_path)

    monkeypatch.setattr(
        build_training_dataset_module, "build_training_dataset", fake_build_training_dataset
    )

    main()

    printed = capsys.readouterr().out
    assert "training_dataset.parquet" in printed
    for name in EXPERIMENT_ROWS:
        assert name in printed


# --- integration check against the real, documented dataset ----------------------

_REAL_FEATURE_PARQUET_EXISTS = (PROCESSED_DIR / "1st_test_features.parquet").is_file()


@pytest.mark.skipif(
    not _REAL_FEATURE_PARQUET_EXISTS,
    reason=(
        "requires data/processed/*_features.parquet from a prior `python -m "
        "src.features.build_dataset` run; not present in a from-scratch CI run "
        "because the 'Run unit tests' step runs before 'Execute notebooks' "
        "populates data/processed -- see docs/training_dataset_versioning.md Section 1"
    ),
)
def test_matches_documented_eda_findings_table_on_the_real_dataset(tmp_path):
    """When the real feature parquets are available (e.g. local dev after running
    build_dataset.py, or CI once the notebook-execution step has populated the
    cache), the join must reproduce docs/eda_findings.md Section 3's table exactly:
    per-experiment critical_multiple and Normal/Degrading/Critical counts."""
    for name in EXPERIMENT_ROWS:
        shutil.copy(
            PROCESSED_DIR / f"{name}_features.parquet", tmp_path / f"{name}_features.parquet"
        )
        shutil.copy(
            PROCESSED_DIR / f"{name}_features_manifest.json",
            tmp_path / f"{name}_features_manifest.json",
        )

    written = pd.read_parquet(build_training_dataset(processed_dir=tmp_path))

    expected_critical_multiple = {"1st_test": 1.932, "2nd_test": 2.866, "3rd_test": 3.049}
    expected_counts = {
        "1st_test": {"Normal": 1906, "Degrading": 233, "Critical": 17},
        "2nd_test": {"Normal": 651, "Degrading": 310, "Critical": 23},
        "3rd_test": {"Normal": 6158, "Degrading": 99, "Critical": 67},
    }
    manifest = json.loads((tmp_path / "training_dataset_manifest.json").read_text())

    for name in EXPERIMENT_ROWS:
        assert manifest["critical_multiple"][name] == expected_critical_multiple[name]
        subset = written.loc[written["experiment"] == name]
        assert subset["label"].value_counts().reindex(LABELS).to_dict() == expected_counts[name]

    assert len(written) == 2156 + 984 + 6324
