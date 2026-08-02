"""Tests for the online serving feature path (Issue #82).

The load-bearing test here is the equivalence suite at the bottom: a whole experiment's
files replayed one at a time through `src.serving`, compared value-by-value against what
`src/features/extraction.py` computes for the same experiment in batch. Everything above it
covers the state machine's edges -- the first nine files (window not yet full) and the 50th
(baseline locks) -- in isolation, where a failure names the cause instead of just the
symptom.
"""
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import pytest

from src.features.extraction import (
    BASELINE_N_FILES,
    ROLLING_WINDOW,
    compute_kurtosis,
    compute_rms,
    compute_skewness,
    extract_experiment_features,
    list_snapshot_files,
    load_channel,
)
from src.labeling import LABELS
from src.serving.features import (
    SERVING_FEATURE_COLUMNS,
    OnlineFeatureExtractor,
    OnlineFeatures,
    compute_online_features,
)
from src.serving.state import (
    STABLE,
    WARMING_UP,
    BearingState,
    BearingStateStore,
    window_mean,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Incremental rolling means are not bit-identical to `pandas`' -- see `window_mean`'s
# docstring for why (one Kahan-compensated running sum across the whole column vs. a fresh
# sum over the current window). Measured maximum across all three real experiments: 2 ULP.
# The bound asserted here leaves one bit of headroom for a `pandas` summation-order change,
# and is still ~13 orders of magnitude tighter than anything that could move a prediction.
MAX_ULP = 4

# The first file index whose `rms_ratio` is computed against the final, locked 50-file
# baseline -- i.e. the 50th file. Before it, serving deliberately differs from batch.
FIRST_STABLE_INDEX = BASELINE_N_FILES - 1


def max_ulp_diff(actual, expected) -> float:
    """Largest difference between two float sequences, in units in the last place.

    ULP rather than a relative tolerance because the claim being tested is "the same
    number, up to floating-point summation order", and a fixed `rel=` would either be
    loose enough to hide a real divergence or tight enough to be arbitrary.
    """
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    diff = np.abs(actual - expected)
    return float(np.max(np.where(diff == 0.0, 0.0, diff / np.spacing(np.abs(expected)))))


def write_snapshot(path, columns):
    """Tab-separated snapshot file, same fixture shape as tests/test_features.py."""
    n_rows = len(columns[0])
    with open(path, "w") as f:
        for row in range(n_rows):
            f.write("\t".join(str(col[row]) for col in columns) + "\n")


def write_synthetic_experiment(raw_dir: Path, n_files: int, n_samples: int = 128) -> Path:
    """A synthetic run-to-failure experiment: `n_files` snapshots, amplitude drifting up.

    Long enough to cross both boundaries this module has (the 10-file window filling at
    file 9, the 50-file baseline locking at file 49) and to leave a substantial stretch
    beyond them. Amplitude and skew both drift with file index so no two files share a
    feature value and a stalled or off-by-one history cannot pass by coincidence.
    """
    raw_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    start = datetime(2004, 2, 12, 10, 32, 39)
    for i in range(n_files):
        amplitude = 0.08 * (1.0 + 0.05 * i)
        signal = rng.normal(0.0, amplitude, n_samples) + 0.01 * i * rng.gamma(2.0, 0.5, n_samples)
        name = (start + timedelta(minutes=10 * i)).strftime("%Y.%m.%d.%H.%M.%S")
        write_snapshot(raw_dir / name, [signal.tolist()])
    return raw_dir


def replay(raw_dir: Path, channel_idx: int = 0, bearing_id: str = "bearing-under-test"):
    """Feed every file of `raw_dir` through the online path, one at a time, in order.

    Returns a frame with one row per file, in the same column names the batch pipeline
    uses, so the two can be compared column by column.
    """
    extractor = OnlineFeatureExtractor()
    rows = [
        extractor.observe(bearing_id, load_channel(path, channel_idx)).as_dict()
        for path in list_snapshot_files(raw_dir)
    ]
    return pd.DataFrame.from_records(rows)


# --- window_mean ------------------------------------------------------------------

def test_window_mean_matches_a_full_pandas_rolling_window():
    values = [1.0, 2.0, 3.0, 4.0]
    assert window_mean(values) == pytest.approx(pd.Series(values).mean())


def test_window_mean_is_exactly_rounded_not_the_builtin_sum():
    """Regression test for a real CI failure on this issue's PR: the builtin `sum()` uses
    Neumaier compensated summation for floats on CPython 3.12+ and naive summation before
    it, so these 50 identical values average to 0.08 on 3.12 and 0.08000000000000003 on
    3.11. A serving process must not produce version-dependent features from
    version-independent inputs -- `math.fsum` is exactly rounded on every version."""
    assert window_mean([0.08] * BASELINE_N_FILES) == 0.08


def test_window_mean_of_a_single_value_is_that_value():
    """File 0's "rolling mean" is just its own RMS -- `min_periods=1`'s behaviour, and
    what makes file 0's `rms_ratio` exactly 1.0 (docs/feature_windowing_decision.md §3)."""
    assert window_mean([0.0812]) == 0.0812


# --- BearingState: the 10-file rolling window -------------------------------------

def test_rolling_history_is_bounded_to_the_window_length():
    state = BearingState()
    for i in range(ROLLING_WINDOW + 5):
        state.observe(rms=float(i), skewness=float(-i))

    assert len(state.rms_history) == ROLLING_WINDOW
    assert len(state.skewness_history) == ROLLING_WINDOW
    assert list(state.rms_history) == [float(i) for i in range(5, ROLLING_WINDOW + 5)]


def test_first_nine_files_reproduce_min_periods_1_shrinking_window():
    """The edge `docs/feature_windowing_decision.md` §3 pins: for file `i < 9` the batch
    rolling mean averages the `i + 1` files available rather than returning `NaN`. The
    deque reproduces that only if it is averaged at its current length, not padded to 10 --
    checked at every one of those nine indices, not just the last."""
    rms_values = [0.07, 0.09, 0.06, 0.11, 0.08, 0.10, 0.07, 0.12, 0.09, 0.08, 0.13]
    expected = pd.Series(rms_values).rolling(ROLLING_WINDOW, min_periods=1).mean()

    state = BearingState()
    for i, value in enumerate(rms_values):
        state.observe(rms=value, skewness=value)
        assert state.rolling_rms == pytest.approx(expected.iloc[i])
        assert state.rolling_skewness == pytest.approx(expected.iloc[i])
        assert not np.isnan(state.rolling_rms)


# --- BearingState: the 50-file cold-start baseline --------------------------------

def test_baseline_locks_on_the_50th_file_not_the_49th_or_51st():
    """The off-by-one this whole module turns on. `add_rolling_rms_ratio` averages
    `head(50)` -- files 0..49 inclusive -- so the baseline must be unset after 49
    observations and equal to the mean of exactly those first 50 after the 50th."""
    rms_values = [0.05 + 0.001 * i for i in range(BASELINE_N_FILES + 10)]
    state = BearingState()

    for value in rms_values[: BASELINE_N_FILES - 1]:
        state.observe(rms=value, skewness=0.0)
    assert state.file_count == BASELINE_N_FILES - 1
    assert state.baseline_rms is None

    state.observe(rms=rms_values[BASELINE_N_FILES - 1], skewness=0.0)
    assert state.file_count == BASELINE_N_FILES
    # Compared within MAX_ULP, not with `==`: this averages the same 50 values pandas does,
    # but not through pandas' accumulator (see `window_mean`). What must be exact is *which*
    # 50 values -- an off-by-one would move this by ~0.001, four orders of magnitude out.
    expected = pd.Series(rms_values).head(BASELINE_N_FILES).mean()
    assert max_ulp_diff([state.baseline_rms], [expected]) <= MAX_ULP
    assert state.baseline_rms != pd.Series(rms_values).head(BASELINE_N_FILES + 1).mean()


def test_baseline_is_frozen_once_locked_even_against_extreme_later_files():
    """A run-to-failure bearing's later RMS is many times its baseline; if any of it leaked
    into the baseline, `rms_ratio` would compress exactly where the model needs it to grow."""
    state = BearingState()
    for _ in range(BASELINE_N_FILES):
        state.observe(rms=0.08, skewness=0.0)
    locked = state.baseline_rms

    for _ in range(200):
        state.observe(rms=5.0, skewness=1.0)

    assert state.baseline_rms == locked  # exact: the stored scalar must not be recomputed
    assert locked == pytest.approx(0.08)
    assert state.effective_baseline_rms == locked


def test_raw_baseline_values_are_dropped_once_the_mean_is_locked():
    """docs/serving_design.md §2: after the 50th file only the scalar needs keeping."""
    state = BearingState()
    for i in range(BASELINE_N_FILES):
        state.observe(rms=0.05 + 0.001 * i, skewness=0.0)

    assert state.baseline_rms_values == []
    assert state.baseline_rms is not None


def test_baseline_status_switches_at_exactly_the_50th_file_and_never_reverts():
    """docs/serving_design.md §3's cold-start rule, at the boundary and beyond it."""
    state = BearingState()
    for i in range(BASELINE_N_FILES - 1):
        state.observe(rms=0.08, skewness=0.0)
        assert state.baseline_status == WARMING_UP, f"file index {i} should be warming up"

    state.observe(rms=0.08, skewness=0.0)  # the 50th file, index 49
    assert state.file_count == BASELINE_N_FILES
    assert state.baseline_status == STABLE

    for _ in range(100):
        state.observe(rms=5.0, skewness=1.0)
        assert state.baseline_status == STABLE


def test_expanding_baseline_is_used_while_warming_up():
    """Before the lock, `rms_ratio`'s denominator is the mean of what has been seen so far
    (1 to 49 files) -- not a placeholder, not the eventual 50-file mean (§3)."""
    state = BearingState()
    values = [0.06, 0.10, 0.08]
    for i, value in enumerate(values):
        state.observe(rms=value, skewness=0.0)
        assert state.effective_baseline_rms == pytest.approx(np.mean(values[: i + 1]))


def test_first_file_of_a_bearing_has_rms_ratio_exactly_one():
    """With one file seen, the 10-file window and the expanding baseline are the same
    single value, so the ratio is 1.0 by construction -- a useful sanity anchor for the
    cold-start rule, and the value a dashboard will see on a bearing's very first frame."""
    signal = np.array([0.1, -0.2, 0.15, -0.05, 0.3, -0.25], dtype=np.float32)
    features = compute_online_features(signal, BearingState())

    assert features.rms_ratio == 1.0
    assert features.skewness_smoothed == features.skewness
    assert features.baseline_status == WARMING_UP
    assert features.file_count == 1


# --- BearingStateStore ------------------------------------------------------------

def test_store_creates_state_on_first_sight_and_reuses_it_afterwards():
    store = BearingStateStore()
    assert "1st_test-bearing3" not in store

    first = store.get_or_create("1st_test-bearing3")
    second = store.get_or_create("1st_test-bearing3")

    assert first is second
    assert "1st_test-bearing3" in store
    assert len(store) == 1


def test_store_keeps_bearings_independent():
    """Two bearings served by the same process must not share rolling history -- the
    failure mode would be silent, since both would still return plausible numbers."""
    store = BearingStateStore()
    signal_a = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
    signal_b = np.array([2.0, -2.0, 3.0, -3.0], dtype=np.float32)

    for _ in range(5):
        compute_online_features(signal_a, store.get_or_create("bearing-a"))
    features_b = compute_online_features(signal_b, store.get_or_create("bearing-b"))

    assert store.get_or_create("bearing-a").file_count == 5
    assert features_b.file_count == 1
    assert features_b.rms == pytest.approx(compute_rms(signal_b))
    assert features_b.rms_ratio == 1.0  # bearing-b's own first file, not bearing-a's sixth


def test_store_reset_restarts_a_bearing_at_file_zero():
    store = BearingStateStore()
    signal = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
    for _ in range(3):
        compute_online_features(signal, store.get_or_create("bearing-a"))
    store.get_or_create("bearing-b")

    store.reset("bearing-a")

    assert store.get_or_create("bearing-a").file_count == 0
    assert "bearing-b" in store

    store.reset()
    assert len(store) == 0


# --- OnlineFeatures ---------------------------------------------------------------

def test_stateless_features_are_the_batch_functions_not_a_reimplementation():
    """`rms`/`kurtosis`/`skewness` must be bit-identical to `extraction.py`'s -- these are
    the same calls, so equality here is exact, not approximate."""
    signal = np.array([0.12, -0.31, 0.05, 0.44, -0.19, 0.27, -0.08], dtype=np.float32)
    features = compute_online_features(signal, BearingState())

    assert features.rms == compute_rms(signal)
    assert features.kurtosis == compute_kurtosis(signal)
    assert features.skewness == compute_skewness(signal)


def test_feature_vector_order_matches_the_persisted_serving_model():
    """The ordering that cannot be checked by the type system: `serving_model.joblib` is a
    fitted `Pipeline` whose `StandardScaler` holds per-column means, so a permuted vector
    would be mis-scaled silently rather than rejected. Cross-checked against the committed
    manifest (Issue #80), which records the columns the model was actually fitted on."""
    manifest = json.loads((REPO_ROOT / "models" / "serving_model_manifest.json").read_text())
    features = OnlineFeatures(
        rms=0.1,
        rms_ratio=1.2,
        kurtosis=3.4,
        skewness=0.02,
        skewness_smoothed=0.03,
        baseline_status=WARMING_UP,
        file_count=7,
    )

    assert SERVING_FEATURE_COLUMNS == manifest["feature_columns"]
    assert features.feature_vector() == [0.1, 1.2, 3.4, 0.02, 0.03]


def test_online_features_feed_the_persisted_model_without_reshaping():
    """End-to-end joint between this issue's output and Issue #80's artifact: the vector
    this module produces is directly scoreable. No API code -- just the two halves meeting."""
    model = joblib.load(REPO_ROOT / "models" / "serving_model.joblib")
    signal = np.random.default_rng(0).normal(0.0, 0.08, 512)
    features = compute_online_features(signal, BearingState())

    prediction = model.predict([features.feature_vector()])

    assert prediction[0] in LABELS


def test_as_dict_carries_the_cold_start_disclosure_alongside_the_features():
    features = compute_online_features(
        np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32), BearingState()
    )

    payload = features.as_dict()

    assert set(payload) == set(SERVING_FEATURE_COLUMNS) | {"baseline_status", "file_count"}
    assert payload["baseline_status"] == WARMING_UP


# --- Equivalence with the batch pipeline (Issue #82's central claim) --------------
#
# These replay a complete experiment one file at a time and compare every column against
# `extract_experiment_features`'s batch output for the same files. The synthetic case runs
# everywhere; the real-data case runs wherever `data/raw/` has been fetched (CI does, per
# .github/workflows -- the dataset is restored before the test step).


def assert_incremental_matches_batch(batch: pd.DataFrame, online: pd.DataFrame) -> None:
    """Every column must match, with exactly one documented exception.

    - `rms`/`kurtosis`/`skewness`: the same function called on the same signal -- exact.
    - `skewness_smoothed` and, from the 50th file on, `rms_ratio`: the same rolling mean
      computed incrementally -- equal to within `MAX_ULP` (see `window_mean`).
    - `rms_ratio` before the 50th file: deliberately different, because serving's baseline
      cannot look ahead to files it has not been sent (docs/serving_design.md §3). Asserted
      to *actually* differ, so this exclusion can never quietly hide a broken warmup.
    """
    assert len(online) == len(batch)

    for column in ["rms", "kurtosis", "skewness"]:
        assert online[column].tolist() == batch[column].tolist(), f"{column} must be exact"

    assert max_ulp_diff(online["skewness_smoothed"], batch["skewness_smoothed"]) <= MAX_ULP

    online_ratio = online["rms_ratio"].to_numpy()
    batch_ratio = batch["rms_ratio"].to_numpy()
    stable = slice(FIRST_STABLE_INDEX, None)
    warming = slice(0, FIRST_STABLE_INDEX)
    assert max_ulp_diff(online_ratio[stable], batch_ratio[stable]) <= MAX_ULP
    assert (
        max_ulp_diff(online_ratio[warming], batch_ratio[warming]) > MAX_ULP
    ), "the warming-up baseline must differ from the batch look-ahead baseline"

    status = online["baseline_status"].to_numpy()
    assert (status[warming] == WARMING_UP).all()
    assert (status[stable] == STABLE).all()
    assert online["file_count"].tolist() == [i + 1 for i in range(len(online))]


def test_incremental_replay_matches_batch_extraction_on_a_full_experiment(tmp_path):
    """The core equivalence claim, on a complete synthetic experiment (140 files, well past
    both the 10-file window and the 50-file baseline lock). Runs with no dataset present."""
    raw_dir = write_synthetic_experiment(tmp_path / "synthetic_test", n_files=140)

    batch = extract_experiment_features(raw_dir, experiment="synthetic_test", channel_idx=0)
    online = replay(raw_dir)

    assert_incremental_matches_batch(batch, online)


def test_replaying_two_bearings_interleaved_matches_replaying_them_separately(tmp_path):
    """Serving does not get one bearing's files in an uninterrupted block. Interleaving two
    bearings through one store must not perturb either one's rolling history."""
    dir_a = write_synthetic_experiment(tmp_path / "a", n_files=60)
    dir_b = write_synthetic_experiment(tmp_path / "b", n_files=60, n_samples=96)

    separate = {"a": replay(dir_a), "b": replay(dir_b)}

    extractor = OnlineFeatureExtractor()
    interleaved = {"a": [], "b": []}
    for path_a, path_b in zip(list_snapshot_files(dir_a), list_snapshot_files(dir_b)):
        interleaved["a"].append(extractor.observe("a", load_channel(path_a, 0)).as_dict())
        interleaved["b"].append(extractor.observe("b", load_channel(path_b, 0)).as_dict())

    for bearing in ("a", "b"):
        pd.testing.assert_frame_equal(
            pd.DataFrame.from_records(interleaved[bearing]), separate[bearing]
        )


REAL_EXPERIMENT = "2nd_test"  # the smallest of the three (984 files), per docs/eda_findings.md §1
REAL_RAW_DIR = REPO_ROOT / "data" / "raw" / REAL_EXPERIMENT


@pytest.mark.skipif(
    not REAL_RAW_DIR.is_dir(),
    reason=f"data/raw/{REAL_EXPERIMENT}/ not present (see data/README.md); "
    "the synthetic full-experiment replay above covers the same claim without it",
)
def test_incremental_replay_matches_batch_extraction_on_a_real_experiment():
    """The same claim on real vibration data: all 984 files of `2nd_test`, replayed
    one at a time, against `extract_experiment_features`'s batch output for the same run.

    Synthetic signals cannot rule out a divergence that only shows up on the real data's
    dynamic range -- `2nd_test`'s RMS grows roughly 6x from baseline to failure, and the
    rolling window spans that growth, which is exactly where a summation-order difference
    would be largest if it mattered.
    """
    from src.features.extraction import EXPERIMENTS

    channel_idx = EXPERIMENTS[REAL_EXPERIMENT].channel_idx
    batch = extract_experiment_features(
        REAL_RAW_DIR, experiment=REAL_EXPERIMENT, channel_idx=channel_idx
    )
    online = replay(REAL_RAW_DIR, channel_idx=channel_idx)

    assert_incremental_matches_batch(batch, online)
