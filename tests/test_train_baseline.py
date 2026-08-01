"""Tests for the M3 baseline training run, ablation, and diagnostics (Issue #72).

The leakage guards carried over from Issue #21 (`tests/test_training.py`) are re-pinned
here against the Step 4 entrypoint specifically, because #72 adds a second axis of
variation (the feature set) and the guards have to hold across it, not just across
imbalance strategies: no hyperparameter is tuned per fold *or per feature set*, and any
fitted preprocessing sees training-fold rows only.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.labeling import LABELS
from src.training.candidate_scalers import (
    averaged_per_experiment_moments,
    evaluate_averaged_per_experiment_scaler,
    evaluate_transductive_scaler,
)
from src.training.evaluation import FEATURE_MATRIX_COLUMNS, loeo_folds
from src.training.imbalance import BASELINE_MODEL_PARAMS
from src.training.train_baseline import (
    BASELINE_STRATEGY,
    FEATURE_SETS,
    critical_band_summary,
    raw_rms_scale_summary,
    run_all_feature_sets,
    run_feature_set,
)

# Synthetic, not the real dataset: `data/processed/` is gitignored and in CI the unit-test
# step runs before the notebook step that populates it -- same rationale as
# tests/test_training.py and tests/test_build_training_dataset.py.
ROWS_PER_EXPERIMENT = {"1st_test": 40, "2nd_test": 30, "3rd_test": 50}

# Per-experiment raw-RMS offsets, chosen so one experiment's *minimum* raw rms sits above
# the others' means -- the shape of the real `1st_test` scale problem
# (docs/class_imbalance_decision.md Section 4), reproduced small enough to assert on.
RMS_OFFSET = {"1st_test": 1.0, "2nd_test": 0.0, "3rd_test": 0.0}

# Per-experiment Critical rms_ratio bands, non-overlapping the way the real experiments'
# are: 1st_test's Critical band sits entirely below the others' (docs/eda_findings.md
# Section 3 -- critical_multiple is 1.932 / 2.866 / 3.049).
CRITICAL_RATIO = {"1st_test": 2.0, "2nd_test": 3.0, "3rd_test": 3.2}


def make_dataset(rows_per_experiment=ROWS_PER_EXPERIMENT, seed=0) -> pd.DataFrame:
    """A training-dataset-shaped frame reproducing both diagnosed pathologies in miniature."""
    rng = np.random.default_rng(seed)
    frames = []
    for name, n in rows_per_experiment.items():
        labels = np.array(["Normal"] * (n - 8) + ["Degrading"] * 5 + ["Critical"] * 3)
        rank = np.array([LABELS.index(v) for v in labels], dtype=float)
        # Critical rows sit at that experiment's own Critical band; lower ranks below it.
        ratio = np.where(rank == 2, CRITICAL_RATIO[name], 1.0 + 0.3 * rank)
        frames.append(
            pd.DataFrame(
                {
                    "experiment": name,
                    "file_index": np.arange(n),
                    "rms": RMS_OFFSET[name] + 0.1 + 0.1 * rank + rng.normal(0, 0.005, n),
                    "rms_ratio": ratio + rng.normal(0, 0.01, n),
                    "kurtosis": 3.0 + 2 * rank + rng.normal(0, 0.05, n),
                    "skewness": rng.normal(0, 0.01, n),
                    "skewness_smoothed": rng.normal(0, 0.01, n),
                    "label": pd.Categorical(labels, categories=LABELS, ordered=True),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# --- leakage guard: no hyperparameter tuning ----------------------------------------

def test_baseline_strategy_is_issue_21s_adopted_decision():
    """Issue #72 Task 1 specifies class_weight='balanced' -- #21's decision
    (docs/class_imbalance_decision.md Section 5) -- and the same model family. This pins
    that the Step 4 entrypoint did not quietly re-open either."""
    assert BASELINE_STRATEGY.name == "class_weight_balanced"

    model = BASELINE_STRATEGY.build_model()

    assert model.get_params()["clf__class_weight"] == "balanced"


def test_no_hyperparameter_is_tuned_per_fold_or_per_feature_set():
    """The leakage guard from src/training/imbalance.py's docstring, extended to #72's
    second axis of variation. docs/evaluation_protocol.md Section 1 requires that
    hyperparameter selection never see the held-out experiment; this module removes the
    mechanism entirely rather than managing it, so every model built -- for any fold, for
    any feature set -- must be identically configured."""
    built = [BASELINE_STRATEGY.build_model() for _ in FEATURE_SETS]

    for model in built:
        params = model.get_params()
        assert params["clf__C"] == built[0].get_params()["clf__C"]
        assert params["clf__max_iter"] == BASELINE_MODEL_PARAMS["max_iter"]
        assert params["clf__random_state"] == BASELINE_MODEL_PARAMS["random_state"]
        assert params["clf__class_weight"] == "balanced"


def test_only_the_feature_columns_differ_between_configurations():
    """The one-variable property of this comparison: the configurations differ in which
    columns they feed the model and in nothing else."""
    df = make_dataset()

    results = run_all_feature_sets(df=df)

    assert set(results) == set(FEATURE_SETS)
    for name, res in results.items():
        assert res["feature_columns"] == FEATURE_SETS[name]
    # every configuration is a strict subset of the full column list -- nothing is
    # engineered or renamed between arms, only dropped
    for name, res in results.items():
        assert set(res["feature_columns"]) <= set(FEATURE_SETS["full"])


# --- leakage guard: preprocessing fit on training-fold rows only ----------------------

def test_scaler_is_fitted_on_training_rows_only_not_the_pooled_dataset():
    """docs/evaluation_protocol.md Section 1's requirement, asserted behaviourally rather
    than structurally: the fitted scaler's centre must equal the fold's *training* rows'
    mean, and must differ from the pooled mean. A scaler fitted on the pooled dataset
    would leak the held-out experiment's own distribution into its evaluation."""
    df = make_dataset()
    columns = FEATURE_SETS["full"]
    fold = loeo_folds(df, feature_columns=columns)[0]

    model = BASELINE_STRATEGY.build_model()
    model.fit(fold.X_train, fold.y_train)
    scaler = model.named_steps["scaler"]

    np.testing.assert_allclose(scaler.mean_, fold.X_train.mean(axis=0))
    pooled_mean = df[columns].to_numpy().mean(axis=0)
    assert not np.allclose(scaler.mean_, pooled_mean)


def test_scaler_never_sees_the_held_out_experiments_rows():
    """Stronger form of the above: changing the held-out experiment's feature values must
    not change the fitted scaler at all. If it did, the held-out rows were in the fit."""
    df = make_dataset()
    columns = FEATURE_SETS["full"]
    fold = loeo_folds(df, feature_columns=columns)[0]
    baseline = BASELINE_STRATEGY.build_model().fit(fold.X_train, fold.y_train)

    perturbed = df.copy()
    is_held_out = perturbed["experiment"] == fold.held_out
    perturbed.loc[is_held_out, columns] = perturbed.loc[is_held_out, columns] + 100.0
    perturbed_fold = loeo_folds(perturbed, feature_columns=columns)[0]
    after = BASELINE_STRATEGY.build_model().fit(perturbed_fold.X_train, perturbed_fold.y_train)

    np.testing.assert_allclose(
        baseline.named_steps["scaler"].mean_, after.named_steps["scaler"].mean_
    )


# --- feature-set configurations -------------------------------------------------------

def test_ablation_configuration_drops_exactly_rms_ratio():
    """Issue #72 Task 2's required comparison: full feature set vs rms_ratio-excluded."""
    assert "rms_ratio" in FEATURE_SETS["full"]
    assert "rms_ratio" not in FEATURE_SETS["no_rms_ratio"]
    assert set(FEATURE_SETS["full"]) - set(FEATURE_SETS["no_rms_ratio"]) == {"rms_ratio"}


def test_scale_fix_configuration_drops_exactly_raw_rms():
    """Task 3's candidate fix: raw `rms` is the absolute amplitude channel whose scale
    does not transfer between bearings; `rms_ratio` is the per-experiment-normalised one."""
    assert set(FEATURE_SETS["full"]) - set(FEATURE_SETS["no_raw_rms"]) == {"rms"}


def test_floor_configuration_drops_both_rms_derived_features():
    assert set(FEATURE_SETS["kurtosis_skewness_only"]) == set(FEATURE_MATRIX_COLUMNS) - {
        "rms",
        "rms_ratio",
    }


# --- diagnostics ----------------------------------------------------------------------

def test_raw_rms_scale_summary_exposes_the_amplitude_mismatch():
    """Issue #72's "diagnosed with evidence, not just described" criterion: the scale
    claim must be recomputable, not cited. On this fixture 1st_test's minimum raw rms is
    built to exceed the other experiments' means, as it does in the real data."""
    summary = raw_rms_scale_summary(make_dataset())

    assert summary["1st_test"]["min"] > summary["2nd_test"]["mean"]
    assert summary["1st_test"]["min"] > summary["3rd_test"]["mean"]


def test_critical_band_summary_detects_an_unreachable_held_out_band():
    """The threshold-transfer diagnostic. When the held-out experiment's whole Critical
    band sits below the lowest rms_ratio its training fold ever labelled Critical, every
    one of its Critical rows is unreachable by a monotone boundary -- which is what
    docs/evaluation_protocol.md Section 2 predicted per-experiment critical_multiple
    would cause."""
    summary = critical_band_summary(make_dataset())

    first = summary["1st_test"]
    assert first["test_critical_rms_ratio_max"] < first["train_min_critical_rms_ratio"]
    assert first["test_critical_rows_below_train_min"] == first["test_critical_rows"]

    # The other two folds train on a fold containing 1st_test's lower band, so their own
    # Critical rows are reachable -- the asymmetry is the point.
    assert summary["2nd_test"]["test_critical_rows_below_train_min"] == 0
    assert summary["3rd_test"]["test_critical_rows_below_train_min"] == 0


# --- end to end -----------------------------------------------------------------------

@pytest.mark.parametrize("feature_set", sorted(FEATURE_SETS), ids=lambda n: n)
def test_every_feature_set_runs_and_reports_the_committed_metrics(feature_set):
    df = make_dataset()

    res = run_feature_set(FEATURE_SETS[feature_set], df=df)

    assert [f["held_out"] for f in res["per_fold"]] == list(ROWS_PER_EXPERIMENT)
    for fold_result in res["per_fold"]:
        assert set(fold_result["recall"]) == set(LABELS)
        assert np.array(fold_result["confusion_matrix"]).shape == (3, 3)


def test_aggregate_reports_mean_and_range_but_no_standard_deviation():
    """docs/evaluation_protocol.md Section 5 deliberately excludes std/CI over three
    folds. Re-pinned at this entrypoint so it isn't quietly 'improved' later."""
    res = run_feature_set(FEATURE_SETS["full"], df=make_dataset())

    for metric in ("critical_recall", "critical_precision", "macro_f1", "normal_recall"):
        agg = res["aggregate"][metric]
        assert {"mean", "min", "max", "range"} <= set(agg)
        assert "std" not in agg
        assert "ci" not in agg


def test_held_out_class_support_is_identical_across_feature_sets():
    """Changing which columns the model sees must not change the held-out rows being
    scored -- otherwise the configurations would not be comparable."""
    df = make_dataset()

    results = run_all_feature_sets(df=df)

    supports = [
        [f["support"] for f in res["per_fold"]] for res in results.values()
    ]
    assert all(s == supports[0] for s in supports)


# --- rejected scaling candidates -------------------------------------------------------

def test_averaged_per_experiment_moments_never_see_the_held_out_experiment():
    """Candidate A is rejected for being ineffective, not for leaking -- but it is only
    fair to reject it on those grounds if it is genuinely leakage-safe. Perturbing the
    held-out experiment must leave its fitted moments unchanged."""
    df = make_dataset()
    columns = FEATURE_SETS["full"]
    train = df[df["experiment"] != "1st_test"]

    before = averaged_per_experiment_moments(train, columns)
    perturbed = df.copy()
    is_held_out = perturbed["experiment"] == "1st_test"
    perturbed.loc[is_held_out, columns] = perturbed.loc[is_held_out, columns] + 100.0
    after = averaged_per_experiment_moments(perturbed[perturbed["experiment"] != "1st_test"], columns)

    np.testing.assert_allclose(before[0], after[0])
    np.testing.assert_allclose(before[1], after[1])


def test_averaged_per_experiment_moments_weight_experiments_not_rows():
    """The whole idea of Candidate A: equal weight per experiment, so the moments are not
    dominated by whichever experiment contributes most rows."""
    df = make_dataset({"2nd_test": 10, "3rd_test": 1000})
    columns = ["rms"]

    mean, _ = averaged_per_experiment_moments(df, columns)

    per_experiment = [g[columns].mean().to_numpy()[0] for _, g in df.groupby("experiment")]
    assert mean[0] == pytest.approx(float(np.mean(per_experiment)))
    assert mean[0] != pytest.approx(float(df[columns].mean().iloc[0]))


def test_the_protocol_violating_candidate_is_not_wired_into_the_adopted_path():
    """Candidate C fits a scaler on the held-out experiment's rows, which
    docs/evaluation_protocol.md Section 1 forbids. It exists only to quantify a declined
    fix, so the adopted training entrypoint must not depend on this module at all."""
    import src.training.train_baseline as train_baseline

    source = Path(train_baseline.__file__).read_text()

    assert "candidate_scalers" not in source


@pytest.mark.parametrize(
    "evaluate", [evaluate_averaged_per_experiment_scaler, evaluate_transductive_scaler]
)
def test_rejected_candidates_still_run_and_report_the_committed_metrics(evaluate):
    """Kept runnable, per the src/features/candidate_features.py convention, so the
    rejection in docs/model_training_decision.md Section 4 can be re-checked rather than
    trusted."""
    results = evaluate(make_dataset(), FEATURE_SETS["full"])

    assert set(results) == set(ROWS_PER_EXPERIMENT)
    for held_out, metrics in results.items():
        assert metrics["held_out"] == held_out
        assert set(metrics["recall"]) == set(LABELS)
