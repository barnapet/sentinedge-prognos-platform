import numpy as np
import pandas as pd
import pytest

from src.labeling import LABELS
from src.training.evaluation import (
    EXPERIMENTS,
    FEATURE_MATRIX_COLUMNS,
    aggregate,
    fold_metrics,
    loeo_folds,
)
from src.training.imbalance import (
    STRATEGIES,
    make_baseline_model,
    prior_correct,
    random_oversample,
    random_undersample,
)
from src.training.compare_imbalance import run_strategy_on_fold

# Synthetic dataset, not the real one: `data/processed/` is gitignored and, in CI, the
# unit-test step runs before the notebook step that populates it -- same rationale as
# tests/test_build_training_dataset.py.
ROWS_PER_EXPERIMENT = {"1st_test": 40, "2nd_test": 30, "3rd_test": 50}


def make_dataset(rows_per_experiment=ROWS_PER_EXPERIMENT, seed=0) -> pd.DataFrame:
    """A training-dataset-shaped frame with an imbalanced, learnable label structure."""
    rng = np.random.default_rng(seed)
    frames = []
    for name, n in rows_per_experiment.items():
        # Imbalanced on purpose: mostly Normal, few Critical -- the shape the strategies
        # under test exist to handle.
        labels = np.array(["Normal"] * (n - 8) + ["Degrading"] * 5 + ["Critical"] * 3)
        rank = np.array([LABELS.index(v) for v in labels], dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "experiment": name,
                    "file_index": np.arange(n),
                    "rms": 0.1 + 0.1 * rank + rng.normal(0, 0.005, n),
                    "rms_ratio": 1.0 + rank + rng.normal(0, 0.02, n),
                    "kurtosis": 3.0 + 2 * rank + rng.normal(0, 0.05, n),
                    "skewness": rng.normal(0, 0.01, n),
                    "skewness_smoothed": rng.normal(0, 0.01, n),
                    "label": pd.Categorical(labels, categories=LABELS, ordered=True),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# --- loeo_folds: the split itself -------------------------------------------------

def test_produces_one_fold_per_experiment_in_registry_order():
    folds = loeo_folds(make_dataset())

    assert [f.held_out for f in folds] == EXPERIMENTS


def test_held_out_experiment_is_never_in_its_own_training_set():
    """The core property of LOEO (docs/evaluation_protocol.md Section 1): each
    experiment is wholly train or wholly test in a given fold, never both. Checked on
    row counts, since the arrays themselves carry no experiment tag by design."""
    df = make_dataset()
    folds = loeo_folds(df)

    for fold in folds:
        held_out_n = int((df["experiment"] == fold.held_out).sum())
        assert len(fold.X_test) == held_out_n
        assert len(fold.X_train) == len(df) - held_out_n
        assert len(fold.X_train) + len(fold.X_test) == len(df)


def test_train_and_test_row_counts_match_the_other_two_experiments():
    df = make_dataset()

    for fold in loeo_folds(df):
        expected_train = sum(n for name, n in ROWS_PER_EXPERIMENT.items() if name != fold.held_out)
        assert len(fold.y_train) == expected_train
        assert len(fold.y_test) == ROWS_PER_EXPERIMENT[fold.held_out]


def test_experiment_and_index_columns_are_not_fed_to_the_model():
    """`experiment` is the fold key -- feeding it would hand the model the grouping LOEO
    exists to hold out -- and `file_index`/`timestamp` are positional identifiers. The
    default feature matrix must contain neither."""
    assert "experiment" not in FEATURE_MATRIX_COLUMNS
    assert "file_index" not in FEATURE_MATRIX_COLUMNS
    assert "timestamp" not in FEATURE_MATRIX_COLUMNS
    assert "label" not in FEATURE_MATRIX_COLUMNS

    folds = loeo_folds(make_dataset())
    assert folds[0].X_train.shape[1] == len(FEATURE_MATRIX_COLUMNS)


def test_feature_columns_are_overridable_for_the_step_4_ablation():
    """Step 4's rms_ratio ablation must be able to drop a column without reimplementing
    the split (Issue #67 Task 3)."""
    reduced = [c for c in FEATURE_MATRIX_COLUMNS if c != "rms_ratio"]

    folds = loeo_folds(make_dataset(), feature_columns=reduced)

    assert folds[0].X_train.shape[1] == len(FEATURE_MATRIX_COLUMNS) - 1


# --- fold_metrics ------------------------------------------------------------------

def test_perfect_prediction_scores_one_everywhere():
    y = np.array(["Normal", "Degrading", "Critical", "Normal"])

    m = fold_metrics(y, y.copy())

    assert m["recall"] == {"Normal": 1.0, "Degrading": 1.0, "Critical": 1.0}
    assert m["precision"] == {"Normal": 1.0, "Degrading": 1.0, "Critical": 1.0}
    assert m["macro_f1"] == 1.0


def test_never_predicting_critical_scores_zero_recall_not_an_error():
    """The always-Normal degenerate case docs/evaluation_protocol.md Section 4 names:
    high accuracy, zero Critical recall. Scoring it 0 rather than raising is the point --
    "never predicted Critical" is a reportable result, not a missing value."""
    y_true = np.array(["Normal"] * 9 + ["Critical"])
    y_pred = np.array(["Normal"] * 10)

    m = fold_metrics(y_true, y_pred)

    assert m["recall"]["Critical"] == 0.0
    assert m["precision"]["Critical"] == 0.0
    assert m["recall"]["Normal"] == 1.0


def test_confusion_matrix_is_true_by_predicted_in_label_order():
    y_true = np.array(["Normal", "Critical"])
    y_pred = np.array(["Normal", "Degrading"])

    cm = fold_metrics(y_true, y_pred)["confusion_matrix"]

    # rows = true, cols = predicted, order Normal/Degrading/Critical
    assert cm == [[1, 0, 0], [0, 0, 0], [0, 1, 0]]


def test_support_counts_the_true_labels():
    y_true = np.array(["Normal"] * 5 + ["Degrading"] * 2 + ["Critical"])

    m = fold_metrics(y_true, y_true.copy())

    assert m["support"] == {"Normal": 5, "Degrading": 2, "Critical": 1}


# --- aggregate ---------------------------------------------------------------------

def test_aggregate_reports_mean_and_range_not_a_standard_deviation():
    """docs/evaluation_protocol.md Section 5 deliberately excludes std/CI over three
    folds. This pins that choice so it isn't quietly 'improved' later."""
    agg = aggregate([0.2, 0.5, 0.8])

    assert agg["mean"] == pytest.approx(0.5)
    assert agg["min"] == pytest.approx(0.2)
    assert agg["max"] == pytest.approx(0.8)
    assert agg["range"] == pytest.approx(0.6)
    assert "std" not in agg
    assert "ci" not in agg


# --- resampling --------------------------------------------------------------------

def test_random_oversample_equalises_class_counts_upward():
    X = np.arange(20).reshape(-1, 1).astype(float)
    y = np.array(["Normal"] * 16 + ["Critical"] * 4)

    X_res, y_res = random_oversample(X, y)

    counts = dict(zip(*np.unique(y_res, return_counts=True)))
    assert counts == {"Normal": 16, "Critical": 16}
    assert len(X_res) == len(y_res) == 32


def test_random_oversample_only_duplicates_existing_rows():
    """Plain duplication, not synthesis: every resampled row must be one of the
    originals. This is what distinguishes it from SMOTE (see
    docs/class_imbalance_decision.md)."""
    X = np.arange(20).reshape(-1, 1).astype(float)
    y = np.array(["Normal"] * 16 + ["Critical"] * 4)

    X_res, _ = random_oversample(X, y)

    assert set(X_res.ravel()) <= set(X.ravel())


def test_random_undersample_equalises_class_counts_downward():
    X = np.arange(20).reshape(-1, 1).astype(float)
    y = np.array(["Normal"] * 16 + ["Critical"] * 4)

    X_res, y_res = random_undersample(X, y)

    counts = dict(zip(*np.unique(y_res, return_counts=True)))
    assert counts == {"Normal": 4, "Critical": 4}


def test_resampling_is_deterministic_for_a_fixed_seed():
    X = np.arange(20).reshape(-1, 1).astype(float)
    y = np.array(["Normal"] * 16 + ["Critical"] * 4)

    assert np.array_equal(random_oversample(X, y)[0], random_oversample(X, y)[0])
    assert np.array_equal(random_undersample(X, y)[0], random_undersample(X, y)[0])


# --- prior correction ---------------------------------------------------------------

def test_prior_correct_favours_the_rarer_class():
    """The whole point: an equal-probability prediction should break toward whichever
    class was rarer in training."""
    classes = np.array(["Critical", "Normal"])
    y_train = np.array(["Normal"] * 90 + ["Critical"] * 10)
    probabilities = np.array([[0.5, 0.5]])

    corrected = prior_correct(probabilities, classes, y_train)

    assert corrected[0, 0] > corrected[0, 1]  # Critical (rare) now scores higher


def test_prior_correct_uses_training_priors_only():
    """Priors must come from the training fold -- reading the held-out experiment's
    class frequencies would be exactly the leakage docs/evaluation_protocol.md Section 1
    rules out. Verified by the signature accepting only y_train, and by the result
    changing when the training distribution changes."""
    classes = np.array(["Critical", "Normal"])
    probabilities = np.array([[0.5, 0.5]])

    mild = prior_correct(probabilities, classes, np.array(["Normal"] * 60 + ["Critical"] * 40))
    severe = prior_correct(probabilities, classes, np.array(["Normal"] * 99 + ["Critical"] * 1))

    assert severe[0, 0] / severe[0, 1] > mild[0, 0] / mild[0, 1]


# --- strategies and the runner --------------------------------------------------------

def test_every_strategy_has_a_control_and_covers_the_three_proposed_directions():
    """Issue #21 proposes class weighting, resampling, and threshold-moving/
    cost-sensitive. All three must be represented, plus an untreated control -- without
    the control there is no way to tell whether any handling helped at all."""
    names = [s.name for s in STRATEGIES]

    assert "none" in names
    assert "class_weight_balanced" in names
    assert {"random_oversample", "random_undersample"} <= set(names)
    assert "prior_correction" in names


def test_scaler_is_inside_the_pipeline_so_it_fits_on_training_rows_only():
    """docs/evaluation_protocol.md Section 1 requires any fitted preprocessing to be fit
    on the fold's training rows only. Keeping StandardScaler inside the Pipeline is what
    enforces that; a scaler fitted outside, on the pooled data, would leak the held-out
    experiment's distribution into its own evaluation."""
    model = make_baseline_model()

    assert [name for name, _ in model.steps] == ["scaler", "clf"]


def test_no_strategy_tunes_hyperparameters_per_fold():
    """The leakage guard described in src/training/imbalance.py's docstring: every arm
    uses the same fixed model, so the only thing differing between arms is the imbalance
    treatment. Two models built from the same strategy must be identically configured."""
    for strategy in STRATEGIES:
        a, b = strategy.build_model(), strategy.build_model()
        assert a.get_params()["clf__C"] == b.get_params()["clf__C"]
        assert a.get_params()["clf__max_iter"] == b.get_params()["clf__max_iter"]

    weights = {s.name: s.build_model().get_params()["clf__class_weight"] for s in STRATEGIES}
    assert weights["class_weight_balanced"] == "balanced"
    assert weights["none"] is None


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.name)
def test_every_strategy_runs_end_to_end_and_reports_the_committed_metrics(strategy):
    folds = loeo_folds(make_dataset())

    metrics = run_strategy_on_fold(strategy, folds[0])

    assert metrics["held_out"] == "1st_test"
    assert set(metrics["recall"]) == set(LABELS)
    assert set(metrics["precision"]) == set(LABELS)
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert np.array(metrics["confusion_matrix"]).shape == (3, 3)


def test_resampling_strategies_change_the_training_row_count_but_not_the_test_set():
    """Resampling must touch the training rows only -- the held-out experiment has to
    stay exactly as observed or the evaluation means nothing."""
    folds = loeo_folds(make_dataset())
    fold = folds[0]
    by_name = {s.name: s for s in STRATEGIES}

    control = run_strategy_on_fold(by_name["none"], fold)
    over = run_strategy_on_fold(by_name["random_oversample"], fold)
    under = run_strategy_on_fold(by_name["random_undersample"], fold)

    assert over["n_train_rows"] > control["n_train_rows"]
    assert under["n_train_rows"] < control["n_train_rows"]
    # test-set support is identical across all three -- the held-out rows never resampled
    assert over["support"] == control["support"] == under["support"]
