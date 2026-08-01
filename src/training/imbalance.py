"""Class-imbalance handling strategies to compare for M3 (Issue #21).

The dataset is 81:1 Normal:Critical pooled (`docs/eda_findings.md` Section 3), so an
unweighted classifier can score >90% accuracy while never predicting the class the
project exists to catch. Issue #21 proposes three directions -- class weighting,
resampling, and threshold-moving/cost-sensitive decisions -- and asks for at least two
to be compared. All three are implemented here, plus an unhandled control, because
without the control there is no way to tell whether any handling helped at all.

Each strategy is a `(name, description, factory)` triple where the factory returns a
fresh, unfitted estimator, plus an optional resampler applied to the training rows only.
See `docs/class_imbalance_decision.md` for the comparison results and the decision.

**No hyperparameter tuning anywhere in this module.** Every strategy uses the same fixed
model with fixed hyperparameters (see `BASELINE_MODEL_PARAMS`). This is a deliberate
leakage-avoidance choice, not laziness: with only three folds, any tuning procedure would
have to select parameters using data from folds it is later evaluated on, or borrow the
held-out experiment -- exactly the leakage `docs/evaluation_protocol.md` Section 1 warns
about for hyperparameter selection. Fixing the model instead makes the comparison a clean
one-variable experiment: the only thing that differs between arms is the imbalance
treatment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Fixed for every strategy -- see the module docstring on why nothing here is tuned.
# `max_iter` is raised from the default 100 only to let the solver converge (the default
# emits a ConvergenceWarning on this data); that is a numerical-convergence setting, not a
# performance-tuned hyperparameter. `random_state` is fixed so the comparison is
# reproducible run to run.
BASELINE_MODEL_PARAMS = {"max_iter": 2000, "random_state": 0}
RESAMPLING_RANDOM_STATE = 0


def make_baseline_model(class_weight: str | dict | None = None) -> Pipeline:
    """A deliberately simple baseline: standardised features + logistic regression.

    This is the *instrument* for comparing imbalance strategies, not a candidate for the
    final M3 model (Issue #21's scope note). Logistic regression was chosen for three
    reasons: it responds directly and predictably to both class weights and resampling
    (so the comparison measures the treatment rather than a model's idiosyncrasies), it
    is fast enough to run all folds x all strategies in seconds, and it produces
    calibrated-enough probabilities for the prior-correction strategy below to be
    meaningful.

    The `StandardScaler` is inside the `Pipeline` on purpose: that is what makes it fit
    on each fold's training rows only and merely transform the held-out rows, which
    `docs/evaluation_protocol.md` Section 1 requires of any fitted preprocessing. Scaling
    outside the pipeline, on the pooled dataset, would leak the held-out experiment's own
    distribution into its evaluation.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight=class_weight, **BASELINE_MODEL_PARAMS)),
        ]
    )


def random_oversample(
    X: np.ndarray, y: np.ndarray, random_state: int = RESAMPLING_RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray]:
    """Duplicate minority-class rows (with replacement) up to the majority count.

    Plain random oversampling rather than SMOTE, which Issue #21 named as the other
    option. Reasoning, recorded in `docs/class_imbalance_decision.md`: SMOTE would mean a
    new dependency (`imbalanced-learn`) whose only use here is one function, and it
    synthesises new points by interpolating between minority neighbours -- with 17
    `Critical` rows in `1st_test`, those neighbours are sparse and scattered along a
    degradation trajectory, so interpolating between them invents vibration states that
    lie between distinct points in time rather than filling in a dense cluster. Plain
    duplication makes no such claim about the space between observed points.
    """
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()

    indices = []
    for cls, count in zip(classes, counts):
        cls_idx = np.flatnonzero(y == cls)
        indices.append(cls_idx)
        if count < target:
            indices.append(rng.choice(cls_idx, size=target - count, replace=True))

    picked = np.sort(np.concatenate(indices))
    return X[picked], y[picked]


def random_undersample(
    X: np.ndarray, y: np.ndarray, random_state: int = RESAMPLING_RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample majority-class rows (without replacement) down to the minority count.

    Included as the resampling counterpart to oversampling because the two fail in
    opposite directions, and on this dataset the failure mode is worth measuring rather
    than assuming: the pooled minority count is 107 `Critical` rows, so undersampling
    discards roughly 97% of the training data. That is a large enough loss that it might
    plausibly hurt more than the imbalance it corrects.
    """
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.min()

    indices = []
    for cls, count in zip(classes, counts):
        cls_idx = np.flatnonzero(y == cls)
        if count > target:
            cls_idx = rng.choice(cls_idx, size=target, replace=False)
        indices.append(cls_idx)

    picked = np.sort(np.concatenate(indices))
    return X[picked], y[picked]


def prior_correct(probabilities: np.ndarray, classes: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Cost-sensitive decision rule: divide predicted probabilities by training priors.

    This is Issue #21's third proposed direction (threshold-moving / cost-sensitive
    evaluation) in the form that generalises cleanly to three classes. Ordinary
    threshold-moving is a binary technique -- "predict positive if p > t" -- and a
    multiclass version needs a rule for what happens when several classes clear their
    thresholds. Dividing each class's probability by its training prior before taking the
    argmax is the standard multiclass equivalent: it is the Bayes-optimal decision rule
    under a uniform class prior, i.e. it asks "which class is this most surprising for,
    relative to how often it occurs" rather than "which class is most probable".

    Applied strictly post-hoc, to a model trained without any imbalance handling, so it is
    genuinely a distinct arm rather than a re-parameterisation of class weighting. Priors
    come from the training fold only -- using the held-out experiment's class frequencies
    would be leakage of exactly the kind `docs/evaluation_protocol.md` Section 1 rules out.
    """
    train_classes, train_counts = np.unique(y_train, return_counts=True)
    prior_by_class = dict(zip(train_classes, train_counts / train_counts.sum()))
    priors = np.array([prior_by_class.get(c, 1.0) for c in classes])
    return probabilities / priors


@dataclass(frozen=True)
class Strategy:
    """One imbalance-handling arm of the comparison."""

    name: str
    description: str
    build_model: Callable[[], Pipeline]
    resample: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None
    apply_prior_correction: bool = False


STRATEGIES: list[Strategy] = [
    Strategy(
        name="none",
        description="Control: no imbalance handling at all",
        build_model=lambda: make_baseline_model(class_weight=None),
    ),
    Strategy(
        name="class_weight_balanced",
        description="sklearn class_weight='balanced' (weights inversely proportional to class frequency)",
        build_model=lambda: make_baseline_model(class_weight="balanced"),
    ),
    Strategy(
        name="random_oversample",
        description="Duplicate minority rows up to the majority count (training fold only)",
        build_model=lambda: make_baseline_model(class_weight=None),
        resample=random_oversample,
    ),
    Strategy(
        name="random_undersample",
        description="Subsample majority rows down to the minority count (training fold only)",
        build_model=lambda: make_baseline_model(class_weight=None),
        resample=random_undersample,
    ),
    Strategy(
        name="prior_correction",
        description="Untreated model, probabilities divided by training-fold class priors before argmax",
        build_model=lambda: make_baseline_model(class_weight=None),
        apply_prior_correction=True,
    ),
]
