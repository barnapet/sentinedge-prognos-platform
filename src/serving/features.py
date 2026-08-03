"""Online (one-file-at-a-time) feature computation for serving (Issue #82).

The serving counterpart of `src/features/extraction.py`. Same five features, same
functions, same windows -- computed for a single incoming signal against a bearing's
`BearingState` instead of for a whole experiment against a `DataFrame`. As
`docs/serving_design.md` Section 2 puts it, this "is not new math."

What that means concretely, and what this module is careful about:

- The three stateless features are not reimplemented. `compute_rms`, `compute_kurtosis`
  and `compute_skewness` are imported from `src.features.extraction` and called on the
  raw signal, so `rms`/`kurtosis`/`skewness` are bit-for-bit what the batch pipeline
  produces for the same window -- including the conventions that are easy to get wrong
  a second time (Pearson kurtosis, `fisher=False`; scipy's biased skewness).
- The two stateful features are the same rolling means, maintained incrementally by
  `src.serving.state` rather than by `pandas`. `docs/feature_windowing_decision.md`
  Section 3's `min_periods=1` behaviour (files 0-8 average a shrinking window rather than
  returning `NaN`) falls out of averaging a not-yet-full deque; no row is ever `NaN`.
- The one intended difference is the cold-start baseline. For the first 49 files
  `rms_ratio` is divided by an *expanding* baseline instead of the eventual fixed 50-file
  mean, because the batch pipeline's baseline looks ahead and serving cannot
  (`docs/serving_design.md` Section 3). Every response says which regime produced it via
  `baseline_status`, and from the 50th file onward the two agree to within 2 ULP --
  `tests/test_serving_features.py` proves this by replaying a whole experiment.

The output column order is `src.training.evaluation.FEATURE_MATRIX_COLUMNS`, imported
rather than restated: `models/serving_model.joblib` (Issue #80) is a fitted `Pipeline`
whose `StandardScaler` holds per-column means and scales, so a feature vector assembled in
a different order would be silently mis-scaled rather than rejected.

No API framework is imported here, and none should be: this module is pure computation and
state, testable and reusable on its own (Issue #82's scope).

Issue #90 adds one more step here: once `skewness_smoothed` is available (it needs
`state.observe()` to have already run), the four `docs/monitoring_design.md` Section 2
monitored values are checked against the drift baseline via `state.observe_drift`, and the
resulting `drift_status` rides along on `OnlineFeatures` next to `baseline_status` -- the
same "bundle what state already computed" convention, not new computation of its own.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.features.extraction import compute_kurtosis, compute_rms, compute_skewness
from src.serving.state import BearingState, BearingStateStore
from src.training.evaluation import FEATURE_MATRIX_COLUMNS

# The model input columns, in the order `models/serving_model.joblib` was fitted on.
SERVING_FEATURE_COLUMNS = list(FEATURE_MATRIX_COLUMNS)


@dataclass(frozen=True)
class OnlineFeatures:
    """One file's five model features, plus the cold-start and drift disclosures that go
    with them."""

    rms: float
    rms_ratio: float
    kurtosis: float
    skewness: float
    skewness_smoothed: float
    baseline_status: str
    file_count: int
    drift_status: str

    def feature_vector(self) -> list[float]:
        """The five features in `SERVING_FEATURE_COLUMNS` order, ready for `predict`."""
        return [getattr(self, column) for column in SERVING_FEATURE_COLUMNS]

    def as_dict(self) -> dict[str, float | str | int]:
        """Flat mapping, for logging or an eventual response body."""
        return {
            **{column: getattr(self, column) for column in SERVING_FEATURE_COLUMNS},
            "baseline_status": self.baseline_status,
            "file_count": self.file_count,
            "drift_status": self.drift_status,
        }


def compute_online_features(signal: np.ndarray, state: BearingState) -> OnlineFeatures:
    """Compute one incoming file's features, advancing `state` by that file.

    **This mutates `state`** -- it is the online update, not a pure read, and calling it
    twice with the same signal advances the bearing twice. The stateless features are
    computed first and then handed to `state.observe`, so the current file is inside its
    own rolling window before the rolling means are read, matching the batch pipeline
    where row `i`'s window includes row `i`.

    Args:
        signal: One channel's raw vibration samples for one snapshot -- the payload
            `docs/serving_design.md` Section 1 has the client send. Channel selection
            happens client-side and is not this module's concern.
        state: The bearing's rolling state, from `BearingStateStore.get_or_create`.
    """
    rms = compute_rms(signal)
    skewness = compute_skewness(signal)
    kurtosis = compute_kurtosis(signal)

    state.observe(rms=rms, skewness=skewness)
    skewness_smoothed = state.rolling_skewness

    # docs/monitoring_design.md Section 2's four monitored values -- `rms_ratio` is
    # deliberately not one of them (that section's exclusion), so it is never passed here
    # and can never reach `state.drift_status`.
    state.observe_drift(
        {
            "rms": rms,
            "kurtosis": kurtosis,
            "skewness": skewness,
            "skewness_smoothed": skewness_smoothed,
        }
    )

    return OnlineFeatures(
        rms=rms,
        rms_ratio=state.rolling_rms / state.effective_baseline_rms,
        kurtosis=kurtosis,
        skewness=skewness,
        skewness_smoothed=skewness_smoothed,
        baseline_status=state.baseline_status,
        file_count=state.file_count,
        drift_status=state.drift_status,
    )


class OnlineFeatureExtractor:
    """`(bearing_id, signal) -> OnlineFeatures`, over a `BearingStateStore`.

    The seam the API layer is expected to hold: it turns the two halves of this issue --
    the state container and the per-file computation -- into the single call a request
    handler makes, without the handler having to know that rolling state exists. Carries
    the store's concurrency constraints unchanged (`src.serving.state`'s module docstring:
    single process, one request at a time per bearing).
    """

    def __init__(self, store: BearingStateStore | None = None) -> None:
        self.store = store if store is not None else BearingStateStore()

    def observe(self, bearing_id: str, signal: np.ndarray) -> OnlineFeatures:
        """Score-ready features for one file of `bearing_id`, advancing its history."""
        return compute_online_features(signal, self.store.get_or_create(bearing_id))

    def record_prediction(self, bearing_id: str, label: str) -> None:
        """Tally the served label against this bearing's state (Issue #90,
        `docs/monitoring_design.md` Section 3). Separate from `observe` because the label is
        only known after the model has run on that call's feature vector -- the API layer
        calls this once it has one.
        """
        self.store.get_or_create(bearing_id).record_prediction(label)
