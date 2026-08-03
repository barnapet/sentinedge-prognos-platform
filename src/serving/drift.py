"""Per-feature drift check against a pooled training baseline (Issue #90,
`docs/monitoring_design.md` Sections 1-3).

`docs/monitoring_design.md` Section 1 decided a per-feature z-score against a fixed,
precomputed `(mean, std)` pair, `|z| > 3` (a distribution-agnostic Chebyshev bound, not an
assumption of normality none of the four monitored features actually have), with a
3-of-last-10-requests persistence rule so a single noisy reading cannot flip the flag alone.
This module holds the pure, stateless half of that decision -- `compute_z_score`/
`is_extreme`, and the constants both the offline baseline-computation script and the
per-bearing state module need to agree on. The persistence rule itself lives on
`BearingState` (`src/serving/state.py`), because it needs per-bearing rolling history, not
just one reading.

`MONITORED_FEATURES`/`BASELINE_PATH`/`load_drift_baseline` are imported from
`src.training.compute_drift_baseline`, not re-declared -- the same anti-drift-duplication
convention `src/serving/features.py` already follows for `FEATURE_MATRIX_COLUMNS`
(imported from `src.training.evaluation`, not restated).
"""
from __future__ import annotations

from src.training.compute_drift_baseline import (
    BASELINE_PATH,
    MONITORED_FEATURES,
    load_drift_baseline,
)

# docs/monitoring_design.md Section 1: Chebyshev's inequality bounds P(|Z| >= 3) <= 1/9 =~
# 11.1% regardless of the underlying distribution's shape -- a conservative threshold, not
# an assumption of normality (none of the four monitored features have it: baseline
# kurtosis sits around 3.4-3.5, baseline |skewness| around 0.03).
Z_SCORE_THRESHOLD = 3.0

# docs/monitoring_design.md Section 3: "at least 3 of its last 10 requests". Reuses
# src.features.extraction.ROLLING_WINDOW (=10) as the rolling window size (see
# src.serving.state), so a persistent shift trips within 3 requests of onset while a
# single rare excursion -- expected occasionally by chance, per the Chebyshev bound above
# -- does not flip the flag alone.
PERSISTENCE_MIN_COUNT = 3

DRIFT_NOMINAL = "nominal"
DRIFT_DRIFTING = "drifting"


def compute_z_score(value: float, mean: float, std: float) -> float:
    """How many standard deviations `value` sits from the training baseline.

    Guarded against `std == 0.0` (only possible if a feature were constant across every
    training row, which none of `MONITORED_FEATURES` are) by returning `0.0` rather than
    raising -- a monitoring check should degrade to "no signal" rather than crash a request.
    """
    if std == 0.0:
        return 0.0
    return (value - mean) / std


def is_extreme(z: float, threshold: float = Z_SCORE_THRESHOLD) -> bool:
    """Whether one reading's z-score alone crosses the (conservative, Chebyshev) bound.

    This is the single-reading check, not the persistent "drifting" status -- see
    `BearingState.feature_drift` for the 3-of-10 rule built on top of this.
    """
    return abs(z) > threshold


# Loaded once at import time: `models/drift_baseline.json` is a small, committed file (same
# treatment as `models/serving_model_manifest.json`, `docs/serving_model_artifact.md`), not
# something serving regenerates per request or per process. Overridable per
# `BearingState`/`BearingStateStore` instance (constructor parameter) so tests can inject a
# synthetic baseline without touching the committed file.
DRIFT_BASELINE: dict[str, tuple[float, float]] = load_drift_baseline()
