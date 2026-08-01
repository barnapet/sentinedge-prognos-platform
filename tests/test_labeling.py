import math

import pandas as pd
import pytest

from src.labeling import ONSET_MULTIPLE, assign_labels, derive_critical_multiple


def make_df(rms_ratio, rms):
    return pd.DataFrame({"rms_ratio": rms_ratio, "rms": rms})


def test_all_normal_no_override():
    """Baseline sanity: ratios well below onset stay Normal, no override anywhere."""
    df = make_df(rms_ratio=[1.0, 1.0, 1.0], rms=[0.1, 0.1, 0.1])

    result = assign_labels(df, critical_multiple=2.0)

    assert result["label"].tolist() == ["Normal", "Normal", "Normal"]
    assert not result["override_applied"].any()


def test_threshold_boundaries_are_strict_greater_than():
    """Boundary values: `>` not `>=`, so a ratio exactly at a threshold does NOT
    cross into the next label."""
    df = make_df(
        rms_ratio=[1.3, 1.30001, 2.0, 2.00001],
        rms=[0.1, 0.1, 0.1, 0.1],
    )

    result = assign_labels(df, critical_multiple=2.0, onset_multiple=1.3)

    assert result["label"].tolist() == ["Normal", "Degrading", "Degrading", "Critical"]
    assert not result["override_applied"].any()


def test_sustained_critical_run_no_override():
    """A run that stays Critical throughout, with no raw-RMS collapse -- the
    1st_test-like case where the rig never shuts down mid-run."""
    df = make_df(rms_ratio=[3.0, 3.0, 3.0], rms=[1.0, 1.0, 1.0])

    result = assign_labels(df, critical_multiple=2.0, collapse_fraction=0.2)

    assert result["label"].tolist() == ["Critical", "Critical", "Critical"]
    assert not result["override_applied"].any()


def test_override_fires_but_does_not_change_label():
    """Override can fire on a row the threshold already called Critical -- the
    flag distinguishes 'fired' from 'changed the label', per label_pre_override."""
    df = make_df(
        rms_ratio=[3.0, 3.0],   # both rows are Critical by threshold alone
        rms=[1.0, 0.1],         # second row collapses: 0.1 < 0.2 * mean([1.0])
    )

    result = assign_labels(df, critical_multiple=2.0, collapse_fraction=0.2)

    assert result["override_applied"].tolist() == [False, True]
    assert result["label"].tolist() == ["Critical", "Critical"]
    # the defining check: override fired, but didn't change anything
    assert (result["label"] == result["label_pre_override"]).all()


def test_override_changes_label():
    """The actual purpose of the override: a raw-RMS collapse after a Critical run
    forces Critical even though the threshold rule alone would say otherwise.

    label_pre_override is "Degrading", not "Normal", here: hysteresis (Issue #20)
    relaxes downward moves one rank at a time, so a single-file plunge from a
    Critical-level ratio doesn't fall straight through to Normal -- it's exactly
    this kind of drastic single-file drop that the rig-shutdown override (not
    hysteresis) is meant to catch, and the override's own behavior is unaffected."""
    df = make_df(
        rms_ratio=[3.0, 0.5],   # second row would be Degrading by threshold+hysteresis alone
        rms=[1.0, 0.05],        # collapses: 0.05 < 0.2 * mean([1.0])
    )

    result = assign_labels(df, critical_multiple=2.0, collapse_fraction=0.2)

    assert result["label_pre_override"].tolist() == ["Critical", "Degrading"]
    assert result["label"].tolist() == ["Critical", "Critical"]
    assert result["override_applied"].tolist() == [False, True]


def test_override_reference_level_uses_critical_run_mean():
    """The collapse reference is the mean raw RMS of the Critical run *so far*,
    not just the immediately preceding value."""
    df = make_df(
        rms_ratio=[3.0, 3.0, 0.5],
        # critical_run mean after two Critical rows = mean([1.0, 3.0]) = 2.0
        # 0.2 * 2.0 = 0.4, so raw=0.3 should collapse
        rms=[1.0, 3.0, 0.3],
    )

    result = assign_labels(df, critical_multiple=2.0, collapse_fraction=0.2)

    assert result["label"].tolist() == ["Critical", "Critical", "Critical"]
    assert result["override_applied"].tolist() == [False, False, True]


# --- Issue #20: onset-boundary hysteresis -------------------------------------

def test_hysteresis_suppresses_1st_test_style_flapping():
    """Regression test for Issue #20: a synthetic ratio series shaped like
    1st_test's real files 1906-1999 (rms_ratio hovers within a few percent of
    ONSET_MULTIPLE=1.3, briefly crosses above, dips ~0.04 below, then climbs back
    and stays above for good). Without hysteresis this reverts to Normal and back;
    with the default hysteresis_margin=0.05 it must transition once, cleanly."""
    ratio = [
        1.290, 1.296,                   # approaching onset, still Normal
        1.301, 1.304, 1.307, 1.305,     # crosses onset -- Degrading (immediate)
        1.299, 1.292,                   # dips just under 1.3 (would revert with no hysteresis)
        1.279, 1.264,                   # deepest dip: 0.036 below onset, matches EDA finding
        1.270, 1.281, 1.290, 1.296,     # climbs back, still below 1.3
        1.300, 1.306, 1.308, 1.309,     # crosses back above and stays -- real onset
    ]
    df = make_df(rms_ratio=ratio, rms=[0.1] * len(ratio))

    no_hysteresis = assign_labels(df, critical_multiple=3.0, hysteresis_margin=0.0)
    with_hysteresis = assign_labels(df, critical_multiple=3.0)  # default margin=0.05

    # Confirms the bug being fixed: with no margin, this shape really does flap.
    assert "Normal" in no_hysteresis["label"].tolist()[2:], (
        "fixture no longer reproduces flapping with margin=0.0 -- fixture is stale"
    )
    no_hysteresis_labels = no_hysteresis["label"].tolist()
    reverts = sum(
        1 for i in range(1, len(no_hysteresis_labels))
        if no_hysteresis_labels[i] == "Normal" and no_hysteresis_labels[i - 1] != "Normal"
    )
    assert reverts >= 1

    # The fix: once Degrading is confirmed, it never reverts to Normal mid-series.
    labels = with_hysteresis["label"].tolist()
    assert "Degrading" in labels
    first_degrading = labels.index("Degrading")
    assert "Normal" not in labels[first_degrading:]


def test_hysteresis_does_not_delay_a_clean_transition():
    """Hysteresis must not introduce lag on a boundary that never flaps (e.g.
    3rd_test's clean single crossing) -- only reverts are gated, not the initial
    upward move."""
    ratio = [1.0, 1.0, 1.0, 1.35, 1.4, 1.45]
    df = make_df(rms_ratio=ratio, rms=[0.1] * len(ratio))

    result = assign_labels(df, critical_multiple=3.0)

    assert result["label"].tolist() == [
        "Normal", "Normal", "Normal", "Degrading", "Degrading", "Degrading",
    ]


def test_hysteresis_requires_dropping_past_margin_not_just_below_threshold():
    """A dip that crosses back below onset_multiple but not past the hysteresis
    margin must NOT revert; a dip that clears the margin must revert."""
    # Dips to 1.26: onset_multiple(1.3) - default margin(0.05) = 1.25, so 1.26
    # is back below 1.3 but doesn't clear the exit threshold -- stays Degrading.
    shallow_dip = make_df(rms_ratio=[1.35, 1.26, 1.35], rms=[0.1, 0.1, 0.1])
    result = assign_labels(shallow_dip, critical_multiple=3.0)
    assert result["label"].tolist() == ["Degrading", "Degrading", "Degrading"]

    # Dips to 1.20, clearing 1.25 -- reverts to Normal.
    deep_dip = make_df(rms_ratio=[1.35, 1.20, 1.35], rms=[0.1, 0.1, 0.1])
    result = assign_labels(deep_dip, critical_multiple=3.0)
    assert result["label"].tolist() == ["Degrading", "Normal", "Degrading"]


def test_hysteresis_does_not_affect_3rd_test_style_clean_run():
    """3rd_test never flapped in the original threshold rule (EDA confirmed its
    Critical/Degrading runs are single contiguous blocks) -- hysteresis must be a
    no-op on a sequence with no near-boundary noise."""
    ratio = [1.0] * 5 + [1.35] * 5 + [3.5] * 5
    df = make_df(rms_ratio=ratio, rms=[0.1] * len(ratio))

    with_hysteresis = assign_labels(df, critical_multiple=3.0)
    no_hysteresis = assign_labels(df, critical_multiple=3.0, hysteresis_margin=0.0)

    assert with_hysteresis["label"].tolist() == no_hysteresis["label"].tolist()
    assert with_hysteresis["label"].tolist() == (
        ["Normal"] * 5 + ["Degrading"] * 5 + ["Critical"] * 5
    )


# --- derive_critical_multiple (Issue #65) --------------------------------------------
#
# The peak rolling RMS ratios below are the real measured values for the three experiments,
# inlined rather than loaded from `data/processed/`. Two reasons: that directory is
# gitignored, and in CI the `pytest` step runs *before* the notebook-execution step that
# would populate it -- a data-loading test would be skipped exactly where it matters. They
# are reproducible from the raw dataset as
# `rms.rolling(10, min_periods=1).mean() / rms.head(50).mean()`, maximised
# (`docs/feature_windowing_decision.md`), and their 2-decimal forms are the
# `peak_ratio_rolling` column of notebooks/02_health_state_labeling.ipynb Section 1.
MEASURED_PEAK_RATIOS = {
    "1st_test": 2.8689768817960495,
    "2nd_test": 6.323153276832926,
    "3rd_test": 7.152643233555911,
}

# docs/eda_findings.md Section 3 / notebooks/04_feature_pipeline_validation.ipynb.
DOCUMENTED_CRITICAL_MULTIPLES = {
    "1st_test": 1.932,
    "2nd_test": 2.866,
    "3rd_test": 3.049,
}


@pytest.mark.parametrize("experiment", sorted(MEASURED_PEAK_RATIOS))
def test_reproduces_documented_critical_multiples_exactly(experiment):
    """The headline requirement of Issue #65: exact equality with the three values
    already published in docs/eda_findings.md Section 3 and hardcoded in notebook 04 --
    not approximate agreement. `==` is deliberate over `pytest.approx` here."""
    result = derive_critical_multiple(MEASURED_PEAK_RATIOS[experiment])

    assert result == DOCUMENTED_CRITICAL_MULTIPLES[experiment]


@pytest.mark.parametrize("experiment", sorted(MEASURED_PEAK_RATIOS))
def test_rounding_chain_is_not_incidental(experiment):
    """Guard against someone "simplifying" the 2-decimal input rounding away.

    Without it the formula returns values that differ in the 3rd decimal for two of the
    three experiments, silently disagreeing with the published figures. This test fails if
    that rounding is dropped, and documents that the difference is real rather than a
    floating-point artifact."""
    peak = MEASURED_PEAK_RATIOS[experiment]
    unrounded_input = round(ONSET_MULTIPLE * (peak / ONSET_MULTIPLE) ** 0.5, 3)

    expected_gap = {"1st_test": 0.001, "2nd_test": 0.001, "3rd_test": 0.0}[experiment]
    assert (
        abs(derive_critical_multiple(peak) - unrounded_input) == pytest.approx(expected_gap)
    )


@pytest.mark.parametrize("experiment", sorted(MEASURED_PEAK_RATIOS))
def test_pre_rounded_input_is_equivalent(experiment):
    """Callers may pass the raw peak or the already-2dp-rounded `peak_ratio_rolling` the
    notebook tabulates -- the rounding is idempotent, so both give the same answer."""
    peak = MEASURED_PEAK_RATIOS[experiment]

    assert derive_critical_multiple(peak) == derive_critical_multiple(round(peak, 2))


def test_default_is_the_geometric_midpoint():
    """f = 0.5 must be sqrt(onset * peak) -- the property the derivation rests on."""
    peak = 5.2

    assert derive_critical_multiple(peak) == round(math.sqrt(ONSET_MULTIPLE * peak), 3)


def test_span_endpoints():
    """f = 0 puts the boundary at onset, f = 1 puts it at the peak -- the span the
    fraction interpolates across."""
    peak = 6.32

    assert derive_critical_multiple(peak, span_fraction=0.0) == ONSET_MULTIPLE
    assert derive_critical_multiple(peak, span_fraction=1.0) == peak


def test_monotonic_in_peak_ratio_and_span_fraction():
    """A worse peak, or a larger fraction, must push the Critical boundary up -- checked
    across the f = 0.40-0.65 range the notebook's sensitivity sweep covers."""
    peaks = [2.87, 4.0, 6.32, 7.15]
    assert all(
        derive_critical_multiple(a) < derive_critical_multiple(b)
        for a, b in zip(peaks, peaks[1:])
    )

    fractions = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    multiples = [derive_critical_multiple(6.32, span_fraction=f) for f in fractions]
    assert all(a < b for a, b in zip(multiples, multiples[1:]))


def test_output_sits_between_onset_and_peak():
    """The derived boundary must land strictly inside the Degrading band it splits --
    otherwise `assign_labels` would produce an empty Degrading or Critical class."""
    for peak in MEASURED_PEAK_RATIOS.values():
        result = derive_critical_multiple(peak)

        assert ONSET_MULTIPLE < result < peak


def test_feeds_assign_labels_and_populates_all_three_states():
    """The intended M3 usage -- derive the threshold, pass it straight to `assign_labels`
    -- on a ratio series shaped like a real run-to-failure experiment."""
    ratio = [1.0] * 10 + [1.5] * 10 + [2.5] * 5 + [2.87] * 5
    df = make_df(rms_ratio=ratio, rms=[0.1] * len(ratio))

    critical_multiple = derive_critical_multiple(max(ratio))
    result = assign_labels(df, critical_multiple)

    assert critical_multiple == 1.932
    assert set(result["label"].unique()) == {"Normal", "Degrading", "Critical"}
