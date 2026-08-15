"""Tier-1 tests for the TAU_TOP/TAU_SUPPORT calibration sweep (Issue #163).

No model, no network, no API key -- Section 8 tier 1's hard requirement. Every test below
builds `ItemScores` by hand, so the sweep's decision logic is exercised without a Qdrant and
without the embedding model. `collect_scores` is the one function not covered here: it is the
thin layer that calls `search()`, and it needs the real index by definition.

**The case that matters most here is the one the real corpus cannot produce.** Against the
committed corpus a feasible pair exists, so `recommend`'s "no pair satisfies the hard
constraint" branch never executes in a real run -- and Issue #163 names that branch's
behaviour explicitly ("state that plainly rather than picking a best-effort compromise"). A
branch that only runs when the corpus degrades is exactly the branch that must be pinned by a
test rather than trusted.
"""
from __future__ import annotations

from tests.fixtures import calibrate_retrieval
from tests.fixtures.calibrate_retrieval import (
    TAU_SUPPORT_CANDIDATES,
    TAU_TOP_CANDIDATES,
    ItemScores,
    evaluate,
    format_recommendation,
    format_separability,
    format_sweep,
    recommend,
    sweep,
)
from tests.fixtures.golden_set_runner import MUST_REFUSE

ANSWERABLE = "Answerable from the docs"


def _answerable(item_id: str, *scores: float) -> ItemScores:
    return ItemScores(item_id=item_id, category=ANSWERABLE, scores=tuple(scores))


def _refusing(item_id: str, *scores: float) -> ItemScores:
    return ItemScores(item_id=item_id, category=MUST_REFUSE, scores=tuple(scores))


# --- The hard constraint --------------------------------------------------------------


def test_one_leaking_must_refuse_item_rejects_the_pair():
    """Zero tolerance, mirroring gate 1's own 100%/no-aggregate rule: seven refusing items
    staying below the bar cannot buy back the eighth reaching it."""
    measured = [
        *(_refusing(f"refuse-{index}", 0.60, 0.59) for index in range(7)),
        _refusing("refuse-leak", 0.80, 0.79),
        _answerable("answerable", 0.90, 0.89),
    ]
    cell = evaluate(measured, tau_top=0.75, tau_support=0.70)

    assert cell.refuse_below == 7
    assert cell.refuse_total == 8
    assert not cell.feasible


def test_a_must_refuse_item_is_judged_on_its_top_score_alone():
    """Section 8 records for these items "the top similarity score and whether it stayed
    below TAU_TOP". A single strong hit with nothing corroborating it would *fail*
    `assess_retrieval`, but it has still reached the bar, and counting it as safe because the
    corroboration rule happened to catch it would make the constraint depend on a rule it is
    not about."""
    lone_strong_hit = _refusing("refuse", 0.80)
    assert not lone_strong_hit.passes(tau_top=0.75, tau_support=0.70)

    cell = evaluate([lone_strong_hit, _answerable("a", 0.90, 0.89)], 0.75, 0.70)
    assert cell.refuse_below == 0
    assert not cell.feasible


def test_recommend_returns_none_when_nothing_in_range_is_feasible():
    """The branch a real run never reaches. Every must-refuse item retrieves above the whole
    swept range, so no pair keeps them all below tau_top."""
    measured = [
        _refusing("refuse", 0.99, 0.98),
        _answerable("answerable", 0.99, 0.98),
    ]
    assert recommend(sweep(measured), measured) is None


def test_the_no_recommendation_report_says_so_plainly_and_offers_no_compromise():
    text = "\n".join(format_recommendation(None, []))
    assert "RECOMMENDATION: none." in text
    assert "best-effort" in text
    # The failure mode this branch exists to refuse: quietly proposing a pair anyway.
    assert "tau_top     =" not in text


# --- The answerable half runs through production's own predicate ------------------------


def test_an_answerable_item_needs_corroboration_not_just_a_strong_top_hit():
    """`assess_retrieval` is the real gate, so `min_supporting` applies here exactly as it
    does to a turn: one chunk above tau_support is not enough, whatever the top score."""
    uncorroborated = _answerable("answerable", 0.80, 0.50)
    corroborated = _answerable("answerable", 0.80, 0.72)

    assert not uncorroborated.passes(tau_top=0.75, tau_support=0.70)
    assert corroborated.passes(tau_top=0.75, tau_support=0.70)


def test_min_supporting_chunks_is_imported_and_never_swept():
    """Section 6 fixes it at 2 and Issue #163 does not calibrate it, so there is deliberately
    no candidate grid for it to be swept over."""
    assert calibrate_retrieval.MIN_SUPPORTING_CHUNKS == 2
    assert not hasattr(calibrate_retrieval, "MIN_SUPPORTING_CANDIDATES")


# --- The grid ---------------------------------------------------------------------------


def test_the_sweep_never_pairs_a_support_floor_above_the_top_floor():
    """`retrieval_confidence.py` states the invariant outright ("TAU_TOP is above
    TAU_SUPPORT by construction"); a pair inverting them is not a threshold pair."""
    measured = [_refusing("r", 0.60, 0.59), _answerable("a", 0.90, 0.89)]
    assert all(cell.tau_support <= cell.tau_top for cell in sweep(measured))


def test_the_sweep_covers_every_valid_pair_of_the_two_grids():
    measured = [_refusing("r", 0.60, 0.59), _answerable("a", 0.90, 0.89)]
    expected = sum(
        1
        for tau_top in TAU_TOP_CANDIDATES
        for tau_support in TAU_SUPPORT_CANDIDATES
        if tau_support <= tau_top
    )
    assert len(sweep(measured)) == expected


# --- Tie-breaks --------------------------------------------------------------------------


def test_the_recommended_tau_top_is_the_balanced_one_among_equal_pass_rates():
    """Refusing tops out at 0.70 and the single answerable sits at 0.80, so every tau_top in
    0.71..0.80 is feasible and admits it -- all tied at 1/1. The recommendation is the one
    centred between the two classes (0.75), not the first or last of the plateau."""
    measured = [_refusing("r", 0.70, 0.69), _answerable("a", 0.80, 0.79)]
    best = recommend(sweep(measured), measured)

    assert best is not None
    assert best.answerable_passed == 1
    assert best.tau_top == 0.75


def test_the_recommended_tau_support_is_the_strongest_that_costs_no_item():
    """The answerable item's second chunk is at 0.72, so a 0.75 corroboration floor would
    drop it while 0.70 does not. Among the pairs tied on pass rate the tighter floor that is
    still free is the one recommended."""
    measured = [_refusing("r", 0.70, 0.69), _answerable("a", 0.80, 0.72)]
    best = recommend(sweep(measured), measured)

    assert best is not None
    assert best.answerable_passed == 1
    assert best.tau_support == 0.70


# --- Reporting ----------------------------------------------------------------------------


def test_separability_names_the_answerable_items_no_threshold_can_admit():
    """The reading that says a perfect score is unavailable, and why -- an answerable item
    scoring at or below the highest must-refuse one cannot be admitted by any feasible
    tau_top."""
    measured = [
        _refusing("refuse-high", 0.73, 0.72),
        _answerable("answerable-blocked", 0.70, 0.69),
        _answerable("answerable-clear", 0.80, 0.79),
    ]
    text = "\n".join(format_separability(measured))

    assert "OVERLAP" in text
    assert "answerable-blocked" in text
    assert "answerable-clear" not in text


def test_separability_reports_a_cleanly_separable_pair_of_classes():
    measured = [_refusing("r", 0.60, 0.59), _answerable("a", 0.80, 0.79)]
    text = "\n".join(format_separability(measured))

    assert "separable" in text
    assert "OVERLAP" not in text


def test_the_sweep_table_publishes_the_rejected_rows_rather_than_only_the_winner():
    """Section 8 requires the sweep itself to be published, so a row rejected by the hard
    constraint is printed and marked, not dropped."""
    measured = [_refusing("r", 0.70, 0.69), _answerable("a", 0.80, 0.79)]
    text = "\n".join(format_sweep(sweep(measured)))

    assert "LEAK" in text
    # One row per swept tau_top, all present.
    for tau_top in TAU_TOP_CANDIDATES:
        assert f"{tau_top:>7.2f}" in text
