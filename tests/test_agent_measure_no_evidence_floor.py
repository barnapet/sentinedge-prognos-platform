"""Tier-1 tests for the no-evidence separability measurement (Issue #177).

No model, no network, no Qdrant -- `docs/agent_design.md` Section 8 tier 1's hard requirement.
Every reading below is built by hand, so the scoring logic is exercised without an index.
`read()` is the one function not covered here beyond a shape check: it is the thin layer that
calls production's `assess_retrieval`, and `collect_scores` needs the real collection by
definition.

What is worth pinning is not the arithmetic but the three ways a measurement of this shape can
quietly lie about itself: a condition counted as "clean" while it refuses a sourceable
question, a margin whose sign is not read (the rejected `< MIN_SUPPORTING_CHUNKS` variant
spares its nearest answerable item while refusing one that scores *above* it), and a verdict
that recommends something when the honest reading is a null result.
"""
from __future__ import annotations

from src.agent.critic.retrieval_confidence import MIN_SUPPORTING_CHUNKS
from tests.fixtures.golden_set_runner import MUST_REFUSE
from tests.fixtures.measure_no_evidence_floor import (
    CANDIDATE_CONDITIONS,
    Reading,
    evaluate,
    format_verdict,
)

ZERO, FEWER = (condition for condition, _ in CANDIDATE_CONDITIONS)


def _reading(
    item_id: str, *, top: float, supporting: int, passed: bool = False, refuse: bool = False
) -> Reading:
    return Reading(
        item_id=item_id, is_must_refuse=refuse, top=top, supporting=supporting, passed=passed
    )


def test_the_two_candidate_conditions_are_the_ones_expressible_from_existing_constants():
    """A third entry would mean a number this project has not calibrated had been introduced
    quietly, which is the failure #163's procedure exists to prevent."""
    assert [condition for condition, _ in CANDIDATE_CONDITIONS] == [
        "supporting_count == 0",
        f"supporting_count < {MIN_SUPPORTING_CHUNKS}",
    ]


def test_a_condition_never_fires_on_a_turn_whose_retrieval_passed():
    """`fires` is asked only about the below-threshold regime. A passing turn with a
    degenerate count would otherwise be reported as refusable, which is a category error --
    the whole point is that this is a sub-case of `below_threshold`."""
    passing = _reading("answerable", top=0.80, supporting=0, passed=True)

    assert passing.fires(ZERO) is False
    assert passing.fires(FEWER) is False


def test_a_condition_that_refuses_an_answerable_item_is_not_clean():
    """The number that decides a candidate. Catching every must-refuse item is worth nothing
    if it also refuses a question the corpus genuinely covers."""
    readings = (
        _reading("refuse", top=0.66, supporting=0, refuse=True),
        _reading("answerable", top=0.69, supporting=0),
    )

    result = evaluate(readings, ZERO, "not one chunk reached TAU_SUPPORT")

    assert result.caught == 1
    assert result.false_refusals == 1
    assert result.clean is False


def test_the_margin_is_negative_when_a_condition_refuses_above_what_it_spares():
    """The measured shape of the rejected variant: it fires on a must-refuse item at 0.7131
    while sparing an answerable one at 0.7015, so it is not separating the classes by score
    at all -- it spares that item only because the item holds exactly MIN_SUPPORTING_CHUNKS
    chunks. A reader who saw only "6/8 caught, 0 false refusals" would take it as the better
    condition."""
    readings = (
        _reading("refuse-strong", top=0.7131, supporting=1, refuse=True),
        _reading("answerable-borderline", top=0.7015, supporting=MIN_SUPPORTING_CHUNKS),
    )

    result = evaluate(readings, FEWER, "fewer than MIN_SUPPORTING_CHUNKS reached TAU_SUPPORT")

    assert result.caught == 1 and result.false_refusals == 0
    assert result.clean is True, "clean on counts alone"
    assert result.margin is not None and result.margin < 0, "and wrong on the score axis"


def test_the_verdict_recommends_the_larger_margin_not_the_larger_catch():
    """#163's own max-margin tie-break, applied here: a condition catching one more item is
    not preferred over one that sits further from the nearest question it must not refuse."""
    readings = (
        _reading("refuse-weak", top=0.6645, supporting=0, refuse=True),
        _reading("refuse-strong", top=0.7131, supporting=1, refuse=True),
        _reading("answerable-borderline", top=0.7015, supporting=MIN_SUPPORTING_CHUNKS),
    )
    results = [
        evaluate(readings, condition, description)
        for condition, description in CANDIDATE_CONDITIONS
    ]

    verdict = "\n".join(format_verdict(results))

    assert f"VERDICT: `{ZERO}` is the condition to implement." in verdict
    assert "Rejected:" in verdict and FEWER in verdict
    assert "scoring ABOVE the answerable one it spares" in verdict


def test_a_null_result_is_reported_as_a_null_result():
    """Issue #173/PR #174's precedent, and the outcome this script had to stay capable of.
    When every candidate refuses something sourceable, the honest output recommends nothing
    rather than the least-bad cutoff."""
    readings = (
        _reading("refuse", top=0.66, supporting=0, refuse=True),
        _reading("answerable", top=0.65, supporting=0),
    )
    results = [
        evaluate(readings, condition, description)
        for condition, description in CANDIDATE_CONDITIONS
    ]

    verdict = "\n".join(format_verdict(results))

    assert "VERDICT: no separation. Recommend implementing nothing." in verdict


def test_the_must_refuse_category_name_is_the_golden_sets_own():
    """`is_must_refuse` reaches these readings from `ItemScores`, which derives it from the
    category string. A drifted copy of that string here would silently classify every item as
    answerable and report a clean separation on an empty must-refuse class."""
    assert MUST_REFUSE == 'Must refuse / "I don\'t know"'
