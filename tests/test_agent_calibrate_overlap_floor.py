"""Tier-1 tests for the LEXICAL_OVERLAP_FLOOR calibration sweep (Issue #173).

No model, no network, no API key -- Section 8 tier 1's hard requirement, and here it is also
the point: the sweep's expensive half is the collection, and everything *downstream* of it must
be checkable for free. Every test below builds a `MeasuredTurn` by hand from real golden-set
items, real chunk text and hand-written verdicts, so the decision logic is exercised without a
Qdrant, a serving process or a credential. `measure_item_async` and `collect` are the two
functions not covered here: they are the thin layer that calls the API, which is what they are
for.

**The one test that pins the whole approach** is
`test_the_swept_response_matches_what_the_pipeline_produces_at_the_current_floor`: the sweep is
only worth reading if a row of it describes what production would really do, so the row at the
current floor is asserted equal to `pipeline.verify_turn_async`'s own output on the same turn,
tier for tier and claim for claim.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from src.agent.answerer import Claim, Draft
from src.agent.critic.deterministic import TurnEvidence, verify
from src.agent.critic.grounding import UNGROUNDED
from src.agent.mcp.tools import DOCS_SOURCE_ID
from src.agent.pipeline import turn_from_payloads, verify_turn_async
from tests.fixtures import calibrate_overlap_floor as calibrate
from tests.fixtures.calibrate_overlap_floor import (
    CURRENT_FLOOR,
    FLOOR_CANDIDATES,
    Collection,
    CollectionError,
    MeasuredTurn,
    SweepCell,
    VerdictMissing,
    _VerdictReplay,
    any_prose,
    baseline_cell,
    evaluate,
    format_errors,
    format_reach,
    format_recommendation,
    format_separability,
    format_sweep,
    measurements_as_dict,
    measurements_from_dict,
    reaches,
    recommend,
    sweep,
)
from tests.fixtures.golden_set_corpus import CORPUS_ITEMS

# --- Fixtures built by hand ---------------------------------------------------------------
#
# Real golden-set items, so the scoring below is Section 8's own contract rather than a
# convenient one: `_ANSWERABLE_ITEM` declares required substrings ("1.3", "critical_multiple")
# and an allowed document, and `_REFUSE_ITEM` declares the refusal wording and the torque units
# a fabricated answer reaches for.

_ANSWERABLE_ITEM = next(
    item for item in CORPUS_ITEMS if item.item_id == "corpus-answerable-health-state-thresholds"
)
_REFUSE_ITEM = next(
    item for item in CORPUS_ITEMS if item.item_id == "corpus-refuse-housing-bolt-torque"
)

# Chunk text close enough to the real thing to make the overlaps meaningful: a claim that reuses
# its vocabulary scores high, one written in its own words scores low.
_THRESHOLD_CHUNK = (
    "The label rule is a ratio of the rolling rms to the bearing's own healthy baseline: "
    "ratio <= 1.3 is Normal, above 1.3 is Degrading, and Critical begins at "
    "critical_multiple, derived per experiment as sqrt(1.3 * peak_ratio_rolling)."
)
_FASTENER_CHUNK = (
    "All bearings are force lubricated by an on-site oil circulation system which regulates "
    "the flow and the temperature of the lubricant. The rig carries a radial load applied to "
    "the shaft by a spring mechanism."
)


def _search_payload(chunk_id: str, text: str, *, score: float = 0.80) -> dict:
    """One `search_documentation` payload, in `src/agent/mcp/results.py`'s shape.

    The chunk's `source_type` is `decision_doc` -- one of `PROSE_SOURCE_TYPES` -- because that
    is the whole precondition of the trigger being calibrated. A payload minting anything else
    would be testing the scoping decision Issue #119 already made and #173 must not touch.
    """
    return {
        "source": {"source_type": "live_endpoint", "source_id": DOCS_SOURCE_ID},
        "data": {
            "results": [
                {
                    "text": text,
                    "score": score,
                    "heading_path": "Health-state labels",
                    "source": {
                        "source_id": chunk_id,
                        "source_type": "decision_doc",
                        "source_ref": chunk_id.rpartition("::")[0],
                    },
                }
            ]
        },
    }


def _measured(item, claims, payloads, verdicts) -> MeasuredTurn:
    """A `MeasuredTurn` assembled the way the replay path assembles one."""
    turn = turn_from_payloads(
        Draft(claims=claims, recommendation=None, unanswered=[]), payloads
    )
    evidence = TurnEvidence.from_tool_payloads(turn.tool_payloads)
    return MeasuredTurn(
        item=item,
        turn=turn,
        evidence=evidence,
        report=verify(turn.draft, evidence),
        verdicts=verdicts,
    )


def _claim(text: str, *source_ids: str) -> Claim:
    return Claim(text=text, source_ids=list(source_ids))


_THRESHOLD_CHUNK_ID = "docs/eda_findings.md::4"
_FASTENER_CHUNK_ID = "ims_bearing_data_readme::0"


def _answerable_turn(verdict: str = "yes") -> MeasuredTurn:
    """An answerable item whose one claim carries both required substrings.

    Its overlap is deliberately not 1.0 -- the claim adds words of its own ("rolling", "rms",
    "baseline" are in the chunk; "state", "begins" are the claim's framing) -- so there are
    floors on the grid both below and above it.
    """
    claim = _claim(
        "Normal is a rolling rms ratio at or below 1.3 against the bearing's own baseline, and "
        "the Critical state begins at the per-experiment critical_multiple.",
        _THRESHOLD_CHUNK_ID,
    )
    return _measured(
        _ANSWERABLE_ITEM,
        [claim],
        [_search_payload(_THRESHOLD_CHUNK_ID, _THRESHOLD_CHUNK)],
        {(0, _THRESHOLD_CHUNK_ID): verdict},
    )


def _refuse_turn(verdict: str = "no") -> MeasuredTurn:
    """A must-refuse item that drafted an answer anyway, citing a real chunk.

    This is the shape the floor exists for: the citation is real, every deterministic check
    passes, and only the critic can tell that the passage does not support the claim. The claim
    names no torque figure, so the item's `forbidden_substrings` are not what decides it -- the
    verdict is decided by whether a claim survives at all.
    """
    claim = _claim(
        "The rig's bearings are force lubricated by an oil circulation system, so the housing "
        "cap screws are tightened against a spring mechanism radial load.",
        _FASTENER_CHUNK_ID,
    )
    return _measured(
        _REFUSE_ITEM,
        [claim],
        [_search_payload(_FASTENER_CHUNK_ID, _FASTENER_CHUNK, score=0.66)],
        {(0, _FASTENER_CHUNK_ID): verdict},
    )


def _overlap_of(measured: MeasuredTurn) -> float:
    pairs = measured.prose_pairs()
    assert len(pairs) == 1, "the fixtures above each carry exactly one prose pair"
    return pairs[0][2]


# --- The grid ------------------------------------------------------------------------------


def test_the_grid_is_issue_173s_range_and_step():
    assert FLOOR_CANDIDATES[0] == 0.40
    assert FLOOR_CANDIDATES[-1] == 0.95
    assert len(FLOOR_CANDIDATES) == 12
    assert all(
        round(later - earlier, 4) == 0.05
        for earlier, later in zip(FLOOR_CANDIDATES, FLOOR_CANDIDATES[1:])
    )


def test_the_grid_contains_the_current_floor_so_the_baseline_row_is_a_candidate():
    """The recommendation may be "keep it", and it can only be that if the current value is on
    the grid. `baseline_cell` measures it independently either way."""
    assert CURRENT_FLOOR in FLOOR_CANDIDATES


def test_the_sweep_touches_neither_of_the_thresholds_163_calibrated():
    """Issue #173's constraint, as a property of the module rather than a promise in its
    docstring: no `TAU_TOP`/`TAU_SUPPORT` candidate grid, and nothing swept but the floor."""
    assert not hasattr(calibrate, "TAU_TOP_CANDIDATES")
    assert not hasattr(calibrate, "TAU_SUPPORT_CANDIDATES")
    assert [cell.floor for cell in sweep([_answerable_turn()])] == list(FLOOR_CANDIDATES)


# --- The mechanism the floor actually controls ---------------------------------------------


def test_escalated_pairs_never_decrease_as_the_floor_rises():
    """The trigger is `overlap < floor`, so the escalated set is monotone in the floor. The
    sweep's cost column is read as a cost, and a cost that could fall as the threshold rises
    would mean the trigger was not what this script thinks it is."""
    measured = [_answerable_turn(), _refuse_turn()]
    counts = [cell.escalated_pairs for cell in sweep(measured)]
    assert counts == sorted(counts)
    assert counts[0] < counts[-1], "the fixtures must straddle the grid for this to be a test"


def test_a_must_refuse_item_refuses_only_once_its_claim_escalates():
    """The whole reason the must-refuse half is floor-sensitive at all.

    Below the claim's overlap nothing escalates, the claim is released, and the item fails its
    refusal sub-score. Above it the claim is escalated, the recorded `no` demotes it, no claim
    survives, and `grounding.py` produces the tier-3 refusal the item is scored on.
    """
    measured = _refuse_turn(verdict="no")
    overlap = _overlap_of(measured)
    below = max(floor for floor in FLOOR_CANDIDATES if floor <= overlap)
    above = min(floor for floor in FLOOR_CANDIDATES if floor > overlap)

    assert measured.response_at(below).grounding_tier != UNGROUNDED
    assert not measured.score_at(below).passed

    refused = measured.response_at(above)
    assert refused.grounding_tier == UNGROUNDED
    assert refused.claims == ()
    assert measured.score_at(above).passed


def test_an_answerable_claim_the_critic_rejects_is_lost_once_the_floor_reaches_it():
    """The cost side of the same lever, which is what keeps the sweep from simply recommending
    0.95: escalating a claim a correct answer needs, and getting `unclear`, drops it."""
    measured = _answerable_turn(verdict="unclear")
    overlap = _overlap_of(measured)
    below = max(floor for floor in FLOOR_CANDIDATES if floor <= overlap)
    above = min(floor for floor in FLOOR_CANDIDATES if floor > overlap)

    assert measured.score_at(below).passed
    assert not measured.score_at(above).passed


def test_a_yes_verdict_costs_a_model_call_and_changes_nothing():
    """Escalating a well-supported claim is spend, not safety -- the reading the sweep's `pairs`
    column exists to make visible."""
    measured = _answerable_turn(verdict="yes")
    assert all(measured.score_at(floor).passed for floor in FLOOR_CANDIDATES)
    assert len(measured.requests_at(0.95)) == 1
    assert len(measured.requests_at(0.40)) == 0


def test_only_prose_citing_claims_are_measured():
    """`prose_pairs` is the population the floor thresholds, and Issue #119's scoping decides
    it. A claim citing only the live-endpoint wrapper contributes nothing -- not a zero."""
    payload = {
        "source": {"source_type": "live_endpoint", "source_id": "GET /monitoring/drift"},
        "data": {"status": {"file_count": 197}},
    }
    measured = _measured(
        _ANSWERABLE_ITEM,
        [
            _claim(
                "Bearing 2nd_test-demo has been scored on 197 windows so far.",
                "GET /monitoring/drift",
            )
        ],
        [payload],
        {},
    )
    assert measured.prose_pairs() == ()
    assert not measured.cites_prose
    assert all(measured.requests_at(floor) == () for floor in FLOOR_CANDIDATES)


def test_an_unclean_deterministic_report_escalates_at_no_floor():
    """Section 6 makes a clean deterministic pass the escalation's precondition, so a draft with
    a fabricated number is out of the floor's reach entirely -- reported as such rather than
    counted as a floor result."""
    claim = _claim(
        "Critical begins at a critical_multiple of 9.87 for every experiment.",
        _THRESHOLD_CHUNK_ID,
    )
    measured = _measured(
        _ANSWERABLE_ITEM,
        [claim],
        [_search_payload(_THRESHOLD_CHUNK_ID, _THRESHOLD_CHUNK)],
        {},
    )
    assert not measured.report.clean
    assert all(measured.requests_at(floor) == () for floor in FLOOR_CANDIDATES)

    reach = reaches([measured], sweep([measured]))[0]
    assert not reach.floor_sensitive
    assert "deterministic pass is not clean" in reach.reason_out_of_reach


# --- The sweep agrees with production ------------------------------------------------------


def test_the_swept_response_matches_what_the_pipeline_produces_at_the_current_floor():
    """The assertion the whole script leans on.

    `MeasuredTurn.response_at` merges two demotion sources the way `pipeline.verify_turn_async`
    does, and a merge that drifted from it would make every row of the table describe something
    production does not do. So the current floor's row is compared against the pipeline's own
    output on the same turn, with the same recorded verdicts replayed through the same
    `escalate_async`.
    """
    measured = _refuse_turn(verdict="no")
    replay = _VerdictReplay(measured, measured.requests_at(CURRENT_FLOOR))

    from_pipeline = asyncio.run(verify_turn_async(measured.turn, critic_client=replay))
    from_sweep = measured.response_at(CURRENT_FLOOR)

    assert from_sweep.grounding_tier == from_pipeline.grounding_tier
    assert from_sweep.claims == from_pipeline.claims
    assert from_sweep.dropped == from_pipeline.dropped
    assert from_sweep.text == from_pipeline.text


def test_the_replay_client_refuses_to_invent_a_verdict():
    """A missing verdict is a pair nobody asked the critic about. Answering it either way would
    put a number nobody measured in the table -- the same refusal `parse_verdict` makes."""
    measured = _refuse_turn()
    unmeasured = MeasuredTurn(
        item=measured.item,
        turn=measured.turn,
        evidence=measured.evidence,
        report=measured.report,
        verdicts={},
    )
    with pytest.raises(VerdictMissing):
        unmeasured.response_at(0.95)


def test_the_union_of_escalated_pairs_is_collected_once_for_the_whole_grid():
    """What keeps the collection cheap: a pair's verdict does not depend on the floor, so the
    pairs escalated anywhere in the range are measured once and every row reuses them."""
    measured = _refuse_turn()
    union = measured.requests_over(FLOOR_CANDIDATES)
    assert len(union) == 1
    assert len(measured.verdicts) == len(union)


# --- The hard constraint -------------------------------------------------------------------


def _cell(floor: float, refuse: int, answerable: int, pairs: int = 0) -> SweepCell:
    return SweepCell(
        floor=floor,
        refuse_passed=refuse,
        refuse_total=8,
        answerable_passed=answerable,
        answerable_total=8,
        escalated_pairs=pairs,
    )


def test_a_floor_that_loses_a_must_refuse_item_is_rejected_however_well_it_scores():
    """Section 8's gate 1 applied to a calibration: a floor cannot buy answerable pass rate with
    a refusal, and the trade is refused before pass rate is looked at."""
    baseline = _cell(CURRENT_FLOOR, refuse=7, answerable=5)
    cells = [baseline, _cell(0.95, refuse=6, answerable=8)]

    assert not cells[1].feasible(baseline)
    assert recommend(cells, baseline) is baseline


def test_a_floor_that_gains_a_must_refuse_item_is_eligible():
    """The constraint is "never lower", not "exactly equal": a floor that refuses *more*
    out-of-corpus questions is not disqualified for improving on the baseline."""
    baseline = _cell(CURRENT_FLOOR, refuse=7, answerable=5)
    better = _cell(0.80, refuse=8, answerable=5)
    assert better.feasible(baseline)
    assert recommend([baseline, better], baseline) is better


def test_the_recommendation_maximizes_answerable_pass_before_anything_else():
    baseline = _cell(CURRENT_FLOOR, refuse=7, answerable=5)
    cells = [baseline, _cell(0.65, refuse=7, answerable=6), _cell(0.70, refuse=8, answerable=4)]
    assert recommend(cells, baseline).floor == 0.65


def test_the_lowest_floor_wins_a_tie_because_it_escalates_fewest_pairs():
    """Two floors with the same verdicts and the same margin differ only in cost, and each
    escalation is a model call on every real turn."""
    baseline = _cell(CURRENT_FLOOR, refuse=7, answerable=5)
    cells = [baseline, _cell(0.85, refuse=7, answerable=5, pairs=9)]
    assert recommend(cells, baseline).floor == CURRENT_FLOOR


def test_the_margin_tie_break_prefers_a_floor_that_is_not_sitting_on_a_measured_overlap():
    """Among floors tied on both verdict columns, the one furthest from any measured overlap is
    the one that keeps gating the same way when a claim is reworded slightly."""
    measured = [_answerable_turn(verdict="yes")]
    overlap = _overlap_of(measured[0])
    on_the_value = min(FLOOR_CANDIDATES, key=lambda floor: abs(floor - overlap))
    baseline = _cell(CURRENT_FLOOR, refuse=8, answerable=8)
    cells = [_cell(floor, refuse=8, answerable=8) for floor in (on_the_value, 0.95)]

    best = recommend(cells, baseline, measured)
    assert best.floor != on_the_value


def test_the_baseline_row_is_measured_independently_of_the_grid():
    """`baseline_cell` evaluates `CURRENT_FLOOR` itself rather than looking it up, so the thing
    every row is compared against does not depend on the grid happening to contain it."""
    measured = [_refuse_turn(), _answerable_turn()]
    assert baseline_cell(measured) == evaluate(measured, CURRENT_FLOOR)


def test_recommend_returns_none_only_when_there_is_nothing_to_sweep():
    assert recommend([], _cell(CURRENT_FLOOR, refuse=8, answerable=8)) is None


# --- Saving and replaying ------------------------------------------------------------------


def test_a_saved_collection_round_trips_into_the_same_sweep():
    """The expensive half runs once. A saved collection that swept differently on replay would
    make the published table unreproducible."""
    measured = [_refuse_turn(), _answerable_turn()]
    payload = json.loads(json.dumps(measurements_as_dict(measured)))
    replayed = measurements_from_dict(payload)

    assert sweep(replayed.turns) == sweep(measured)


def test_a_saved_turn_the_golden_set_does_not_recognise_is_an_error_not_a_skip():
    """Dropping it quietly would shrink the denominator of every row in the table."""
    payload = measurements_as_dict([_refuse_turn()])
    payload["turns"][0]["item_id"] = "corpus-refuse-a-question-nobody-wrote"
    with pytest.raises(KeyError):
        measurements_from_dict(payload)


def test_an_errored_item_survives_the_round_trip_and_is_named_in_the_report():
    """A billed run that lost an item must say so where the numbers are: an errored item is the
    absence of a result, and it contributes to no row -- so the denominators stay honest."""
    collection = Collection(
        turns=(_refuse_turn(),),
        errors=(CollectionError("corpus-refuse-last-service-date", "ServingUnreachable: down"),),
    )
    replayed = measurements_from_dict(json.loads(json.dumps(measurements_as_dict(collection))))
    assert replayed.errors == collection.errors

    text = "\n".join(format_errors(replayed.errors))
    assert "corpus-refuse-last-service-date" in text
    assert "did not measure at all" in text
    assert "not a verdict on the full 16" in text


def test_a_collection_with_no_prose_citing_claim_recommends_nothing():
    """With no measured pair for the floor to threshold, every row is identical and a value
    picked from them would be a value nobody measured. `main` refuses, and exits non-zero."""
    payload = {
        "source": {"source_type": "live_endpoint", "source_id": "GET /monitoring/drift"},
        "data": {"status": {"file_count": 197}},
    }
    live_only = _measured(
        _ANSWERABLE_ITEM,
        [_claim("Bearing 2nd_test-demo is tracked.", "GET /monitoring/drift")],
        [payload],
        {},
    )
    assert not any_prose([live_only])


# --- Reporting -----------------------------------------------------------------------------


def test_the_sweep_table_prints_every_floor_including_the_rejected_ones():
    """Issue #173: print the full table, not the winner. A row rejected by the hard constraint
    is printed and marked, because "this floor would score better if the constraint did not
    exist" is the trade a reader is entitled to watch being refused."""
    baseline = _cell(CURRENT_FLOOR, refuse=8, answerable=5)
    cells = [
        _cell(floor, refuse=8 if floor <= 0.70 else 7, answerable=5)
        for floor in FLOOR_CANDIDATES
    ]
    text = "\n".join(format_sweep(cells, baseline, baseline))

    for floor in FLOOR_CANDIDATES:
        assert f"{floor:>5.2f}" in text
    assert "REGRESS" in text
    assert "current" in text


def test_the_recommendation_names_the_current_value_and_does_not_apply_it():
    baseline = _cell(CURRENT_FLOOR, refuse=8, answerable=5)
    text = "\n".join(format_recommendation(_cell(0.75, refuse=8, answerable=7), baseline))

    assert "LEXICAL_OVERLAP_FLOOR = 0.75" in text
    assert f"currently {CURRENT_FLOOR}" in text
    assert "does not apply" in text
    assert "separate, reviewed change" in text


def test_the_recommendation_says_plainly_when_it_is_to_keep_the_current_value():
    baseline = _cell(CURRENT_FLOOR, refuse=8, answerable=5)
    text = "\n".join(format_recommendation(baseline, baseline))
    assert "keep the current value" in text


def test_separability_reports_the_two_populations_a_floor_sits_between():
    """Not the two categories: a `yes` pair belongs to neither, and an answerable claim the
    critic rejects is what makes a high floor expensive."""
    measured = [_refuse_turn(verdict="no"), _answerable_turn(verdict="unclear")]
    text = "\n".join(format_separability(measured))

    assert "must escalate" in text
    assert "must not escalate" in text
    assert _REFUSE_ITEM.item_id in text
    assert _ANSWERABLE_ITEM.item_id in text


def test_separability_says_so_when_no_measured_pair_has_a_non_yes_verdict():
    text = "\n".join(format_separability([_answerable_turn(verdict="yes")]))
    assert "nothing to separate" in text


def test_the_reach_section_names_why_an_item_is_out_of_the_floors_reach():
    """The reading Issue #173 asks the PR to state plainly, computed rather than argued."""
    unreachable = _answerable_turn(verdict="yes")
    text = "\n".join(format_reach(reaches([unreachable], sweep([unreachable]))))

    assert "out of reach" in text
    assert unreachable.item_id in text
