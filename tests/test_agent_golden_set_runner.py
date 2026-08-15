"""Tier-1 tests for the golden-set runner, tool resolution and gate (Issue #150, part 3a).

No model call, no API key, no network. Two halves:

- **The tool-name mapping is checked against the five real tools**, by calling each one for
  real and resolving the `source` block it actually minted -- not against a hand-written
  payload, which would only prove the mapping agrees with itself. Every tool is driven down
  a path that needs no network: the three HTTP-backed ones are pointed at a closed port and
  return their real failure envelopes, which carry the same minted `source` block a success
  does.
- **The scoring and reporting are checked against small hand-built `ItemRun`s**, so each
  rule is exercised in isolation and a failure names one rule rather than a pipeline.
"""
from __future__ import annotations

import socket

import pytest

from src.agent.inventory.build_db import build_db
from src.agent.mcp import tools
from src.agent.mcp.readonly_server import READONLY_TOOL_NAMES
from src.agent.mcp.results import payload_of
from src.agent.similarity.archive import archive_source_id
from tests.fixtures.cassette import LIVE
from tests.fixtures.golden_set import GoldenSetItem
from tests.fixtures.golden_set_runner import (
    AGGREGATE_FLOOR,
    ATTEMPTS_BY_MODE,
    CORRECT_REFUSAL,
    GROUNDED_ANSWER,
    MUST_REFUSE,
    CategoryResult,
    ItemRun,
    ItemScore,
    ToolResolutionError,
    citation_matches_allowed,
    format_report,
    mapped_tool_names,
    resolve_tool_name,
    resolve_tool_names,
    run_and_score,
    score_item,
    summarize,
    unmapped_readonly_tools,
)
from tests.fixtures import golden_set_runner


def _closed_port_url() -> str:
    """A URL nothing is listening on: bind a port, read it, release it.

    The three HTTP-backed tools then take their `ServingUnreachable` path, which is a real
    tool result with a real minted `source` block -- exactly what the mapping consumes.
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"


@pytest.fixture()
def real_tool_payloads(tmp_path):
    """One real payload from each of the five read-only tools, by name."""
    url = _closed_port_url()
    db_path = tmp_path / "inventory.db"
    build_db(db_path)

    def _unreachable_search(*args, **kwargs):
        raise RuntimeError("the documentation index is not reachable in this test")

    return {
        "get_bearing_status": payload_of(tools.get_bearing_status(base_url=url)),
        "predict_health_state": payload_of(
            tools.predict_health_state("2nd_test-demo", [1.0, 2.0], base_url=url)
        ),
        "check_inventory": payload_of(
            tools.check_inventory(part_number="ZA-2115", db_path=db_path)
        ),
        "search_documentation": payload_of(
            tools.search_documentation("anything", search=_unreachable_search)
        ),
        "find_similar_historical_pattern": payload_of(
            tools.find_similar_historical_pattern("2nd_test-demo", base_url=url)
        ),
    }


# --- The mapping, against the real tools -------------------------------------------------


def test_every_real_readonly_tool_resolves_to_its_own_name(real_tool_payloads):
    """The mapping's whole job, checked end to end on real minted `source` blocks."""
    for name, payload in real_tool_payloads.items():
        assert resolve_tool_name(payload["source"]) == name, payload["source"]


def test_no_two_real_tools_share_a_source_block(real_tool_payloads):
    """Issue #150's stop condition: two tools that could produce the same id would make the
    mapping unsound, and set-equality scoring would then silently accept the wrong tool."""
    resolved = [resolve_tool_name(payload["source"]) for payload in real_tool_payloads.values()]
    assert sorted(resolved) == sorted(READONLY_TOOL_NAMES)
    assert len(set(resolved)) == len(READONLY_TOOL_NAMES)


def test_every_readonly_tool_has_a_mapping_entry():
    """A sixth read-only tool added without a mapping entry fails here, not at scoring time
    where it would resolve to nothing and quietly weaken a set-equality check."""
    assert unmapped_readonly_tools() == frozenset()
    assert mapped_tool_names() == frozenset(READONLY_TOOL_NAMES)


def test_the_archive_source_id_is_matched_by_prefix_in_both_of_its_real_forms():
    """`find_similar_historical_pattern` is the one tool whose `source_id` is not a constant:
    it carries the archive manifest's content hash, or a fallback when the artifact cannot be
    read. Both real forms have to hit the prefix rule."""
    assert resolve_tool_name(
        {"source_type": "trajectory_match", "source_id": archive_source_id()}
    ) == "find_similar_historical_pattern"
    assert resolve_tool_name(
        {"source_type": "trajectory_match", "source_id": tools.ARCHIVE_SOURCE_ID_FALLBACK}
    ) == "find_similar_historical_pattern"


def test_an_unknown_source_block_raises_rather_than_resolving_to_nothing():
    with pytest.raises(ToolResolutionError, match="no read-only tool mints"):
        resolve_tool_name({"source_type": "inventory", "source_id": "somewhere/else.db"})


def test_resolve_tool_names_collapses_repeat_calls_and_keeps_failed_ones(real_tool_payloads):
    payloads = [
        real_tool_payloads["get_bearing_status"],
        real_tool_payloads["get_bearing_status"],
        real_tool_payloads["check_inventory"],
    ]
    assert resolve_tool_names(payloads) == frozenset({"get_bearing_status", "check_inventory"})


def test_the_declared_tools_of_every_real_golden_set_item_are_resolvable():
    """Ties #146/#148's content to this mapping: an item expecting a tool the runner cannot
    name could never pass, however well the agent behaved."""
    from tests.fixtures.golden_set import load_golden_set

    for item in load_golden_set():
        assert item.expected_tool_names <= mapped_tool_names(), item.item_id


# --- Citation matching --------------------------------------------------------------------


def test_a_chunk_id_matches_its_document_in_allowed_source_ids():
    allowed = frozenset({"docs/eda_findings.md"})
    assert citation_matches_allowed("docs/eda_findings.md::4", allowed)
    assert citation_matches_allowed("docs/eda_findings.md", allowed)


def test_a_chunk_of_a_different_document_does_not_match():
    allowed = frozenset({"docs/eda_findings.md"})
    assert not citation_matches_allowed("docs/serving_design.md::4", allowed)
    assert not citation_matches_allowed("docs/eda_findings.md.bak::4", allowed)


def test_a_tool_source_id_is_matched_exactly():
    allowed = frozenset({"GET /monitoring/drift"})
    assert citation_matches_allowed("GET /monitoring/drift", allowed)
    assert not citation_matches_allowed("POST /predict", allowed)


# --- Scoring ------------------------------------------------------------------------------


def _item(**overrides) -> GoldenSetItem:
    defaults = dict(
        item_id="x",
        category="Requires a live tool",
        question="?",
        expected_tool_names=frozenset({"get_bearing_status"}),
        allowed_source_ids=frozenset({"GET /monitoring/drift"}),
    )
    return GoldenSetItem(**{**defaults, **overrides})


def _run(**overrides) -> ItemRun:
    defaults = dict(
        item_id="x",
        tool_names=frozenset({"get_bearing_status"}),
        cited=("GET /monitoring/drift",),
        text="2nd_test-demo is drifting.",
        grounding_tier="grounded",
    )
    return ItemRun(**{**defaults, **overrides})


def test_a_clean_item_passes_every_applicable_sub_score():
    score = score_item(_item(), _run())
    assert score.passed
    assert [sub.name for sub in score.sub_scores] == ["correct_tool_call", GROUNDED_ANSWER]


def test_expected_tool_names_is_set_equality_not_subset():
    """An answer that called the right tool *and* another one fails. This is the rule
    `tools-live-no-raw-signal-to-score` depends on: reaching for `predict_health_state`
    without a signal is only visible as an extra name."""
    extra = _run(tool_names=frozenset({"get_bearing_status", "predict_health_state"}))
    score = score_item(_item(), extra)
    assert not score.passed
    assert any("also called ['predict_health_state']" in reason for reason in score.reasons)


def test_a_missing_expected_tool_fails():
    score = score_item(_item(), _run(tool_names=frozenset()))
    assert not score.passed
    assert any("did not call ['get_bearing_status']" in reason for reason in score.reasons)


def test_citing_outside_the_allowed_set_fails():
    score = score_item(_item(), _run(cited=("GET /monitoring/drift", "POST /predict")))
    assert not score.passed
    assert any("outside allowed_source_ids" in reason for reason in score.reasons)


def test_citing_nothing_fails_even_though_it_is_vacuously_a_subset():
    """Section 8's "at least one of which must be cited", which the subset rule alone cannot
    express -- an empty set is a subset of everything."""
    score = score_item(_item(), _run(cited=()))
    assert not score.passed
    assert any("released no cited claim" in reason for reason in score.reasons)


def test_a_chunk_citation_of_an_allowed_document_passes():
    item = _item(
        expected_tool_names=frozenset({"search_documentation"}),
        allowed_source_ids=frozenset({"docs/eda_findings.md"}),
    )
    run = _run(
        tool_names=frozenset({"search_documentation"}), cited=("docs/eda_findings.md::4",)
    )
    assert score_item(item, run).passed


def test_a_missing_required_substring_fails():
    score = score_item(_item(required_substrings=("3.93",)), _run())
    assert not score.passed
    assert any("required substring(s) absent" in reason for reason in score.reasons)


def test_a_present_forbidden_substring_fails():
    score = score_item(_item(forbidden_substrings=("drifting",)), _run())
    assert not score.passed
    assert any("forbidden substring(s) present" in reason for reason in score.reasons)


# --- Must-refuse items: the structural check instead of the grounding one ------------------


def _refusal_item(**overrides) -> GoldenSetItem:
    defaults = dict(
        item_id="r",
        category=MUST_REFUSE,
        question="?",
        expected_tool_names=frozenset({"search_documentation"}),
        required_substrings=("I don't have a sourced answer",),
    )
    return GoldenSetItem(**{**defaults, **overrides})


def _refusal_run(**overrides) -> ItemRun:
    defaults = dict(
        item_id="r",
        tool_names=frozenset({"search_documentation"}),
        cited=(),
        text="I don't have a sourced answer for this.",
        grounding_tier="ungrounded",
        n_claims=0,
        has_recommendation=False,
    )
    return ItemRun(**{**defaults, **overrides})


def test_a_correct_refusal_is_scored_on_shape_not_on_grounding():
    score = score_item(_refusal_item(), _refusal_run())
    assert score.passed
    assert [sub.name for sub in score.sub_scores] == ["correct_tool_call", CORRECT_REFUSAL]


def test_a_released_answer_fails_a_must_refuse_item():
    score = score_item(
        _refusal_item(),
        _refusal_run(grounding_tier="partial", text="The torque is 40 N·m.", n_claims=1),
    )
    assert not score.passed
    assert any("was released rather than refused" in reason for reason in score.reasons)


def test_a_refusal_carrying_a_recommendation_fails():
    score = score_item(_refusal_item(), _refusal_run(has_recommendation=True))
    assert not score.passed
    assert any("withholds it" in reason for reason in score.reasons)


def test_a_refusal_that_never_searched_still_fails_the_tool_sub_score():
    """Searching and *then* refusing is the correct trajectory; refusing without looking is
    not, and Section 8 scores the two independently."""
    score = score_item(_refusal_item(), _refusal_run(tool_names=frozenset()))
    assert not score.passed
    assert any("did not call" in reason for reason in score.reasons)


# --- Errored runs --------------------------------------------------------------------------


def test_an_errored_run_fails_without_pretending_its_sub_scores_ran():
    score = score_item(_item(), ItemRun(item_id="x", error="CassetteMissing: no cassette"))
    assert not score.passed
    assert all(not sub.applicable for sub in score.sub_scores)
    assert any("did not complete" in reason for reason in score.reasons)


# --- Attempts: Section 8's 3x non-determinism rule ------------------------------------------


def test_live_runs_three_attempts():
    assert ATTEMPTS_BY_MODE[LIVE] == 3


def test_all_three_attempts_must_pass(monkeypatch):
    attempts: list[int] = []

    def _fake_run_item(item, *, serving_url=None, db_path=None):
        attempts.append(len(attempts) + 1)
        # The second attempt calls an extra tool -- the exact non-determinism Section 8's
        # "an item passes only if all 3 pass" exists to catch.
        if len(attempts) == 2:
            return _run(tool_names=frozenset({"get_bearing_status", "check_inventory"}))
        return _run()

    monkeypatch.setattr(golden_set_runner, "run_item", _fake_run_item)
    score = run_and_score(_item())
    assert not score.passed
    assert attempts == [1, 2], "a failing attempt should stop the run and be the one reported"


def test_passing_all_three_attempts_passes(monkeypatch):
    attempts: list[int] = []

    def _fake_run_item(item, *, serving_url=None, db_path=None):
        attempts.append(len(attempts) + 1)
        return _run()

    monkeypatch.setattr(golden_set_runner, "run_item", _fake_run_item)
    assert run_and_score(_item()).passed
    assert attempts == [1, 2, 3]


# --- Reporting ------------------------------------------------------------------------------


def _scored(category: str, passed: bool, index: int) -> ItemScore:
    item = _refusal_item(item_id=f"{category}-{index}") if category == MUST_REFUSE else _item(
        item_id=f"{category}-{index}", category=category
    )
    run = _refusal_run() if category == MUST_REFUSE else _run()
    if not passed:
        run = ItemRun(**{**run.__dict__, "tool_names": frozenset({"nothing_expected"})})
    return score_item(item, run)


def _full_set(must_refuse_passed: int, remaining_passed: int) -> list[ItemScore]:
    """A 30-item result set with exactly the requested number of passes in each half."""
    scores: list[ItemScore] = []
    index = 0
    remaining_budget = remaining_passed
    for category, count in [
        ("Answerable from the docs", 8),
        ("Requires a live tool", 6),
        ("Inventory", 4),
        ("Historical similarity", 4),
    ]:
        for _ in range(count):
            take = remaining_budget > 0
            remaining_budget -= 1 if take else 0
            scores.append(_scored(category, take, index))
            index += 1
    for i in range(8):
        scores.append(_scored(MUST_REFUSE, i < must_refuse_passed, index + i))
    return scores


def test_a_clean_run_passes_both_gates():
    report = summarize(_full_set(8, 22))
    assert report.must_refuse_gate_passed
    assert report.remaining_gate_passed
    assert report.gates_passed


def test_one_must_refuse_failure_fails_its_gate_however_good_the_rest_is():
    """Section 8's whole point: 22/22 elsewhere cannot buy back a single refusal failure."""
    report = summarize(_full_set(7, 22))
    assert not report.must_refuse_gate_passed
    assert report.remaining_gate_passed
    assert not report.gates_passed


def test_the_remaining_floor_is_ninety_percent_of_twenty_two():
    """20/22 is 90.9% and passes; 19/22 is 86.4% and does not. Two failures pass, three do
    not -- the coarseness Section 8 states outright."""
    assert summarize(_full_set(8, 20)).remaining_gate_passed
    assert not summarize(_full_set(8, 19)).remaining_gate_passed


def test_the_two_gates_are_counted_over_disjoint_halves():
    report = summarize(_full_set(8, 22))
    assert report.must_refuse.total == 8
    assert report.remaining_total == 22


def test_an_empty_must_refuse_category_does_not_pass_its_gate_vacuously():
    report = summarize([_scored("Inventory", True, 0)])
    assert not report.must_refuse_gate_passed


def test_the_report_never_offers_a_single_blended_number():
    """The aggregate-hides-the-subgroup failure is cheapest to reintroduce by exposing one
    overall rate for convenience, so the absence is asserted rather than left to review."""
    report = summarize(_full_set(8, 22))
    for forbidden in ("overall_rate", "total_rate", "pass_rate", "score"):
        assert not hasattr(report, forbidden)


def test_the_report_text_shows_every_category_individually_and_both_gates():
    text = format_report(summarize(_full_set(7, 20)))
    for category in ("Answerable from the docs", "Requires a live tool", "Inventory",
                     "Historical similarity", MUST_REFUSE):
        assert category in text
    assert "gate 1 -- must refuse, 100% required" in text
    assert f">= {AGGREGATE_FLOOR:.0%} required" in text
    assert "7/8" in text and "20/22" in text


def test_the_report_text_names_the_mode():
    text = format_report(summarize(_full_set(8, 22)))
    assert "mode: live (3 attempt(s) per item, all must pass)" in text


def test_category_rate_is_zero_rather_than_undefined_for_an_empty_category():
    assert CategoryResult("Inventory", 0, 0).rate == 0.0


def test_a_filtered_run_says_on_its_face_that_it_is_not_a_verdict_on_the_golden_set():
    """`--category Inventory` prints the same two gates over 4 items that a full run prints
    over 30. Without the marker the two are indistinguishable in a PR comment."""
    full = summarize(_full_set(8, 22))
    assert full.is_full_set
    assert "PARTIAL RUN" not in format_report(full)

    partial = summarize([_scored("Inventory", True, i) for i in range(4)])
    assert not partial.is_full_set
    assert "PARTIAL RUN: 4 of 30 items" in format_report(partial)


# --- The seam to the real pipeline ----------------------------------------------------------


def test_item_run_is_built_from_a_real_recorded_turn_and_a_real_grounded_response():
    """`ItemRun.from_turn` against the real thing, with no model call.

    `tests/fixtures/answerer_turn.json` is a turn recorded against real infrastructure
    (#116): real tool payloads from three different tools, and a draft. Rebuilding it and
    running the real critic half is what `tests/test_agent_pipeline.py` already does — this
    adds the one seam this issue owns, that a real turn plus a real `GroundedResponse` become
    an `ItemRun` whose tool names and citations are the ones that were actually produced.

    `escalate=False` so no API key or client is needed; the deterministic tiers are what the
    citations come out of either way.
    """
    import asyncio
    import json
    from pathlib import Path

    from src.agent.answerer import Draft
    from src.agent.pipeline import turn_from_payloads, verify_turn_async

    fixture = Path(__file__).resolve().parents[0] / "fixtures" / "answerer_turn.json"
    recorded = json.loads(fixture.read_text(encoding="utf-8"))
    turn = turn_from_payloads(
        Draft.model_validate(recorded["draft"]), recorded["tool_payloads"]
    )
    response = asyncio.run(verify_turn_async(turn, escalate=False))

    run = ItemRun.from_turn(_item(item_id="recorded"), turn, response)
    assert run.tool_names == frozenset(
        {"get_bearing_status", "search_documentation", "check_inventory"}
    )
    assert run.grounding_tier == response.grounding_tier
    assert run.n_claims == len(response.claims)
    assert run.has_recommendation == (response.recommendation is not None)
    # Every id on the `ItemRun` came off a released claim, and every released claim's ids are
    # in this turn's own evidence — the property the pipeline exists to make true.
    from src.agent.pipeline import evidence_for

    assert set(run.cited) <= evidence_for(turn).source_ids
