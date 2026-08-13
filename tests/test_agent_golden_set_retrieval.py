"""Tier-1 tests for golden-set retrieval quality (Issue #152, part 3b).

No model call, no API key, no network. Distinct from `tests/test_agent_golden_set_runner.py`
in subject as well as in file: 3a's tests are about the gates, and every test here is about a
number that is deliberately **not** in them.

Three halves, in the same order the module builds them:

- **The ranked list is read off real envelopes.** Every payload a retrieval metric is
  computed from here is minted by calling the real `src.agent.mcp.tools.search_documentation`
  with a stubbed `search` function -- so the `results`/`source`/`score` shape under test is
  the one the tool actually produces, not a hand-written imitation of it. One test goes
  further and uses `tests/fixtures/answerer_turn.json`, a turn recorded against real
  infrastructure with three real chunk ids and real cosine similarities.
- **The metrics are checked one rule at a time** against small outcomes, so a failure names a
  rule rather than a pipeline.
- **The additive-ness is checked as its own property**: that the two gates, their dict keys
  and 3a's other readers of the same payload shape are bit-for-bit what they were.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src.agent.critic.deterministic import TurnEvidence
from src.agent.critic.retrieval_confidence import TAU_TOP, assess_retrieval
from src.agent.mcp import tools
from src.agent.mcp.readonly_server import READONLY_TOOL_NAMES
from src.agent.mcp.results import payload_of
from src.agent.rag.retrieval import DEFAULT_LIMIT, RetrievedChunk
from tests.fixtures.golden_set import GoldenSetItem, load_golden_set
from tests.fixtures.golden_set_retrieval import (
    BELOW_THRESHOLD,
    CELL_UNGROUNDED_CORRECT,
    K,
    NOT_APPLICABLE,
    RECALL_PRECISION,
    SEARCH_TOOL_NAME,
    RankedChunk,
    RetrievalQuality,
    SearchOutcome,
    format_retrieval_section,
    is_search_payload,
    precision_at_k,
    ranked_chunks,
    recall_at_k,
    retrieval_as_dict,
    retrieval_kind,
    score_retrieval,
    search_outcome,
    summarize_retrieval,
    top_score_below_tau,
)
from tests.fixtures.golden_set_runner import (
    CORRECT_TOOL_CALL,
    GROUNDED_ANSWER,
    MUST_REFUSE,
    ItemRun,
    ItemScore,
    SubScore,
    format_report,
    report_as_dict,
    resolve_tool_name,
    resolve_tool_names,
    summarize,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# --- Real envelopes, built by the real tool -----------------------------------------------


def _chunk(chunk_id: str, score: float) -> RetrievedChunk:
    """One `RetrievedChunk` as `src/agent/rag/retrieval.py` builds them, with a `chunk_id`
    spelled the way `schema.py` spells it (`source_id::chunk_index`)."""
    source_id, _, index = chunk_id.rpartition("::")
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_type="decision_doc",
        source_id=source_id,
        source_ref=source_id,
        heading_path="A heading",
        chunk_index=int(index),
        text=f"text of {chunk_id}",
        score=score,
    )


def search_payload(*ranked: tuple[str, float]) -> dict:
    """A real `search_documentation` success envelope carrying `ranked`, in that order."""
    chunks = [_chunk(chunk_id, score) for chunk_id, score in ranked]
    return payload_of(tools.search_documentation("q", search=lambda *a, **k: chunks))


def failed_search_payload() -> dict:
    """The real failure envelope: the documentation index is unreachable."""

    def _unreachable(*args, **kwargs):
        raise RuntimeError("the documentation index is not reachable in this test")

    return payload_of(tools.search_documentation("q", search=_unreachable))


def _closed_port_payload() -> dict:
    """A real non-search payload -- `get_bearing_status` against a port nothing listens on."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return payload_of(tools.get_bearing_status(base_url=f"http://127.0.0.1:{port}"))


# --- k ------------------------------------------------------------------------------------


def test_k_is_search_documentations_own_default_limit():
    """Issue #152: k stays at `DEFAULT_LIMIT` unless a PR says otherwise explicitly. Pinned
    against the module that owns it, so a change there cannot silently redefine every number
    this file measures against #146's hand-verified labels."""
    assert K == DEFAULT_LIMIT == 5


# --- Reading the ranked list --------------------------------------------------------------


def test_the_ranked_list_comes_off_a_real_search_envelope_in_rank_order():
    payload = search_payload(
        ("docs/a.md::0", 0.81), ("docs/a.md::1", 0.77), ("docs/b.md::3", 0.7)
    )
    assert ranked_chunks([payload]) == (
        RankedChunk("docs/a.md::0", 0.81),
        RankedChunk("docs/a.md::1", 0.77),
        RankedChunk("docs/b.md::3", 0.7),
    )


def test_a_lower_scored_hit_listed_first_is_still_ranked_by_score():
    """The metrics define "top k" by score, so a payload whose order and scores disagree is
    read by score. A single real search call never produces this -- two calls in one turn
    can."""
    first = search_payload(("docs/a.md::0", 0.5))
    second = search_payload(("docs/b.md::0", 0.9))
    assert [chunk.chunk_id for chunk in ranked_chunks([first, second])] == [
        "docs/b.md::0",
        "docs/a.md::0",
    ]


def test_a_chunk_retrieved_by_two_searches_is_counted_once_at_its_best_score():
    """Otherwise a duplicate inflates precision@k's denominator with a chunk seen once."""
    first = search_payload(("docs/a.md::0", 0.6), ("docs/b.md::0", 0.55))
    second = search_payload(("docs/a.md::0", 0.72))
    ranked = ranked_chunks([first, second])
    assert [chunk.chunk_id for chunk in ranked] == ["docs/a.md::0", "docs/b.md::0"]
    assert ranked[0].score == pytest.approx(0.72)


def test_non_search_payloads_contribute_nothing_to_the_ranked_list():
    """A live-tool result carries a `source` block too, and its `source_id` is not a chunk."""
    payload = _closed_port_payload()
    assert is_search_payload(payload) is False
    assert ranked_chunks([payload]) == ()
    assert search_outcome([payload]) == SearchOutcome(ranked=(), searched=False, failed=False)


def test_a_search_payload_is_identified_by_the_source_block_the_tool_minted():
    """And it resolves, through 3a's own mapping, to the name this module assumes."""
    payload = search_payload(("docs/a.md::0", 0.5))
    assert is_search_payload(payload) is True
    assert resolve_tool_name(payload["source"]) == SEARCH_TOOL_NAME
    assert SEARCH_TOOL_NAME in READONLY_TOOL_NAMES


def test_a_failed_search_is_searched_and_failed_rather_than_an_empty_retrieval():
    """The distinction the must-refuse mirror metric depends on: an unreachable index is not
    a below-threshold result, it is the absence of one."""
    outcome = search_outcome([failed_search_payload()])
    assert outcome.searched is True
    assert outcome.failed is True
    assert outcome.ranked == ()
    assert outcome.top_score is None


def test_a_turn_that_never_searched_is_distinguishable_from_one_that_retrieved_nothing():
    never = search_outcome([_closed_port_payload()])
    empty = search_outcome([search_payload()])
    assert (never.searched, never.ranked) == (False, ())
    assert (empty.searched, empty.failed, empty.ranked) == (True, False, ())


def test_the_ranked_list_of_a_real_recorded_turn_is_its_three_real_chunks():
    """`tests/fixtures/answerer_turn.json` was recorded against real infrastructure (#116):
    three real chunk ids with real cosine similarities, alongside a drift result and an
    inventory result that must not leak into the ranking."""
    recorded = json.loads((FIXTURES / "answerer_turn.json").read_text(encoding="utf-8"))
    outcome = search_outcome(recorded["tool_payloads"])
    assert outcome.searched is True
    assert outcome.failed is False
    assert [chunk.chunk_id for chunk in outcome.ranked] == [
        "docs/class_imbalance_decision.md::8",
        "docs/class_imbalance_decision.md::9",
        "docs/model_training_decision.md::9",
    ]
    assert outcome.top_score == pytest.approx(0.7930768)


# --- recall@k -----------------------------------------------------------------------------


def _outcome(*ranked: tuple[str, float]) -> SearchOutcome:
    return search_outcome([search_payload(*ranked)])


def test_recall_needs_only_one_relevant_chunk_not_all_of_them():
    """Section 8's "at least one": the ~200-character overlap puts a fact in two adjacent
    chunks and either is a legitimate retrieval."""
    outcome = _outcome(("docs/a.md::0", 0.8), ("docs/z.md::9", 0.7))
    relevant = frozenset({"docs/a.md::0", "docs/a.md::1"})
    assert recall_at_k(outcome, relevant) is True


def test_recall_is_a_miss_when_no_relevant_chunk_was_retrieved():
    outcome = _outcome(("docs/z.md::9", 0.7), ("docs/z.md::8", 0.6))
    assert recall_at_k(outcome, frozenset({"docs/a.md::0"})) is False


def test_recall_is_a_miss_when_nothing_was_retrieved_at_all():
    assert recall_at_k(SearchOutcome(), frozenset({"docs/a.md::0"})) is False


def test_a_relevant_chunk_below_rank_k_does_not_count_as_retrieved():
    """The whole point of measuring @k: the chunk existed in the index and the answerer never
    saw it."""
    ranked = tuple((f"docs/z.md::{i}", 0.9 - i / 100) for i in range(K))
    outcome = _outcome(*ranked, ("docs/a.md::0", 0.4))
    assert outcome.retrieved_count == K + 1
    assert len(outcome.top_k(K)) == K
    assert recall_at_k(outcome, frozenset({"docs/a.md::0"})) is False


# --- precision@k --------------------------------------------------------------------------


def test_precision_is_the_fraction_of_the_ranked_list_that_was_relevant():
    outcome = _outcome(
        ("docs/a.md::0", 0.9),
        ("docs/z.md::1", 0.8),
        ("docs/a.md::1", 0.7),
        ("docs/z.md::2", 0.6),
    )
    relevant = frozenset({"docs/a.md::0", "docs/a.md::1"})
    assert precision_at_k(outcome, relevant) == pytest.approx(0.5)


def test_precision_denominator_is_what_was_returned_not_k_when_fewer_came_back():
    """Three of three relevant is precision 1.0, not 0.6 -- a search that returned fewer than
    k chunks was not thereby imprecise."""
    outcome = _outcome(("docs/a.md::0", 0.9), ("docs/a.md::1", 0.8), ("docs/a.md::2", 0.7))
    relevant = frozenset({"docs/a.md::0", "docs/a.md::1", "docs/a.md::2"})
    assert precision_at_k(outcome, relevant) == pytest.approx(1.0)


def test_precision_is_none_rather_than_zero_when_nothing_was_retrieved():
    """`None`, so an item that measured nothing cannot drag the mean down as if it had
    measured badly. Its recall is still an unambiguous miss."""
    outcome = SearchOutcome(searched=True)
    assert precision_at_k(outcome, frozenset({"docs/a.md::0"})) is None
    assert recall_at_k(outcome, frozenset({"docs/a.md::0"})) is False


def test_good_recall_with_poor_precision_is_visible_as_two_different_numbers():
    """Section 8's stated reason for computing both: "the right chunk is present but buried
    among near-misses" is a distinct pathology from not finding it."""
    outcome = _outcome(
        ("docs/z.md::1", 0.9),
        ("docs/z.md::2", 0.88),
        ("docs/z.md::3", 0.86),
        ("docs/z.md::4", 0.84),
        ("docs/a.md::0", 0.82),
    )
    relevant = frozenset({"docs/a.md::0"})
    assert recall_at_k(outcome, relevant) is True
    assert precision_at_k(outcome, relevant) == pytest.approx(0.2)


# --- The must-refuse mirror metric --------------------------------------------------------


def test_the_mirror_metric_passes_when_the_top_score_stayed_below_tau_top():
    assert top_score_below_tau(_outcome(("docs/a.md::0", TAU_TOP - 0.01))) is True


def test_the_mirror_metric_fails_when_a_chunk_cleared_tau_top():
    """Section 8: a refusal issued while a spuriously similar chunk cleared threshold "came
    out right for the wrong reason"."""
    assert top_score_below_tau(_outcome(("docs/a.md::0", TAU_TOP + 0.01))) is False


def test_the_mirror_metric_is_the_top_score_alone_not_assess_retrievals_two_part_gate():
    """One chunk over `TAU_TOP` with nothing behind it fails this metric, while
    `assess_retrieval` calls the same turn "not passed" for want of a second supporting
    chunk. Reusing that verdict would invert this metric on the exact shape it exists to
    catch."""
    outcome = _outcome(("docs/a.md::0", TAU_TOP + 0.2))
    assert assess_retrieval([chunk.score for chunk in outcome.ranked]).passed is False
    assert top_score_below_tau(outcome) is False


def test_the_mirror_metric_passes_on_nothing_retrieved_but_says_it_measured_nothing():
    """Issue #152's own wording ("or 'nothing retrieved'"), with the trivial pass counted out
    loud so a run made against an unreachable index cannot read as a green safety metric."""
    outcome = search_outcome([failed_search_payload()])
    assert top_score_below_tau(outcome) is True

    quality = score_retrieval(
        _refuse_item(), outcome, must_refuse_category=MUST_REFUSE, answer_correct=True
    )
    assert quality.top_below_tau is True
    assert quality.trivially_below_tau is True

    report = summarize_retrieval([quality])
    assert (report.mirror_passed, report.mirror_total, report.mirror_trivial) == (1, 1, 1)
    assert "passed trivially" in "\n".join(format_retrieval_section(report))


def test_a_real_below_threshold_pass_is_not_counted_as_trivial():
    quality = score_retrieval(
        _refuse_item(),
        _outcome(("docs/a.md::0", TAU_TOP - 0.05)),
        must_refuse_category=MUST_REFUSE,
        answer_correct=True,
    )
    assert (quality.top_below_tau, quality.trivially_below_tau) == (True, False)
    assert summarize_retrieval([quality]).mirror_trivial == 0


def test_the_mirror_metric_is_not_folded_into_recall_or_precision():
    """Issue #152 task 4: "reported as its own pass/fail". A must-refuse item has no recall
    or precision reading at all, and it is not one of the recall report's items."""
    quality = score_retrieval(
        _refuse_item(),
        _outcome(("docs/a.md::0", 0.9)),
        must_refuse_category=MUST_REFUSE,
        answer_correct=True,
    )
    assert quality.kind == BELOW_THRESHOLD
    assert quality.recall_at_k is None
    assert quality.precision_at_k is None

    report = summarize_retrieval([quality])
    assert report.recall_total == 0
    assert report.mean_precision is None
    assert report.table.total == 0


# --- Which items get a reading ------------------------------------------------------------


def _corpus_item(**overrides) -> GoldenSetItem:
    defaults = dict(
        item_id="corpus-item",
        category="Answerable from the docs",
        question="q",
        expected_tool_names=frozenset({SEARCH_TOOL_NAME}),
        allowed_source_ids=frozenset({"docs/a.md"}),
        relevant_chunk_ids=frozenset({"docs/a.md::0"}),
    )
    return GoldenSetItem(**{**defaults, **overrides})


def _refuse_item(**overrides) -> GoldenSetItem:
    defaults = dict(
        item_id="refuse-item",
        category=MUST_REFUSE,
        question="q",
        expected_tool_names=frozenset({SEARCH_TOOL_NAME}),
    )
    return GoldenSetItem(**{**defaults, **overrides})


def _tool_item(**overrides) -> GoldenSetItem:
    defaults = dict(
        item_id="tool-item",
        category="Inventory",
        question="q",
        expected_tool_names=frozenset({"check_inventory"}),
    )
    return GoldenSetItem(**{**defaults, **overrides})


def test_a_tool_grounded_item_gets_no_retrieval_reading_at_all():
    """Issue #152's constraint. A recall@k of 0 for an item that correctly never searched
    would look like a failure and mean nothing."""
    quality = score_retrieval(
        _tool_item(),
        search_outcome([_closed_port_payload()]),
        must_refuse_category=MUST_REFUSE,
        answer_correct=True,
    )
    assert quality.kind == NOT_APPLICABLE
    assert quality.applicable is False
    assert (quality.recall_at_k, quality.precision_at_k, quality.top_below_tau) == (
        None,
        None,
        None,
    )


def test_the_real_golden_set_partitions_into_eight_eight_and_fourteen():
    """The module docstring's table, against the committed golden set rather than a claim
    about it: 8 items with relevant chunks, the 8 must-refuse mirrors, 14 with no reading."""
    kinds = [
        retrieval_kind(item, must_refuse_category=MUST_REFUSE) for item in load_golden_set()
    ]
    assert kinds.count(RECALL_PRECISION) == 8
    assert kinds.count(BELOW_THRESHOLD) == 8
    assert kinds.count(NOT_APPLICABLE) == 14


def test_every_must_refuse_item_really_declares_no_relevant_chunks():
    """The mirror metric's precondition, checked rather than assumed -- a must-refuse item
    that grew a `relevant_chunk_ids` would silently switch to recall/precision."""
    for item in load_golden_set():
        if item.category == MUST_REFUSE:
            assert item.relevant_chunk_ids == frozenset(), item.item_id


def test_an_item_whose_run_produced_no_observation_is_marked_unscored_not_missed():
    """An errored run has no retrieval to read. Scoring it as recall 0 would put an absence
    of a result into a cell of the table."""
    quality = score_retrieval(
        _corpus_item(), None, must_refuse_category=MUST_REFUSE, answer_correct=None
    )
    assert quality.kind == RECALL_PRECISION
    assert quality.scored is False
    assert quality.recall_at_k is None
    report = summarize_retrieval([quality])
    assert report.recall_total == 0
    assert report.table.not_scored == 1


# --- The 2x2 ------------------------------------------------------------------------------


def _quality(recall: bool, correct: bool, item_id: str) -> RetrievalQuality:
    return RetrievalQuality(
        item_id=item_id,
        category="Answerable from the docs",
        kind=RECALL_PRECISION,
        scored=True,
        searched=True,
        recall_at_k=recall,
        precision_at_k=0.2,
        answer_correct=correct,
    )


def test_the_two_by_two_separates_bad_retrieval_from_bad_generation():
    report = summarize_retrieval(
        [
            _quality(True, True, "a"),
            _quality(True, False, "b"),
            _quality(False, True, "c"),
            _quality(False, False, "d"),
            _quality(False, False, "e"),
        ]
    )
    table = report.table
    assert (table.retrieved_correct, table.retrieved_wrong) == (1, 1)
    assert (table.missed_correct, table.missed_wrong) == (1, 2)
    assert table.total == 5


def test_the_right_answer_no_evidence_cell_is_reported_as_a_failure():
    """Section 8 records that cell as a failure, "not a pass" -- and it is the cell an
    end-to-end score cannot see."""
    report = summarize_retrieval([_quality(False, True, "a"), _quality(True, True, "b")])
    assert report.table.ungrounded_correct == 1
    assert report.table.no_ungrounded_correct_answers is False
    text = "\n".join(format_retrieval_section(report))
    assert CELL_UNGROUNDED_CORRECT in text
    assert "FAILURE" in text


def test_a_clean_table_says_no_ungrounded_correct_answers():
    report = summarize_retrieval([_quality(True, True, "a"), _quality(True, False, "b")])
    assert report.table.no_ungrounded_correct_answers is True


def test_the_two_by_two_is_built_only_from_recall_scored_items():
    """Must-refuse items are next to the table, not a row of it: their retrieval axis asks a
    different question, and one axis labelled two ways is not a 2x2."""
    mirror = score_retrieval(
        _refuse_item(),
        _outcome(("docs/a.md::0", 0.9)),
        must_refuse_category=MUST_REFUSE,
        answer_correct=True,
    )
    report = summarize_retrieval([_quality(True, True, "a"), mirror])
    assert report.table.total == 1
    assert report.mirror_total == 1


# --- Aggregates ---------------------------------------------------------------------------


def test_the_precision_mean_excludes_items_that_retrieved_nothing_and_says_how_many():
    measured = score_retrieval(
        _corpus_item(item_id="measured"),
        _outcome(("docs/a.md::0", 0.9), ("docs/z.md::0", 0.8)),
        must_refuse_category=MUST_REFUSE,
        answer_correct=True,
    )
    nothing = score_retrieval(
        _corpus_item(item_id="nothing"),
        search_outcome([failed_search_payload()]),
        must_refuse_category=MUST_REFUSE,
        answer_correct=False,
    )
    report = summarize_retrieval([measured, nothing])
    assert report.mean_precision == pytest.approx(0.5)
    assert report.precision_undefined == 1
    assert (report.recall_hits, report.recall_total) == (1, 2)
    text = "\n".join(format_retrieval_section(report))
    assert "1 retrieved nothing" in text


def test_retrieval_as_dict_reports_recall_precision_and_the_mirror_under_separate_keys():
    report = summarize_retrieval(
        [
            _quality(True, True, "a"),
            score_retrieval(
                _refuse_item(),
                _outcome(("docs/a.md::0", 0.9)),
                must_refuse_category=MUST_REFUSE,
                answer_correct=True,
            ),
        ]
    )
    data = retrieval_as_dict(report)
    assert data["k"] == K
    assert data["recall_at_k"] == {"hits": 1, "total": 1, "rate": 1.0}
    assert data["must_refuse_mirror"]["below_threshold"] == 0
    assert data["two_by_two"]["retrieved_answer_correct"] == 1
    assert data["two_by_two"]["ungrounded_correct_is_a_failure"] is True
    assert {entry["item_id"] for entry in data["per_item"]} == {"a", "refuse-item"}


# --- Additive: nothing above this line changed anything below it --------------------------


def _score(item: GoldenSetItem, passed: bool, run: ItemRun) -> ItemScore:
    return ItemScore(
        item_id=item.item_id,
        category=item.category,
        sub_scores=(
            SubScore(CORRECT_TOOL_CALL, True, passed),
            SubScore(GROUNDED_ANSWER, True, passed),
        ),
        run=run,
    )


def test_the_two_gates_are_unchanged_by_the_retrieval_reading():
    """The property Issue #152 turns on: a report with a retrieval section computes both
    gates from exactly the sub-scores 3a computed them from."""
    item = _corpus_item()
    run = ItemRun(item_id=item.item_id, search=_outcome(("docs/z.md::0", 0.9)))
    scores = [_score(item, True, run)]

    with_items = summarize(scores, items=[item])
    without_items = summarize(scores, items=[])

    assert with_items.retrieval is not None and with_items.retrieval.recall_total == 1
    assert without_items.retrieval is not None and without_items.retrieval.recall_total == 0
    for report in (with_items, without_items):
        assert report.must_refuse_gate_passed is False  # no must-refuse item in this subset
        assert report.remaining_passed == 1
        assert report.remaining_total == 1
        assert report.gates_passed == report.must_refuse_gate_passed


def test_a_failing_retrieval_reading_does_not_fail_a_passing_item():
    """recall 0 on an item whose answer passed both sub-scores: the 2x2 records the
    'right answer, no evidence' cell, and the item still passes 3a's gates."""
    item = _corpus_item()
    run = ItemRun(item_id=item.item_id, search=_outcome(("docs/z.md::0", 0.9)))
    report = summarize([_score(item, True, run)], items=[item])

    assert report.scores[0].passed is True
    assert report.retrieval.table.missed_correct == 1
    assert report.retrieval.table.no_ungrounded_correct_answers is False
    assert report.remaining_gate_passed is True


def test_report_as_dict_keeps_every_key_it_had_and_adds_exactly_one():
    item = _corpus_item()
    scores = [_score(item, True, ItemRun(item_id=item.item_id, search=_outcome()))]
    data = report_as_dict(summarize(scores, items=[item]))
    assert set(data) == {
        "mode",
        "attempts_per_item",
        "categories",
        "must_refuse_gate",
        "remaining_gate",
        "failures",
        "retrieval_quality",
    }
    assert set(data["must_refuse_gate"]) == {"passed", "total", "floor", "gate_passed"}


def test_a_report_with_no_retrieval_reading_omits_the_key_rather_than_nulling_it():
    """"Not measured" and "measured as zero" must not look the same to a consumer."""
    from tests.fixtures.golden_set_runner import CategoryResult, GoldenSetReport

    bare = GoldenSetReport(mode="replay", attempts=1, categories=(CategoryResult("x", 0, 0),))
    assert bare.retrieval is None
    assert "retrieval_quality" not in report_as_dict(bare)
    assert "retrieval quality" not in format_report(bare)


def test_the_printed_report_puts_retrieval_beneath_the_gates_and_says_it_is_not_in_them():
    item = _corpus_item()
    run = ItemRun(item_id=item.item_id, search=_outcome(("docs/a.md::0", 0.9)))
    text = format_report(summarize([_score(item, True, run)], items=[item]))
    assert text.index("gate 2") < text.index("retrieval quality")
    assert "folded into neither" in text
    assert f"recall@{K}" in text and f"precision@{K}" in text


def test_the_existing_readers_of_the_same_payload_shape_see_exactly_what_they_saw():
    """Issue #152's "without disturbing what 3a already reads from the same payload shape":
    the tool-name resolution and the critic's evidence assembly are run over the same
    payloads the ranked list is read from, and still produce their own answers."""
    payloads = [search_payload(("docs/a.md::0", 0.9)), _closed_port_payload()]

    assert resolve_tool_names(payloads) == frozenset(
        {SEARCH_TOOL_NAME, "get_bearing_status"}
    )
    evidence = TurnEvidence.from_tool_payloads(payloads)
    assert "docs/a.md::0" in evidence.source_ids
    assert evidence.retrieval_scores == (pytest.approx(0.9),)
    assert search_outcome(payloads).ranked == (RankedChunk("docs/a.md::0", 0.9),)


def test_item_run_carries_the_ranked_list_of_a_real_recorded_turn():
    """The seam, on the same recorded turn 3a's own seam test uses: `ItemRun.from_turn` now
    also carries the retrieval observation, and every field 3a asserted on is unchanged."""
    from src.agent.answerer import Draft
    from src.agent.pipeline import turn_from_payloads, verify_turn_async

    recorded = json.loads((FIXTURES / "answerer_turn.json").read_text(encoding="utf-8"))
    turn = turn_from_payloads(
        Draft.model_validate(recorded["draft"]), recorded["tool_payloads"]
    )
    response = asyncio.run(verify_turn_async(turn, escalate=False))
    run = ItemRun.from_turn(_corpus_item(item_id="recorded"), turn, response)

    assert run.tool_names == frozenset(
        {"get_bearing_status", "search_documentation", "check_inventory"}
    )
    assert run.grounding_tier == response.grounding_tier
    assert run.n_claims == len(response.claims)
    assert run.search is not None
    assert [chunk.chunk_id for chunk in run.search.ranked] == [
        "docs/class_imbalance_decision.md::8",
        "docs/class_imbalance_decision.md::9",
        "docs/model_training_decision.md::9",
    ]

    item = _corpus_item(
        item_id="recorded",
        relevant_chunk_ids=frozenset({"docs/class_imbalance_decision.md::9"}),
    )
    quality = score_retrieval(
        item, run.search, must_refuse_category=MUST_REFUSE, answer_correct=True
    )
    assert quality.recall_at_k is True
    assert quality.precision_at_k == pytest.approx(1 / 3)
