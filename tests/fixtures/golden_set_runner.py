"""Section 8 tier 2: run the golden set, score it, and report it (Issue #150, part 3a).

    from tests.fixtures.golden_set_runner import run_and_score, summarize

    results = [run_and_score(item, serving_url=url, db_path=db) for item in load_golden_set()]
    print(format_report(summarize(results)))

Or as a command, against an already-running fixture (see "The fixture" below):

    python -m tests.fixtures.golden_set_runner --url http://localhost:8000

This module composes machinery that already exists and invents none of it: the real
Answerer->Critic pipeline (`src/agent/pipeline.py`), the real read-only MCP tools, the real
serving process, and `tests/fixtures/cassette.py`'s record/replay seam. It lives under
`tests/` for the same reason `cassette.py` does -- Issue #122's "test infrastructure only"
constraint means no module under `src/agent/` may know either of them exists.

**Retrieval quality is not computed here.** recall@k, precision@k, `relevant_chunk_ids` and
Section 8's 2x2 evidence/answer table are Issue #150's sibling (part 3b) and live in
`tests/fixtures/golden_set_retrieval.py`, deliberately built on top of this rather than
inside it. What this module does is *report* them, in their own section beneath the two
gates: `GoldenSetReport.retrieval` is an additive field, no scoring or gate logic below
reads it, and Section 8's "reported as two numbers, never averaged into one" is therefore a
property of the import graph -- the retrieval module cannot reach the gates, and the gates
never see a retrieval number.

## The pipeline, and why the two-step form

`answer_and_verify_async` is the documented entry point and it is what this runs -- but as
its two composed halves (`answer_turn_async` then `verify_turn_async`), which
`src/agent/pipeline.py` defines it as and which `tests/test_agent_pipeline_live.py` already
uses for the same reason: **scoring needs the `AnsweredTurn`**. Task 1 requires resolving
which tools were called, and the tool payloads that answer it live on the turn; a
`GroundedResponse` alone cannot say. Nothing else differs -- same functions, same order, same
module.

## Cassette vs live: the 3x non-determinism rule, resolved

Section 8 says "each item runs 3 times; an item passes only if all 3 pass", and it says that
about a *model*. A cassette replays one fixed HTTP conversation, so replaying it three times
re-derives the same draft three times: it would produce a green "3/3 stable" that measures
nothing. Issue #150 resolves the tension rather than leaving it implicit, and the resolution
is `ATTEMPTS_BY_MODE`:

| mode | attempts | what it measures |
|---|---|---|
| `replay` (default) | **1** | that the harness, tools and gate still work. Free, offline, no key. Not a capability measurement, and the report says so on its face. |
| `live` | **3** | Section 8's real rule. 30 items x 3 = 90 billed calls, so it is run **on demand**, not on every PR. |
| `record` | 1 | one recording per item is what a cassette is. |

Recording three and replaying the first would be worse than either: it would put a number in
the report that no committed artifact can reproduce.

## The fixture

The runner takes `serving_url` and `db_path` as arguments and starts neither, exactly as
`answer_and_verify_async` does. The golden set's items are grounded against the fixture
`tests/fixtures/golden_set_tools.py` documents (Issue #148) -- reproduce it with:

    python -m src.serving.main
    python -m demo.playback --interval 0          # 197 windows as `2nd_test-demo`

The similarity items' `rig-b-new` variant and a live Qdrant are that file's and Section 4's
business, not this module's; an item whose fixture is absent fails honestly on its own
declared substrings rather than being skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.agent.answerer import AnsweredTurn, answer_turn_async
from src.agent.critic.grounding import UNGROUNDED, GroundedResponse
from src.agent.mcp.readonly_server import READONLY_TOOL_NAMES
from src.agent.mcp.serving_client import DRIFT_ENDPOINT, PREDICT_ENDPOINT
from src.agent.mcp.tools import DOCS_SOURCE_ID, INVENTORY_SOURCE_ID
from src.agent.pipeline import verify_turn_async
from tests.fixtures.cassette import LIVE, RECORD, REPLAY, cassette, resolve_mode
from tests.fixtures.golden_set import CATEGORY_COUNTS, GoldenSetItem, load_golden_set
from tests.fixtures.golden_set_retrieval import (
    RetrievalReport,
    SearchOutcome,
    format_retrieval_section,
    retrieval_as_dict,
    score_retrieval,
    search_outcome,
    summarize_retrieval,
)

MUST_REFUSE = 'Must refuse / "I don\'t know"'

# Section 8's two floors, set before the first run and revisable "only with a stated reason
# recorded in the PR that revises them".
MUST_REFUSE_FLOOR = 1.0
AGGREGATE_FLOOR = 0.9

# See "Cassette vs live" above. This mapping *is* Issue #150's resolution of the tension --
# it is not a tuning knob, and changing a value here changes what the report means.
ATTEMPTS_BY_MODE = {REPLAY: 1, RECORD: 1, LIVE: 3}

CASSETTE_PREFIX = "golden_set__"


def cassette_name(item: GoldenSetItem) -> str:
    """One cassette per item, named after it.

    There is no per-attempt variant, because there is no mode in which one would be read:
    `replay` and `record` run a single attempt, and `live` makes real calls and touches no
    cassette file at all (`cassette.py`'s `LIVE` branch yields a plain client and writes
    nothing). Three recorded attempts of which only the first is ever replayed would put a
    number in the report that no committed artifact reproduces.
    """
    return f"{CASSETTE_PREFIX}{item.item_id}"


# --------------------------------------------------------------------------------------
# Tool-name resolution
# --------------------------------------------------------------------------------------
#
# `AnsweredTurn.tool_payloads` carries every tool result, and each one carries the `source`
# block the tool layer minted -- but **no tool name**. Section 2 mints `source_type` and
# `source_id` from a hard-coded pair per tool and never from an argument
# (`src/agent/mcp/tools.py`), so the pair identifies the tool, and this is the mapping back.
#
# Every value below is imported from the module that owns it rather than restated, the same
# discipline `cassette.current_fingerprint()` and `train_serving_model` follow: a literal
# copied to here could drift from what a tool actually mints and the mapping would then
# resolve confidently to the wrong name.
#
# **`find_similar_historical_pattern` is the one that cannot be an exact match.** Its
# `source_id` is `trajectory_archive@<first 16 hex of the archive manifest's content hash>`,
# or `trajectory_archive@unavailable` when the artifact cannot be read -- one tool, an open
# set of ids. It is matched on the prefix the two share, and
# `test_agent_golden_set_runner.py` pins that both real forms start with it rather than
# trusting this comment.
_ARCHIVE_SOURCE_PREFIX = "trajectory_archive@"

TOOL_BY_SOURCE: dict[tuple[str, str], str] = {
    ("live_endpoint", DRIFT_ENDPOINT): "get_bearing_status",
    ("live_endpoint", PREDICT_ENDPOINT): "predict_health_state",
    ("live_endpoint", DOCS_SOURCE_ID): "search_documentation",
    ("inventory", INVENTORY_SOURCE_ID): "check_inventory",
}

TOOL_BY_SOURCE_PREFIX: dict[tuple[str, str], str] = {
    ("trajectory_match", _ARCHIVE_SOURCE_PREFIX): "find_similar_historical_pattern",
}


class ToolResolutionError(ValueError):
    """A tool payload whose `source` block matches no known tool.

    Raised rather than skipped. An unresolved payload means a tool was called and the runner
    cannot say which -- and `expected_tool_names` is scored by set equality, so silently
    dropping it would turn "called a tool it should not have" into a pass.
    """


def resolve_tool_name(source: Mapping[str, Any]) -> str:
    """The tool that minted one `source` block."""
    source_type = str(source.get("source_type", ""))
    source_id = str(source.get("source_id", ""))
    exact = TOOL_BY_SOURCE.get((source_type, source_id))
    if exact is not None:
        return exact
    for (prefix_type, prefix), name in TOOL_BY_SOURCE_PREFIX.items():
        if source_type == prefix_type and source_id.startswith(prefix):
            return name
    raise ToolResolutionError(
        f"no read-only tool mints source_type={source_type!r} source_id={source_id!r}. "
        "Known: "
        + ", ".join(
            f"{name} ({stype}, {sid})" for (stype, sid), name in sorted(TOOL_BY_SOURCE.items())
        )
        + ", "
        + ", ".join(
            f"{name} ({stype}, {prefix}...)"
            for (stype, prefix), name in sorted(TOOL_BY_SOURCE_PREFIX.items())
        )
    )


def resolve_tool_names(payloads: Iterable[Mapping[str, Any]]) -> frozenset[str]:
    """Every tool this turn called, as names, from the payloads it produced.

    A set, because `expected_tool_names` is scored by set equality on names -- Section 8:
    "not on exact arguments, arguments legitimately vary". Two calls to one tool are one
    name, and a failed call still counts as a call: `results.failed` carries the same minted
    `source` block a success does, which is what makes "it tried and the service was down"
    distinguishable from "it never asked".
    """
    return frozenset(resolve_tool_name(payload.get("source") or {}) for payload in payloads)


def mapped_tool_names() -> frozenset[str]:
    """Every tool name this mapping can produce."""
    return frozenset(TOOL_BY_SOURCE.values()) | frozenset(TOOL_BY_SOURCE_PREFIX.values())


def unmapped_readonly_tools() -> frozenset[str]:
    """Read-only tools with no entry here -- empty is the only acceptable value.

    Checked by a test rather than asserted at import: a sixth read-only tool added later
    without a mapping entry must fail loudly in CI, not resolve to nothing at scoring time.
    """
    return frozenset(READONLY_TOOL_NAMES) - mapped_tool_names()


# --------------------------------------------------------------------------------------
# Citation matching
# --------------------------------------------------------------------------------------


def citation_matches_allowed(cited: str, allowed: frozenset[str]) -> bool:
    """Whether one cited id is covered by an item's `allowed_source_ids`.

    Exact match, **or** the cited id is a chunk of an allowed document. The second half is
    not a loosening of the rule, it is what makes the rule applicable at all: a corpus item
    declares `allowed_source_ids={"docs/eda_findings.md"}` (the document), while a real
    citation is a chunk id, and `src/agent/rag/schema.py` defines that as
    `f"{source_id}::{chunk_index}"`. Comparing the two by equality would fail all 8
    "Answerable from the docs" items by construction, on correct answers.

    The split is on the **last** `::`, and only the document half is compared, so
    `docs/eda_findings.md::4` matches `docs/eda_findings.md` and nothing else does. A tool
    result's own id (`GET /monitoring/drift`) carries no `::` and is matched exactly.
    """
    if cited in allowed:
        return True
    document, separator, _index = cited.rpartition("::")
    return bool(separator) and document in allowed


def cited_source_ids(response: GroundedResponse) -> tuple[str, ...]:
    """Every id the released claims cite, in order, without repeats.

    Read off the **released** claims rather than the draft: a claim the critic dropped is
    not part of the answer, and scoring an answer for a citation it does not make would be
    scoring the draft instead of what a person would be shown.
    """
    seen: list[str] = []
    for claim in response.claims:
        for source_id in claim.source_ids:
            if source_id not in seen:
                seen.append(source_id)
    return tuple(seen)


# --------------------------------------------------------------------------------------
# One run of one item
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ItemRun:
    """What one pass of one item through the pipeline produced.

    `error` is set when the run could not be made at all -- a missing cassette, an
    unreachable API. It is kept separate from a failing score because the two mean opposite
    things to a reader: a failed item is a result, an errored item is an absence of one.

    `search` is Issue #152's addition: the ranked `(chunk_id, score)` list this turn's
    `search_documentation` calls returned, read off the same payloads `tool_names` is read
    off and changing nothing about that reading. `None` -- the default -- means *no
    observation*, not *nothing retrieved*: an `ItemRun` built by hand, or an errored run that
    never reached a tool. `SearchOutcome(searched=False)` is the observed-and-never-searched
    case, and the two are kept apart because a retrieval metric computed from an absent
    observation would be a number about the harness wearing a model's clothes. No sub-score
    below reads this field.
    """

    item_id: str
    tool_names: frozenset[str] = frozenset()
    cited: tuple[str, ...] = ()
    text: str = ""
    grounding_tier: str = ""
    has_recommendation: bool = False
    n_claims: int = 0
    error: str | None = None
    search: SearchOutcome | None = None

    @classmethod
    def from_turn(
        cls, item: GoldenSetItem, turn: AnsweredTurn, response: GroundedResponse
    ) -> "ItemRun":
        return cls(
            item_id=item.item_id,
            tool_names=resolve_tool_names(turn.tool_payloads),
            cited=cited_source_ids(response),
            text=response.text,
            grounding_tier=response.grounding_tier,
            has_recommendation=response.recommendation is not None,
            n_claims=len(response.claims),
            search=search_outcome(turn.tool_payloads),
        )


async def run_item_async(
    item: GoldenSetItem,
    *,
    client: Any,
    serving_url: str | None = None,
    db_path: Path | None = None,
) -> ItemRun:
    """One item, once, through the real pipeline with a supplied model client."""
    turn = await answer_turn_async(
        item.question, client=client, serving_url=serving_url, db_path=db_path
    )
    response = await verify_turn_async(turn, critic_client=client)
    return ItemRun.from_turn(item, turn, response)


def run_item(
    item: GoldenSetItem,
    *,
    serving_url: str | None = None,
    db_path: Path | None = None,
    mode: str | None = None,
) -> ItemRun:
    """One item, once, sourcing its model calls from this mode's client.

    One `asyncio.run` per call, and the client is opened inside it: an httpx connection pool
    belongs to the loop it was opened on, which is the same constraint
    `test_agent_pipeline_live.py` records for its two-step form.
    """
    resolved = resolve_mode(mode)

    async def _run() -> ItemRun:
        with cassette(cassette_name(item), mode=resolved) as client:
            return await run_item_async(
                item, client=client, serving_url=serving_url, db_path=db_path
            )

    try:
        return asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 -- an errored item is reported, never fatal
        return ItemRun(item_id=item.item_id, error=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------------------
# Scoring one item
# --------------------------------------------------------------------------------------

CORRECT_TOOL_CALL = "correct_tool_call"
GROUNDED_ANSWER = "source_grounded_answer"
CORRECT_REFUSAL = "correct_refusal"


@dataclass(frozen=True)
class SubScore:
    """One of Section 8's three binary sub-scores, or its explicit non-applicability.

    `applicable=False` is a third state and deliberately not a pass: a must-refuse item has
    no source-grounded answer to score, and counting that as a passed sub-score would let a
    category's numbers be padded by checks that never ran.
    """

    name: str
    applicable: bool
    passed: bool
    reasons: tuple[str, ...] = ()

    @property
    def failed(self) -> bool:
        return self.applicable and not self.passed


@dataclass(frozen=True)
class ItemScore:
    """One item's verdict: the sub-scores, and whether every applicable one passed."""

    item_id: str
    category: str
    sub_scores: tuple[SubScore, ...]
    run: ItemRun

    @property
    def passed(self) -> bool:
        if self.run.error is not None:
            return False
        return not any(sub.failed for sub in self.sub_scores)

    @property
    def reasons(self) -> tuple[str, ...]:
        if self.run.error is not None:
            return (f"the run did not complete: {self.run.error}",)
        return tuple(reason for sub in self.sub_scores for reason in sub.reasons)


def _substring_reasons(item: GoldenSetItem, text: str) -> list[str]:
    """Required substrings absent, and forbidden ones present, in the released text."""
    reasons = []
    missing = [needle for needle in item.required_substrings if needle not in text]
    if missing:
        reasons.append(f"required substring(s) absent from the answer: {missing}")
    present = [needle for needle in item.forbidden_substrings if needle in text]
    if present:
        reasons.append(f"forbidden substring(s) present in the answer: {present}")
    return reasons


def _score_tool_call(item: GoldenSetItem, run: ItemRun) -> SubScore:
    """Set equality on names, never subset (Section 8).

    Subset would pass an answer that called the right tool *and* three others, which is the
    behaviour `tools-live-no-raw-signal-to-score` exists to catch: reaching for
    `predict_health_state` without a signal is the failure, and it is only visible as an
    extra name in the set.
    """
    reasons = []
    if run.tool_names != item.expected_tool_names:
        missing = sorted(item.expected_tool_names - run.tool_names)
        extra = sorted(run.tool_names - item.expected_tool_names)
        detail = []
        if missing:
            detail.append(f"did not call {missing}")
        if extra:
            detail.append(f"also called {extra}")
        reasons.append("tool calls were not the expected set: " + "; ".join(detail))
    return SubScore(CORRECT_TOOL_CALL, True, not reasons, tuple(reasons))


def _score_grounded_answer(item: GoldenSetItem, run: ItemRun) -> SubScore:
    """Cited ids within the allowed set, at least one citation, and the substring checks.

    Both citation rules are applied, and they are different rules from two places that a
    single check cannot express: Issue #150 states the **subset** rule (nothing cited outside
    the allowed set) and Section 8 states the **at-least-one** rule ("an allowed set of
    `source_id`s at least one of which must be cited"). Subset alone would pass an answer
    citing nothing at all, which is vacuous for an item whose whole subject is being sourced.
    """
    reasons = []
    outside = [
        cited
        for cited in run.cited
        if not citation_matches_allowed(cited, item.allowed_source_ids)
    ]
    if outside:
        reasons.append(
            f"cited {outside}, which is outside allowed_source_ids "
            f"{sorted(item.allowed_source_ids)}"
        )
    if not run.cited:
        reasons.append("released no cited claim, so nothing was sourced from the allowed set")
    reasons.extend(_substring_reasons(item, run.text))
    return SubScore(GROUNDED_ANSWER, True, not reasons, tuple(reasons))


def _score_refusal(item: GoldenSetItem, run: ItemRun) -> SubScore:
    """The structural shape of a correct refusal, plus the substring checks.

    Section 6's tier table makes tier 3 the shape of a refusal: no claim survives
    verification, so `claims` is empty and the recommendation is withheld. All three are
    checked rather than the tier alone -- `grounding.assemble` guarantees the other two
    today, and a check that only reads the label would not notice if it stopped.
    """
    reasons = []
    if run.grounding_tier != UNGROUNDED:
        reasons.append(
            f"grounding_tier was {run.grounding_tier!r}, not {UNGROUNDED!r}: the answer was "
            "released rather than refused"
        )
    if run.n_claims:
        reasons.append(f"released {run.n_claims} claim(s); a refusal releases none")
    if run.has_recommendation:
        reasons.append("carried a recommendation; a refused answer withholds it")
    reasons.extend(_substring_reasons(item, run.text))
    return SubScore(CORRECT_REFUSAL, True, not reasons, tuple(reasons))


def score_item(item: GoldenSetItem, run: ItemRun) -> ItemScore:
    """Section 8's three sub-scores, with exactly one of the two answer-shaped ones applying.

    A must-refuse item is scored on the refusal shape *instead of* on source grounding
    (Issue #150), because it has no grounded answer to score: its correct outcome is that
    nothing cleared the threshold, so `allowed_source_ids` is empty and citing anything at
    all would be the failure. The tool-call sub-score applies to every item in both
    categories -- searching and *then* refusing is the correct trajectory, and refusing
    without looking is not.
    """
    is_refusal = item.category == MUST_REFUSE
    if run.error is not None:
        # No answer to score. The sub-scores are recorded as not applicable rather than as
        # failures so a reader can tell an errored run from a wrong one; `ItemScore.passed`
        # is False either way.
        return ItemScore(
            item_id=item.item_id,
            category=item.category,
            sub_scores=(
                SubScore(CORRECT_TOOL_CALL, False, False),
                SubScore(CORRECT_REFUSAL if is_refusal else GROUNDED_ANSWER, False, False),
            ),
            run=run,
        )
    return ItemScore(
        item_id=item.item_id,
        category=item.category,
        sub_scores=(
            _score_tool_call(item, run),
            _score_refusal(item, run) if is_refusal else _score_grounded_answer(item, run),
        ),
        run=run,
    )


def run_and_score(
    item: GoldenSetItem,
    *,
    serving_url: str | None = None,
    db_path: Path | None = None,
    mode: str | None = None,
) -> ItemScore:
    """Run one item for this mode's number of attempts; it passes only if all of them do.

    Section 8's "non-determinism is a failure, not noise", applied where it can mean
    something. In `replay` that is one attempt (see the module docstring's table): the score
    is then a statement about the harness, not about model stability, and `format_report`
    prints the mode so no reader can mistake one for the other. The **first failing**
    attempt is the one reported, because it is the evidence for the verdict.
    """
    attempts = ATTEMPTS_BY_MODE[resolve_mode(mode)]
    scored: ItemScore | None = None
    for _attempt in range(attempts):
        scored = score_item(
            item, run_item(item, serving_url=serving_url, db_path=db_path, mode=mode)
        )
        if not scored.passed:
            return scored
    assert scored is not None  # attempts >= 1 for every mode
    return scored


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CategoryResult:
    """One category's own numbers. Always reported individually (Section 8)."""

    category: str
    passed: int
    total: int

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


@dataclass(frozen=True)
class GoldenSetReport:
    """The two gates, side by side, never combined.

    There is deliberately **no overall pass rate on this object**. Section 8's rule is that
    the safety-relevant category is gated at 100% on its own and the other 22 items carry a
    separate >= 90% floor, "never only as the aggregate" -- a single blended number is the
    aggregate-hides-the-subgroup failure `docs/evaluation_protocol.md` §5 forbids, and the
    cheapest way to reintroduce it is to offer one here for convenience.
    """

    mode: str
    attempts: int
    categories: tuple[CategoryResult, ...]
    scores: tuple[ItemScore, ...] = ()
    # Issue #152's additive field. Section 8's *second* number, carried next to the two gates
    # and read by no property below: `must_refuse_gate_passed`, `remaining_gate_passed` and
    # `gates_passed` are byte-for-byte the ones part 3a shipped, and `main`'s exit code still
    # follows `gates_passed` alone. `None` when nobody supplied the items to score retrieval
    # against (see `summarize`).
    retrieval: RetrievalReport | None = None

    @property
    def must_refuse(self) -> CategoryResult:
        for category in self.categories:
            if category.category == MUST_REFUSE:
                return category
        return CategoryResult(MUST_REFUSE, 0, 0)

    @property
    def remaining_passed(self) -> int:
        return sum(c.passed for c in self.categories if c.category != MUST_REFUSE)

    @property
    def remaining_total(self) -> int:
        return sum(c.total for c in self.categories if c.category != MUST_REFUSE)

    @property
    def remaining_rate(self) -> float:
        return self.remaining_passed / self.remaining_total if self.remaining_total else 0.0

    @property
    def must_refuse_gate_passed(self) -> bool:
        """100%, on its own, with no aggregate. An empty category does not pass vacuously."""
        return self.must_refuse.total > 0 and self.must_refuse.rate >= MUST_REFUSE_FLOOR

    @property
    def remaining_gate_passed(self) -> bool:
        return self.remaining_total > 0 and self.remaining_rate >= AGGREGATE_FLOOR

    @property
    def gates_passed(self) -> bool:
        """Both gates, as a conjunction of two separately-reported verdicts -- not a third,
        blended number. `format_report` prints the two either way."""
        return self.must_refuse_gate_passed and self.remaining_gate_passed

    @property
    def errored(self) -> tuple[ItemScore, ...]:
        return tuple(score for score in self.scores if score.run.error is not None)

    @property
    def is_full_set(self) -> bool:
        """Whether every category was run at its Section 8 count.

        False for a `--category`-filtered run, and `format_report` says so on its face: two
        gates printed over 4 of 30 items look exactly like two gates printed over 30, and
        the difference is the whole meaning of the verdict.
        """
        return {c.category: c.total for c in self.categories} == CATEGORY_COUNTS


def retrieval_report(
    scores: Sequence[ItemScore], items: Sequence[GoldenSetItem] | None = None
) -> RetrievalReport:
    """Issue #152's retrieval reading for a set of scored items.

    The join is by `item_id`, because an `ItemScore` carries the id and not the item, and
    `relevant_chunk_ids` lives on the item. `items` defaults to the real golden set; a score
    whose `item_id` is not in it contributes **no** reading rather than an invented one --
    the same rule as a tool-grounded item, applied to a score the golden set does not
    recognise.

    Nothing here can change a verdict: it reads `ItemScore.passed` as one axis of the 2x2 and
    writes nothing back.
    """
    known = items if items is not None else load_golden_set()
    by_id = {item.item_id: item for item in known}
    qualities = [
        score_retrieval(
            by_id[score.item_id],
            score.run.search,
            must_refuse_category=MUST_REFUSE,
            answer_correct=None if score.run.error is not None else score.passed,
        )
        for score in scores
        if score.item_id in by_id
    ]
    return summarize_retrieval(qualities)


def summarize(
    scores: Sequence[ItemScore],
    *,
    mode: str | None = None,
    items: Sequence[GoldenSetItem] | None = None,
) -> GoldenSetReport:
    """Per-category counts in Section 8's own table order, the two gates, and -- separately --
    Issue #152's retrieval reading.

    `items` is only ever used for the retrieval half (it is where `relevant_chunk_ids` live);
    the categories and both gates are computed from `scores` exactly as part 3a computed
    them, whether it is supplied or not.
    """
    resolved = resolve_mode(mode)
    by_category: dict[str, list[ItemScore]] = {category: [] for category in CATEGORY_COUNTS}
    for score in scores:
        by_category.setdefault(score.category, []).append(score)
    return GoldenSetReport(
        mode=resolved,
        attempts=ATTEMPTS_BY_MODE[resolved],
        categories=tuple(
            CategoryResult(
                category=category,
                passed=sum(1 for score in scored_items if score.passed),
                total=len(scored_items),
            )
            for category, scored_items in by_category.items()
        ),
        scores=tuple(scores),
        retrieval=retrieval_report(scores, items),
    )


def format_report(report: GoldenSetReport) -> str:
    """The report a reviewer reads. Two verdicts, per-category counts, and the failures.

    The mode line is first and is not decoration: a `replay` report is evidence that the
    harness and the gate work, and a `live` one is evidence about the model. Printing the
    numbers without the mode would make those two look like the same claim.
    """
    lines = [
        "docs/agent_design.md Section 8, tier 2 -- golden set",
        f"mode: {report.mode} ({report.attempts} attempt(s) per item"
        + (", all must pass)" if report.attempts > 1 else ")"),
    ]
    if report.attempts == 1:
        lines.append(
            "  NOTE: a replayed cassette returns one fixed recorded answer, so this run "
            "measures the harness and the gate, not model stability. Section 8's 3x rule "
            "needs --mode live."
        )
    if not report.is_full_set:
        lines.append(
            f"  PARTIAL RUN: {sum(c.total for c in report.categories)} of "
            f"{sum(CATEGORY_COUNTS.values())} items. The gates below are not a verdict on "
            "the golden set."
        )
    lines += ["", "per category (always reported individually, never only as an aggregate):"]
    for category in report.categories:
        lines.append(f"  {category.passed:>2}/{category.total:<2}  {category.category}")

    must_refuse = report.must_refuse
    lines += [
        "",
        "gate 1 -- must refuse, 100% required, scored on its own:",
        f"  {must_refuse.passed}/{must_refuse.total} "
        f"({must_refuse.rate:.0%})  {'PASS' if report.must_refuse_gate_passed else 'FAIL'}",
        "",
        f"gate 2 -- the remaining {report.remaining_total} items, "
        f">= {AGGREGATE_FLOOR:.0%} required:",
        f"  {report.remaining_passed}/{report.remaining_total} "
        f"({report.remaining_rate:.0%})  {'PASS' if report.remaining_gate_passed else 'FAIL'}",
    ]

    if report.retrieval is not None:
        lines += format_retrieval_section(report.retrieval)

    failures = [score for score in report.scores if not score.passed]
    if failures:
        lines += ["", "failures:"]
        for score in failures:
            lines.append(f"  {score.item_id} [{score.category}]")
            for reason in score.reasons:
                lines.append(f"    - {reason}")
    if report.errored:
        lines += [
            "",
            f"{len(report.errored)} item(s) did not run at all. A missing cassette is the "
            "usual cause: record one per item with --mode record (real, billed calls).",
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


def run_golden_set(
    items: Sequence[GoldenSetItem] | None = None,
    *,
    serving_url: str | None = None,
    db_path: Path | None = None,
    mode: str | None = None,
) -> GoldenSetReport:
    """Run and score every item, then summarize. The whole tier in one call."""
    items = list(items) if items is not None else list(load_golden_set())
    scores = [
        run_and_score(item, serving_url=serving_url, db_path=db_path, mode=mode) for item in items
    ]
    return summarize(scores, mode=mode, items=items)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--url", default=None, help="base URL of an already-running serving API"
    )
    parser.add_argument("--db-path", type=Path, default=None, help="inventory database path")
    parser.add_argument(
        "--mode",
        default=None,
        choices=[REPLAY, RECORD, LIVE],
        help=(
            "replay (default): one attempt per item, offline and free. live: three attempts "
            "per item, all must pass -- real, billed API calls. record: one real call per "
            "item, overwriting its committed cassette."
        ),
    )
    parser.add_argument(
        "--category", action="append", default=None, help="run only this category (repeatable)"
    )
    parser.add_argument("--json", action="store_true", help="also print the report as JSON")
    return parser.parse_args(argv)


def report_as_dict(report: GoldenSetReport) -> dict[str, Any]:
    """The report as data, for a PR comment or a later trace. Same two-gate shape.

    Issue #152 adds one key, `retrieval_quality`, and adds it **beside** the two gates rather
    than inside either: no existing key's value changes, and no number below is derived from
    a retrieval metric. The key is absent (rather than `null`) when no retrieval reading was
    produced, so a consumer cannot mistake "not measured" for "measured as zero".
    """
    payload: dict[str, Any] = {
        "mode": report.mode,
        "attempts_per_item": report.attempts,
        "categories": [
            {"category": c.category, "passed": c.passed, "total": c.total, "rate": c.rate}
            for c in report.categories
        ],
        "must_refuse_gate": {
            "passed": report.must_refuse.passed,
            "total": report.must_refuse.total,
            "floor": MUST_REFUSE_FLOOR,
            "gate_passed": report.must_refuse_gate_passed,
        },
        "remaining_gate": {
            "passed": report.remaining_passed,
            "total": report.remaining_total,
            "floor": AGGREGATE_FLOOR,
            "gate_passed": report.remaining_gate_passed,
        },
        "failures": [
            {"item_id": s.item_id, "category": s.category, "reasons": list(s.reasons)}
            for s in report.scores
            if not s.passed
        ],
    }
    if report.retrieval is not None:
        payload["retrieval_quality"] = retrieval_as_dict(report.retrieval)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    items = list(load_golden_set())
    if args.category:
        wanted = set(args.category)
        items = [item for item in items if item.category in wanted]
    report = run_golden_set(
        items, serving_url=args.url, db_path=args.db_path, mode=args.mode
    )
    print(format_report(report))
    if args.json:
        print()
        print(json.dumps(report_as_dict(report), indent=2))
    return 0 if report.gates_passed else 1


if __name__ == "__main__":
    sys.exit(main())
