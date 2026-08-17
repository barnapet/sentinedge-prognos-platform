"""Calibrate `LEXICAL_OVERLAP_FLOOR` against the golden set's prose-citing items (Issue #173).

    python -m tests.fixtures.calibrate_overlap_floor --url http://localhost:8000
    python -m tests.fixtures.calibrate_overlap_floor --measurements <saved.json>

`src/agent/critic/escalation.py`'s own comment calls the floor "a starting value, still
uncalibrated", and `docs/agent_design.md` Section 6 sends it to Section 8's golden set to be
measured -- the same standing `TAU_TOP`/`TAU_SUPPORT` had before Issues #163/#165 swept and
applied them. This script is that sweep, for that one constant.

**What it is not.** `lexical_overlap` is evaluated for claims citing prose sources only
(`PROSE_SOURCE_TYPES`, Issue #119), so this floor moves nothing about a claim citing only
`live_endpoint`/`inventory` results -- that case has no overlap check at all by design, and
Issue #171's concept-domain check (`src/agent/critic/relevance.py`) is what addresses it. The
sweep below is therefore about a narrower thing than "the golden set's pass rate": it is about
claims grounded in real retrieved chunks.

## This one cannot be run with zero Anthropic calls, and the reason is worth recording

Issue #163's sweep cost nothing because a retrieval threshold is applied to *retrieval
scores*, which `search()` produces without a model. This one is different in both halves:

- **The claim side of every claim/chunk pair is model output.** `lexical_overlap(claim_text,
  chunk_text)` cannot be evaluated against the golden set without drafts, and a draft is what
  the answerer writes. There is no offline source of one in this repo that would do: the
  committed `tests/fixtures/answerer_turn.json` is marked
  `"synthesized_from_recorded_payloads"` (its claims are hand-written, citing real ids), and
  all four committed cassettes were recorded with the documentation index deliberately down
  (Section 8's note), so every recorded claim in them cites `GET /monitoring/drift` and not one
  cites a prose chunk. The 30 golden-set recordings that *would* have supplied real drafts were
  deleted on purpose by #161/#162, and re-creating them is exactly what that revert refused.
- **Whether an escalation changes an item's verdict is a model judgement.** Raising the floor
  escalates a claim; it is the critic's `no`/`unclear` that demotes it, and for a must-refuse
  item that demotion is the entire mechanism by which a drafted answer becomes the tier-3
  refusal the item is scored on (`grounding.py`'s `_tier`). A sweep that assumed the verdict
  would be measuring an assumption.

So the collection phase makes **real, billed calls**: one answerer turn per swept item, plus
one critic call per distinct escalated claim/chunk pair. What keeps that small is that the
verdicts are **measured once and swept against many times** -- the same discipline
`calibrate_retrieval.ItemScores` records. A pair's verdict depends on the claim and the chunk,
not on the floor, so the union of pairs escalated *anywhere* in the swept range is collected
once and every one of the 12 floors is then scored from that cache with no further calls. The
critic cost of a full sweep is therefore the count at the widest floor, not the sum over the
grid, and `--measurements` re-runs the whole sweep from a saved collection for free.

## It changes no production constant

Issue #173 recommends a value with evidence; editing `LEXICAL_OVERLAP_FLOOR` is a separate,
reviewed follow-on, matching how #163 (recommend) and #165 (apply) were split. The current
value is imported and printed beside the recommendation purely so the two can be compared.

Nothing here reimplements the mechanism it is measuring, for the reason #163 gives: a
reimplementation could agree with itself while disagreeing with what actually gates a turn.
`escalations_needed` already takes `floor` as a keyword argument, so **that** is the swept
knob; the verdict-to-demotion mapping is production's own `escalate_async` driven by a replay
client; the response is production's `assemble`; and the pass/fail is
`golden_set_runner.score_item`, Section 8's scoring, unmodified. `tests/
test_agent_calibrate_overlap_floor.py` pins that a swept response at the current floor is
identical to what `pipeline.verify_turn_async` produces on the same turn.

## What it needs

The fixture the golden-set runner documents, since it runs the real answerer:

    docker compose --profile agent up -d qdrant
    python -m src.agent.rag.index
    python -m src.serving.main
    python -m demo.playback --interval 0
    python -m src.agent.inventory.build_db

plus Anthropic credentials. `--measurements PATH` replays a previously saved collection
instead, which needs none of it.

## The hard constraint, and why it is not a term to trade against

**No swept floor may leave fewer must-refuse items passing than the current 0.6 floor leaves
today**, measured in the same run, with zero tolerance. That is Section 8's gate 1 ("every one
of the 8 must-refuse items must pass individually. 100%, no aggregate") applied to a
calibration: a floor that buys answerable pass rate by letting one out-of-corpus question
through is not a better floor, and averaging that away is the failure
`docs/evaluation_protocol.md` §5 forbids. So it is a filter applied before pass rate is looked
at, not a penalty term.

The baseline row is measured, not assumed, and it is measured from the same drafts and the same
verdicts as every other row -- which is also why a recommendation always exists: the current
floor satisfies its own constraint, so "keep 0.6" is a possible answer to this sweep and it is
reported as one when it wins.

## Why it lives under `tests/`

Issue #122's constraint, the same one `calibrate_retrieval.py` and `golden_set_runner.py`
record: the golden set is test infrastructure, and no module under `src/agent/` may know
`tests/fixtures/golden_set_corpus.py` exists. The dependency runs one way only.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.agent.answerer import AnsweredTurn, answer_turn_async
from src.agent.critic.deterministic import DeterministicReport, TurnEvidence, verify
from src.agent.critic.escalation import (
    LEXICAL_OVERLAP_FLOOR,
    PROSE_SOURCE_TYPES,
    EscalationRequest,
    escalate_async,
    escalations_needed,
    judge_async,
    lexical_overlap,
)
from src.agent.critic.grounding import GroundedResponse, assemble
from src.agent.critic.relevance import off_domain_demotions
from src.agent.critic.retrieval_confidence import assess_evidence
from src.agent.inventory.build_db import DB_PATH
from src.agent.mcp.serving_client import DEFAULT_BASE_URL
from src.agent.untrusted import escape_payload
from tests.fixtures.golden_set import GoldenSetItem
from tests.fixtures.golden_set_corpus import CORPUS_ITEMS
from tests.fixtures.golden_set_runner import MUST_REFUSE, ItemRun, ItemScore, score_item

# The swept grid: Issue #173's range and step, 0.40 to 0.95 inclusive by 0.05. Coarser than
# #163's `tau_top` grid on purpose -- this is a *rate* over a claim's handful of content words,
# not a similarity. A one-sentence claim carries roughly 5-15 content terms, so the attainable
# overlaps are a small set of fractions and a 0.01 grid would print 56 rows describing the same
# handful of distinct decisions.
FLOOR_CANDIDATES: tuple[float, ...] = tuple(round(0.40 + 0.05 * step, 4) for step in range(12))

# The floor whose behaviour every row is measured against (the hard constraint above). Imported,
# never restated: the constraint is "no worse than what production does today", and a literal
# copied to here would keep describing 0.6 after the follow-on issue moves it.
CURRENT_FLOOR = LEXICAL_OVERLAP_FLOOR

ANSWERABLE = "Answerable from the docs"


# --------------------------------------------------------------------------------------
# One measured turn: the real draft, the real evidence, and the verdicts for its pairs
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MeasuredTurn:
    """One item's turn, measured once, swept against every floor.

    `verdicts` maps `(claim_index, source_id)` -- the identity of one escalated claim/chunk
    pair -- to the critic's own `yes`/`no`/`unclear`. Keyed by the pair rather than by the
    floor, because that is what the verdict actually depends on: the same pair escalated at
    0.55 and at 0.95 is the same question, and asking it twice would make one measurement look
    like two.

    `report` and `evidence` are stored rather than recomputed per floor for the same reason
    `calibrate_retrieval` keeps its scores: the deterministic pass does not read the floor, so
    re-running it 12 times would produce 12 identical reports and hide that the sweep moves
    exactly one thing.
    """

    item: GoldenSetItem
    turn: AnsweredTurn
    evidence: TurnEvidence
    report: DeterministicReport
    verdicts: Mapping[tuple[int, str], str] = field(default_factory=dict)

    @property
    def item_id(self) -> str:
        return self.item.item_id

    @property
    def category(self) -> str:
        return self.item.category

    @property
    def is_must_refuse(self) -> bool:
        return self.category == MUST_REFUSE

    def prose_pairs(self) -> tuple[tuple[int, str, float], ...]:
        """Every verified claim's best prose pair: `(claim_index, source_id, overlap)`.

        This is the population the floor is a threshold *on*, so it is computed with the same
        two rules `escalations_needed` uses and no others: only claims that survived the
        deterministic pass (`report.verified`), and only cited sources whose `source_type` is in
        `PROSE_SOURCE_TYPES`. A claim citing no prose contributes nothing -- not a zero.
        """
        pairs: list[tuple[int, str, float]] = []
        for checked in self.report.verified:
            best: tuple[str, float] | None = None
            for source_id in checked.claim.source_ids:
                for item in self.evidence.items:
                    if item.source_id != source_id or item.source_type not in PROSE_SOURCE_TYPES:
                        continue
                    overlap = lexical_overlap(checked.claim.text, item.text)
                    if best is None or overlap > best[1]:
                        best = (source_id, overlap)
            if best is not None:
                pairs.append((checked.index, best[0], best[1]))
        return tuple(pairs)

    @property
    def cites_prose(self) -> bool:
        """Whether the floor can reach this item's turn at all."""
        return bool(self.prose_pairs())

    def requests_at(self, floor: float) -> tuple[EscalationRequest, ...]:
        """Production's escalation rule at one candidate floor. No model call."""
        return escalations_needed(self.turn.draft, self.report, self.evidence, floor=floor)

    def requests_over(self, floors: Sequence[float]) -> tuple[EscalationRequest, ...]:
        """Every distinct pair escalated anywhere in `floors`, in first-seen order.

        Not simply the set at the widest floor: `escalations_needed` falls back to trigger (a)
        for a claim whose prose overlap clears the floor, and trigger (a) chooses the best
        cited source of *any* type -- so a claim can be put to the critic against one passage
        at 0.50 and a different one at 0.95. Collecting the union means every row of the sweep
        is scored from a real verdict rather than from the nearest one.
        """
        seen: dict[tuple[int, str], EscalationRequest] = {}
        for floor in floors:
            for request in self.requests_at(floor):
                seen.setdefault((request.claim_index, request.source_id), request)
        return tuple(seen.values())

    def response_at(self, floor: float) -> GroundedResponse:
        """The response production would assemble for this turn at this floor.

        Every step is production's: `escalations_needed` decides what escalates,
        `escalate_async` turns the recorded verdicts into demotion reasons (`no` and `unclear`
        both demote, the reason string written once in `grounding.py`), `off_domain_demotions`
        contributes Issue #171's deterministic check, and `assemble` produces the tier. The
        merge of the two demotion sources is `pipeline.verify_turn_async`'s, and a test asserts
        this method agrees with it claim-for-claim at `CURRENT_FLOOR` rather than leaving that
        to inspection.
        """
        requests = self.requests_at(floor)
        demotions: dict[int, str] = dict(off_domain_demotions(self.report, self.evidence))
        if requests:
            replay = _VerdictReplay(self, requests)
            demoted = asyncio.run(escalate_async(requests, client=replay))
            for index, reason in demoted.items():
                already = demotions.get(index)
                demotions[index] = f"{already}; {reason}" if already else reason
        return assemble(
            self.turn.draft,
            self.evidence,
            report=self.report,
            retrieval=assess_evidence(self.evidence),
            demotions=demotions,
        )

    def score_at(self, floor: float) -> ItemScore:
        """Section 8's own scoring of this item at this floor, via `score_item`."""
        response = self.response_at(floor)
        return score_item(self.item, ItemRun.from_turn(self.item, self.turn, response))


class VerdictMissing(KeyError):
    """A pair the sweep needs a verdict for that the collection does not carry.

    Raised rather than defaulted. A missing verdict means the sweep reached a claim/chunk pair
    nobody asked the critic about, and inventing `yes` or `no` for it would put a number nobody
    measured into the table -- the same refusal `escalation.parse_verdict` makes when the
    response it is handed carries no verdict to parse.
    """


class _VerdictReplay:
    """An `AsyncAnthropic` stand-in that answers from the measured verdicts.

    It exists so the sweep runs production's `escalate_async` -- the `no`/`unclear` rule and the
    demotion-reason text written once in `grounding.py` -- instead of a copy of it, while making
    no call.

    **A pair is identified by what the request actually carries**, not by call order and not by
    an envelope this class supplies: the `source_id` attribute (emitted verbatim, outside the
    envelope) plus the claim text as `escape_payload` renders it inside one. Order-based replay
    would silently answer the wrong question if `escalations_needed` ever reordered its output,
    and keying on the whole rendered string would only work for an envelope built here -- the
    nonce is random per envelope by construction, so `pipeline.verify_turn_async`'s own critic
    call could not be replayed against it. This way it can, which is what lets a test assert
    that the sweep and the pipeline agree at the current floor.
    """

    def __init__(self, measured: MeasuredTurn, requests: Sequence[EscalationRequest]) -> None:
        self._table: list[tuple[str, str, str]] = []
        for request in requests:
            key = (request.claim_index, request.source_id)
            if key not in measured.verdicts:
                raise VerdictMissing(
                    f"no recorded verdict for claim {request.claim_index} against "
                    f"{request.source_id} on item {measured.item_id}; the collection phase "
                    "did not ask about this pair"
                )
            self._table.append(
                (
                    f'source_id="{request.source_id}"',
                    escape_payload(request.claim_text),
                    measured.verdicts[key],
                )
            )
        self.messages = self

    async def create(self, **kwargs: Any) -> Any:
        content = kwargs["messages"][0]["content"]
        matched = {
            verdict
            for source_attribute, claim, verdict in self._table
            if source_attribute in content and claim in content
        }
        if not matched:
            raise VerdictMissing(
                "the critic was asked about a claim/chunk pair with no recorded verdict"
            )
        if len(matched) > 1:
            # Two pairs whose claim text *and* cited source id both match one request, with
            # different verdicts. Raised rather than resolved by picking one: the identity above
            # is meant to be unique, and quietly answering with either would put a verdict the
            # critic gave about a different pair into the table.
            raise VerdictMissing(
                f"ambiguous recorded verdicts {sorted(matched)} for one escalation"
            )
        return _ReplayedMessage(matched.pop())


@dataclass(frozen=True)
class _TextBlock:
    text: str
    type: str = "text"


@dataclass(frozen=True)
class _ReplayedMessage:
    """The one shape `escalation.parse_verdict` reads: text blocks carrying the JSON verdict."""

    verdict: str

    @property
    def content(self) -> tuple[_TextBlock, ...]:
        return (_TextBlock(json.dumps({"verdict": self.verdict})),)


# --------------------------------------------------------------------------------------
# Collection -- the only part that calls the API
# --------------------------------------------------------------------------------------


async def measure_item_async(
    item: GoldenSetItem,
    *,
    client: Any,
    serving_url: str | None = None,
    db_path: Path | None = None,
    floors: Sequence[float] = FLOOR_CANDIDATES,
) -> MeasuredTurn:
    """One real answerer turn, then one critic call per distinct escalated pair.

    One turn, not Section 8's three attempts: three drafts per item would triple the cost and
    the sweep's rows would then differ for two reasons at once -- the floor, and which of three
    drafts each row happened to score. The consequence is stated where the numbers are
    (`format_report`): the pass rates here are a single-draft reading and not the 3-of-3 gate.
    """
    turn = await answer_turn_async(
        item.question, client=client, serving_url=serving_url, db_path=db_path
    )
    evidence = TurnEvidence.from_tool_payloads(turn.tool_payloads)
    report = verify(turn.draft, evidence)
    measured = MeasuredTurn(item=item, turn=turn, evidence=evidence, report=report)

    verdicts: dict[tuple[int, str], str] = {}
    for request in measured.requests_over(floors):
        entailment = await judge_async(
            request.claim_text,
            request.chunk_text,
            source_id=request.source_id,
            client=client,
        )
        verdicts[(request.claim_index, request.source_id)] = entailment.verdict

    return MeasuredTurn(
        item=item, turn=turn, evidence=evidence, report=report, verdicts=verdicts
    )


def collect(
    items: Sequence[GoldenSetItem] = CORPUS_ITEMS,
    *,
    serving_url: str | None = None,
    db_path: Path | None = None,
    floors: Sequence[float] = FLOOR_CANDIDATES,
) -> tuple[MeasuredTurn, ...]:
    """Measure every item, against the real API.

    `CORPUS_ITEMS` is the default swept set for `calibrate_retrieval.collect_scores`' reason,
    read one level along: those 16 items (8 answerable, 8 must-refuse) are the ones grounded in
    the docs corpus, and a claim citing prose is a claim citing a retrieved chunk. The
    tool-grounded, inventory and similarity items reach their evidence through a live tool, so a
    *prose* overlap floor calibrated on them would be calibrated on chunks they never cite. Which
    items actually produced a prose-citing claim is reported rather than assumed.
    """

    async def _run(item: GoldenSetItem) -> MeasuredTurn:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic()
        return await measure_item_async(
            item, client=client, serving_url=serving_url, db_path=db_path, floors=floors
        )

    return tuple(asyncio.run(_run(item)) for item in items)


# --------------------------------------------------------------------------------------
# Saving and replaying a collection
# --------------------------------------------------------------------------------------


def measurements_as_dict(measured: Sequence[MeasuredTurn]) -> dict[str, Any]:
    """A collection as data, so the expensive half runs once and the sweep stays auditable."""
    return {
        "floors": list(FLOOR_CANDIDATES),
        "current_floor": CURRENT_FLOOR,
        "turns": [
            {
                "item_id": turn.item_id,
                "question": turn.item.question,
                "draft": turn.turn.draft.model_dump(),
                "tool_payloads": [dict(payload) for payload in turn.turn.tool_payloads],
                "verdicts": [
                    {"claim_index": index, "source_id": source_id, "verdict": verdict}
                    for (index, source_id), verdict in sorted(turn.verdicts.items())
                ],
            }
            for turn in measured
        ],
    }


def measurements_from_dict(
    payload: Mapping[str, Any], items: Sequence[GoldenSetItem] = CORPUS_ITEMS
) -> tuple[MeasuredTurn, ...]:
    """Rebuild a collection, through `pipeline.turn_from_payloads` -- the documented seam.

    An entry whose `item_id` is not in `items` is an error rather than a skip: a saved turn the
    golden set no longer recognises cannot be scored against a contract, and dropping it
    quietly would shrink the denominator of every row in the table.
    """
    from src.agent.answerer import Draft
    from src.agent.pipeline import turn_from_payloads

    by_id = {item.item_id: item for item in items}
    rebuilt: list[MeasuredTurn] = []
    for entry in payload["turns"]:
        item = by_id.get(entry["item_id"])
        if item is None:
            raise KeyError(
                f"saved measurement names item {entry['item_id']!r}, which is not in the "
                "swept golden-set items"
            )
        turn = turn_from_payloads(Draft.model_validate(entry["draft"]), entry["tool_payloads"])
        evidence = TurnEvidence.from_tool_payloads(turn.tool_payloads)
        rebuilt.append(
            MeasuredTurn(
                item=item,
                turn=turn,
                evidence=evidence,
                report=verify(turn.draft, evidence),
                verdicts={
                    (record["claim_index"], record["source_id"]): record["verdict"]
                    for record in entry["verdicts"]
                },
            )
        )
    return tuple(rebuilt)


# --------------------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepCell:
    """One candidate floor's full result: the hard constraint, then the pass rate."""

    floor: float
    refuse_passed: int
    refuse_total: int
    answerable_passed: int
    answerable_total: int
    escalated_pairs: int
    passing_item_ids: tuple[str, ...] = ()

    def feasible(self, baseline: "SweepCell") -> bool:
        """No must-refuse item lost, relative to what the current floor gives today."""
        return self.refuse_passed >= baseline.refuse_passed


def evaluate(measured: Sequence[MeasuredTurn], floor: float) -> SweepCell:
    """Score every measured item at one floor, through Section 8's own `score_item`.

    Both halves are scored by the *same* rule here, unlike `calibrate_retrieval.evaluate`'s two
    -- and for the same underlying reason. There, a must-refuse item had no answer to score and
    was read off its top retrieval score instead; here every item has a real drafted answer and
    a real verdict, so what a floor does to a must-refuse item *is* Section 8's refusal
    sub-score (tier 3, no claim released) and what it does to an answerable item is Section 8's
    grounded-answer sub-score. Two categories, one scorer, no second definition of passing.
    """
    scores = [(turn, turn.score_at(floor)) for turn in measured]
    return SweepCell(
        floor=floor,
        refuse_passed=sum(1 for turn, score in scores if turn.is_must_refuse and score.passed),
        refuse_total=sum(1 for turn, _ in scores if turn.is_must_refuse),
        answerable_passed=sum(
            1 for turn, score in scores if not turn.is_must_refuse and score.passed
        ),
        answerable_total=sum(1 for turn, _ in scores if not turn.is_must_refuse),
        escalated_pairs=sum(len(turn.requests_at(floor)) for turn in measured),
        passing_item_ids=tuple(
            turn.item_id for turn, score in scores if score.passed
        ),
    )


def sweep(
    measured: Sequence[MeasuredTurn], floors: Sequence[float] = FLOOR_CANDIDATES
) -> tuple[SweepCell, ...]:
    """Every candidate floor, in grid order."""
    return tuple(evaluate(measured, floor) for floor in floors)


def baseline_cell(measured: Sequence[MeasuredTurn]) -> SweepCell:
    """The current floor's own row, measured from the same drafts and verdicts.

    Computed on its own rather than looked up in the sweep, so the constraint holds even if the
    grid is later changed to a range that does not contain `CURRENT_FLOOR` -- the thing every
    row is compared against must not depend on the grid happening to include it.
    """
    return evaluate(measured, CURRENT_FLOOR)


def recommend(
    cells: Sequence[SweepCell],
    baseline: SweepCell,
    measured: Sequence[MeasuredTurn] = (),
) -> SweepCell | None:
    """The floor to recommend, or `None` when there is nothing to sweep.

    Ordered exactly as Issue #173 words it:

    1. **Feasible only.** A floor that leaves fewer must-refuse items passing than the current
       one is removed, not ranked last.
    2. **Maximum answerable pass count.** This is the objective Issue #173 states; the
       must-refuse column above it is a constraint, not a second objective.
    3. **Maximum must-refuse pass count** among what is left. Not a re-weighting of step 2: it
       only ever chooses between floors that already tie on the objective, and there the
       safety-relevant category is the one Section 8 gates at 100% -- taking the floor that
       refuses one *more* out-of-corpus question costs nothing measured.
    4. **The largest distance to the nearest measured overlap.** Ties here are floors that gate
       today's measured pairs identically; the one sitting in the middle of a gap is the one
       that keeps gating that way when a draft is reworded slightly.
    5. **The lowest floor** among what remains. A tie at this point is a set of floors with the
       same verdicts and the same margin, so the cheapest is the honest choice: a lower floor
       escalates strictly fewer pairs, and each escalation is a model call on every real turn.

    `None` only when there are no cells at all. It is not the "no feasible candidate" branch
    `calibrate_retrieval` has -- the baseline satisfies its own constraint, so a recommendation
    always exists whenever the grid contains `CURRENT_FLOOR`, and it may well be to keep it.
    """
    feasible = [cell for cell in cells if cell.feasible(baseline)]
    if not feasible:
        return None
    overlaps = all_overlaps(measured)
    best_rate = max(cell.answerable_passed for cell in feasible)
    return min(
        (cell for cell in feasible if cell.answerable_passed == best_rate),
        key=lambda cell: (-cell.refuse_passed, -_margin(cell, overlaps), cell.floor),
    )


def _margin(cell: SweepCell, overlaps: Sequence[float]) -> float:
    """How far this floor sits from the nearest measured overlap, in overlap units.

    A floor is a threshold on the measured overlaps, so the room it has is its distance to the
    closest one; a floor sitting exactly on a measured value flips that pair's escalation on the
    smallest possible change to a claim's wording. With no measured overlaps to compare against
    the margin is 0.0 for every floor, which makes the tie-break fall through to the lowest
    floor rather than to grid order.
    """
    return min((abs(cell.floor - overlap) for overlap in overlaps), default=0.0)


def all_overlaps(measured: Sequence[MeasuredTurn]) -> tuple[float, ...]:
    """Every measured prose overlap, sorted -- the population the floor is a threshold on."""
    return tuple(sorted(overlap for turn in measured for _, _, overlap in turn.prose_pairs()))


# --------------------------------------------------------------------------------------
# What the floor can and cannot reach
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Reach:
    """Why one item's verdict does or does not move with the floor.

    This is the reading Issue #173 asks the PR to state plainly, computed rather than argued:
    the floor is one threshold on one trigger, and most ways a golden-set item fails are outside
    it.
    """

    item_id: str
    category: str
    cites_prose: bool
    deterministic_clean: bool
    escalatable_pairs: int
    non_yes_verdicts: int
    passes_at: tuple[float, ...]

    @property
    def floor_sensitive(self) -> bool:
        """Whether any swept floor changes this item's verdict."""
        return 0 < len(self.passes_at) < len(FLOOR_CANDIDATES)

    @property
    def reason_out_of_reach(self) -> str:
        """Why the floor cannot move this item, or "" when it can."""
        if self.floor_sensitive:
            return ""
        if not self.deterministic_clean:
            return (
                "the deterministic pass is not clean, so Section 6's escalation precondition "
                "fails and no floor escalates anything"
            )
        if not self.cites_prose:
            return "no verified claim cites a prose chunk, so the overlap trigger never applies"
        if not self.non_yes_verdicts:
            return (
                "the critic answered yes on every escalated pair, so escalating more of them "
                "demotes nothing"
            )
        if len(self.passes_at) == len(FLOOR_CANDIDATES):
            return "it passes at every swept floor"
        return "it fails at every swept floor for reasons the escalation does not decide"


def reaches(
    measured: Sequence[MeasuredTurn], cells: Sequence[SweepCell]
) -> tuple[Reach, ...]:
    passing = {cell.floor: set(cell.passing_item_ids) for cell in cells}
    return tuple(
        Reach(
            item_id=turn.item_id,
            category=turn.category,
            cites_prose=turn.cites_prose,
            deterministic_clean=turn.report.clean,
            escalatable_pairs=len(turn.requests_over(FLOOR_CANDIDATES)),
            non_yes_verdicts=sum(1 for verdict in turn.verdicts.values() if verdict != "yes"),
            passes_at=tuple(
                floor for floor in sorted(passing) if turn.item_id in passing[floor]
            ),
        )
        for turn in measured
    )


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def format_measurements(measured: Sequence[MeasuredTurn]) -> list[str]:
    """Every verified claim's best prose overlap -- the evidence the sweep is computed from.

    Printed per claim rather than summarized, so a reader can recompute any row of the table by
    hand: a claim escalates at a floor above its overlap, and its recorded verdict is printed
    beside it.
    """
    lines = ["measured prose overlaps, per verified claim (claim -> best cited prose chunk):", ""]
    for label, wanted in ((ANSWERABLE, False), (MUST_REFUSE, True)):
        lines.append(f"  {label}:")
        for turn in measured:
            if turn.is_must_refuse is not wanted:
                continue
            pairs = turn.prose_pairs()
            if not pairs:
                lines.append(f"    {turn.item_id:<48s} (no verified claim cites prose)")
                continue
            lines.append(f"    {turn.item_id}")
            for index, source_id, overlap in pairs:
                verdict = turn.verdicts.get((index, source_id), "-")
                lines.append(
                    f"      claim {index:<2d} {overlap:.4f}  critic={verdict:<8s} {source_id}"
                )
        lines.append("")
    return lines


def format_separability(measured: Sequence[MeasuredTurn]) -> list[str]:
    """Whether one floor can escalate what must be caught without escalating what must not.

    The two populations the floor sits between, and they are not the two categories:

    - **Pairs that must escalate**: a must-refuse item's claim whose recorded verdict is not
      `yes`. Escalation is what demotes it, and demoting every released claim is what makes the
      item's answer the tier-3 refusal it is scored on. A floor at or below such a pair's
      overlap leaves it released.
    - **Pairs that must not**: an answerable item's claim whose recorded verdict is not `yes`.
      Escalating it demotes a claim a correct answer needs, which costs that item its
      grounded-answer sub-score.

    A pair whose verdict is `yes` sits in neither: escalating it changes nothing but cost. If
    the first population's highest overlap is below the second's lowest, a floor exists that
    catches everything and costs nothing; where they interleave, no floor does, and the sweep's
    ceiling is a property of the drafts and the critic rather than of the threshold.
    """
    must: list[tuple[str, float]] = []
    must_not: list[tuple[str, float]] = []
    for turn in measured:
        for index, source_id, overlap in turn.prose_pairs():
            if turn.verdicts.get((index, source_id), "yes") == "yes":
                continue
            (must if turn.is_must_refuse else must_not).append(
                (f"{turn.item_id} claim {index}", overlap)
            )
    if not must and not must_not:
        return [
            "separability: nothing to separate.",
            "",
            "  No measured pair has a non-yes verdict, so no floor in the swept range demotes",
            "  anything: every row below differs only in how many pairs it pays to escalate.",
            "",
        ]

    def listed(pairs: Sequence[tuple[str, float]]) -> list[str]:
        rows = [f"    {name:<48s} {overlap:.4f}" for name, overlap in sorted(pairs)]
        return rows or ["    (none)"]

    lines = ["separability of the two populations the floor sits between:", ""]
    lines.append("  pairs that must escalate (must-refuse, critic said no/unclear):")
    lines += listed(must)
    lines.append("")
    lines.append("  pairs that must not escalate (answerable, critic said no/unclear):")
    lines += listed(must_not)
    lines.append("")
    if must and must_not:
        highest = max(overlap for _, overlap in must)
        lowest = min(overlap for _, overlap in must_not)
        if highest < lowest:
            lines += [
                f"  SEPARABLE: a floor in ({highest:.4f}, {lowest:.4f}] catches every pair that",
                "  must escalate and none that must not.",
                "",
            ]
        else:
            lines += [
                f"  They OVERLAP: the highest must-escalate overlap ({highest:.4f}) is at or",
                f"  above the lowest must-not ({lowest:.4f}), so no single floor separates them",
                "  and the trade below is real rather than an artefact of the grid.",
                "",
            ]
    return lines


def format_sweep(
    cells: Sequence[SweepCell], baseline: SweepCell, best: SweepCell | None
) -> list[str]:
    """The full sweep, every candidate floor -- Issue #173's "print the table, not the winner".

    The rejected rows are printed and marked rather than dropped: "this floor would score better
    on the answerable half if the must-refuse constraint did not exist" is exactly the trade a
    reader is entitled to watch being refused. `pairs` is the number of claim/chunk pairs that
    escalate across the whole swept set at that floor -- the per-sweep model-call count, and the
    cost side of the trade.
    """
    header = (
        "  floor    must-refuse    answerable    pairs   note"
    )
    lines = [
        "full sweep -- Section 8 scoring at each candidate floor:",
        "",
        f"  (must-refuse is the hard constraint: >= {baseline.refuse_passed}/"
        f"{baseline.refuse_total}, what the current floor {CURRENT_FLOOR} gives today.",
        "   REGRESS = fewer must-refuse items pass, so the floor is rejected whatever its",
        "   answerable column says.)",
        "",
        header,
        "  " + "-" * (len(header) - 2),
    ]
    for cell in cells:
        notes = []
        if not cell.feasible(baseline):
            notes.append("REGRESS")
        if cell.floor == CURRENT_FLOOR:
            notes.append("current")
        if best is not None and cell.floor == best.floor:
            notes.append("recommended")
        lines.append(
            (
                f"  {cell.floor:>5.2f}    "
                f"{cell.refuse_passed:>2d}/{cell.refuse_total:<2d}         "
                f"{cell.answerable_passed:>2d}/{cell.answerable_total:<2d}        "
                f"{cell.escalated_pairs:>3d}   {' '.join(notes)}"
            ).rstrip()
        )
    lines.append("")
    return lines


def format_reach(reach: Sequence[Reach]) -> list[str]:
    """Which items the floor moves, and why it cannot move the rest."""
    lines = [
        "what this floor reaches:",
        "",
        "  A floor is one threshold on one trigger. An item whose verdict is the same at every",
        "  swept floor is not evidence about the floor, and the reason it is out of reach is",
        "  named rather than left to be inferred from a flat column.",
        "",
    ]
    movable = [entry for entry in reach if entry.floor_sensitive]
    lines.append(f"  floor-sensitive items ({len(movable)} of {len(reach)}):")
    if movable:
        for entry in movable:
            passes = ", ".join(f"{floor:.2f}" for floor in entry.passes_at)
            lines.append(f"    {entry.item_id:<48s} passes at {passes}")
    else:
        lines.append("    (none)")
    lines.append("")
    out_of_reach = [entry for entry in reach if not entry.floor_sensitive]
    lines.append(f"  out of reach ({len(out_of_reach)} of {len(reach)}):")
    if out_of_reach:
        lines += [
            f"    {entry.item_id:<48s} {entry.reason_out_of_reach}" for entry in out_of_reach
        ]
    else:
        lines.append("    (none)")
    lines.append("")
    return lines


def format_recommendation(best: SweepCell | None, baseline: SweepCell) -> list[str]:
    if best is None:
        return [
            "RECOMMENDATION: none.",
            "",
            "  Nothing was measured, so there is nothing to recommend. A sweep with no",
            "  prose-citing claim in it is not a calibration of a prose overlap floor.",
            "",
        ]
    verdict = (
        "keep the current value"
        if best.floor == CURRENT_FLOOR
        else f"change {CURRENT_FLOOR} -> {best.floor:.2f}"
    )
    return [
        "RECOMMENDATION:",
        "",
        f"  LEXICAL_OVERLAP_FLOOR = {best.floor:.2f}   (currently {CURRENT_FLOOR}) -- {verdict}",
        "",
        f"  must-refuse passing : {best.refuse_passed}/{best.refuse_total}  "
        f"(the hard constraint; the current floor gives {baseline.refuse_passed}/"
        f"{baseline.refuse_total})",
        f"  answerable passing  : {best.answerable_passed}/{best.answerable_total}  "
        f"(currently {baseline.answerable_passed}/{baseline.answerable_total})",
        f"  pairs escalated     : {best.escalated_pairs}  "
        f"(currently {baseline.escalated_pairs}) -- one model call each, per real turn",
        "",
        "  This script does not apply the value. Editing LEXICAL_OVERLAP_FLOOR in",
        "  src/agent/critic/escalation.py is a separate, reviewed change, the same way #163",
        "  recommended TAU_TOP/TAU_SUPPORT and #165 applied them.",
        "",
    ]


def format_report(
    measured: Sequence[MeasuredTurn],
    cells: Sequence[SweepCell],
    baseline: SweepCell,
    best: SweepCell | None,
    *,
    source: str,
) -> str:
    prose = sum(1 for turn in measured if turn.cites_prose)
    critic_calls = sum(len(turn.verdicts) for turn in measured)
    lines = [
        "docs/agent_design.md Section 6 / Section 8 -- LEXICAL_OVERLAP_FLOOR calibration",
        f"  items swept          : {len(measured)} "
        f"({sum(1 for t in measured if not t.is_must_refuse)} answerable, "
        f"{sum(1 for t in measured if t.is_must_refuse)} must-refuse)",
        f"  prose-citing items   : {prose} of {len(measured)} "
        "(the rest cannot be reached by this floor at all)",
        f"  measurement source   : {source}",
        f"  critic verdicts held : {critic_calls} (one real call each at collection time; the",
        "                         sweep below reuses them and makes none)",
        f"  candidate floors     : {len(cells)} ({FLOOR_CANDIDATES[0]:.2f} to "
        f"{FLOOR_CANDIDATES[-1]:.2f}, step 0.05)",
        f"  current floor        : {CURRENT_FLOOR}",
        "  attempts per item    : 1 (a single-draft reading, deliberately not Section 8's",
        "                         3-of-3 gate -- see `measure_item_async`)",
        "",
    ]
    lines += format_measurements(measured)
    lines += format_separability(measured)
    lines += format_sweep(cells, baseline, best)
    lines += format_reach(reaches(measured, cells))
    lines += format_recommendation(best, baseline)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--url", default=DEFAULT_BASE_URL, help="base URL of an already-running serving API"
    )
    parser.add_argument("--db-path", type=Path, default=DB_PATH, help="inventory database path")
    parser.add_argument(
        "--measurements",
        type=Path,
        default=None,
        help="sweep a saved collection instead of calling the API (no credentials needed)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="write the collection to this path, so the sweep can be re-run without calls",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.measurements is not None:
        payload = json.loads(args.measurements.read_text(encoding="utf-8"))
        measured = measurements_from_dict(payload)
        source = f"replayed from {args.measurements}"
    else:
        measured = collect(serving_url=args.url, db_path=args.db_path)
        source = "real answerer + critic calls, made now"
        if args.save is not None:
            args.save.write_text(
                json.dumps(measurements_as_dict(measured), indent=2) + "\n", encoding="utf-8"
            )
            source += f", saved to {args.save}"

    cells = sweep(measured)
    baseline = baseline_cell(measured)
    best = recommend(cells, baseline, measured)
    print(format_report(measured, cells, baseline, best, source=source))
    return 0 if best is not None else 1


if __name__ == "__main__":
    sys.exit(main())
