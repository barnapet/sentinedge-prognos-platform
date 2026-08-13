"""Section 8's retrieval quality, measured separately from generation quality (Issue #152,
part 3b).

    from tests.fixtures.golden_set_retrieval import score_retrieval, summarize_retrieval

`docs/agent_design.md` Section 8's "Retrieval quality, measured separately from generation
quality" (added by Issue #102) asks for four things this module computes and
`golden_set_runner.py` reports: **recall@k**, **precision@k**, the must-refuse category's
**mirror metric**, and the **2x2 table** of retrieval outcome against answer outcome.

**Why a second module rather than more of `golden_set_runner.py`.** Part 3a's gates and this
are two numbers that Section 8 requires never be averaged into one ("a single blended 'RAG
score' is the same aggregate-hides-the-subgroup failure `docs/evaluation_protocol.md` §5
forbids, one level down"). Keeping them in separate modules makes that a property of the
import graph rather than a promise: nothing here can reach `score_item`, `_score_tool_call`,
`_score_grounded_answer` or `_score_refusal`, because nothing here imports them. The
dependency runs one way only -- the runner imports this, this imports the runner's *inputs*
(`GoldenSetItem`) and the constants' owning modules, never the runner.

## What "the retrieval" is read from

`search_documentation` returns one envelope per call carrying a `results` list, each entry
its own `source` block whose `source_id` is the chunk's real `chunk_id`, plus a `score`,
ranked most-similar-first (`src/agent/mcp/tools.py`). That list is the input to every metric
here, and it is read **without disturbing** what part 3a and the critic already read from the
same payloads: `TurnEvidence.from_tool_payloads` (which flattens both id levels into citable
evidence) and `resolve_tool_names` (which reads only the top-level `source` block) are
untouched and still see exactly what they saw before.

## k

`k` is `search_documentation`'s own `DEFAULT_LIMIT`, imported rather than restated. Issue
#146's `relevant_chunk_ids` were hand-verified against that same limit, so 5 is already what
"the harness" means operationally, and a different `k` here would be measuring against labels
nobody made.

A model may pass a larger `limit`, and two searches in one turn can return two lists, so
"the top k" is defined here as: every retrieved chunk, de-duplicated by `chunk_id` keeping its
best score, ordered by score descending (ties keeping the order the payloads listed them in),
truncated to `k`. `retrieved_count` is reported alongside, so a turn that retrieved more than
`k` is visible rather than silently truncated.

## Which items get a reading, and which deliberately do not

| items | reading |
|---|---|
| the 8 with non-empty `relevant_chunk_ids` | recall@k and precision@k |
| the 8 must-refuse items (empty by design) | the mirror metric only |
| the 14 tool-grounded items (Issue #148) | **none** |

The third row is Issue #152's constraint and it is a real decision, not an omission: those
items are answered from `get_bearing_status`, `check_inventory` and
`find_similar_historical_pattern`, which perform no vector search at all. A recall@k of 0 for
an item that correctly never retrieved would be a number that looks like a failure and means
nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from src.agent.critic.retrieval_confidence import TAU_TOP
from src.agent.mcp.tools import DOCS_SOURCE_ID
from src.agent.rag.retrieval import DEFAULT_LIMIT
from tests.fixtures.golden_set import GoldenSetItem

# Section 8's "at the same *k* the answerer actually retrieves with", imported from the module
# that owns it. Issue #152: do not change this without saying so explicitly and why.
K = DEFAULT_LIMIT

# The `source` block `search_documentation` mints for the search itself (the per-chunk blocks
# inside `data.results` carry `decision_doc`/`public_reference` and the chunk ids). Imported
# from `src/agent/mcp/tools.py` for the same reason `golden_set_runner.TOOL_BY_SOURCE` does:
# a literal copied to here could drift from what the tool actually mints.
SEARCH_SOURCE = ("live_endpoint", DOCS_SOURCE_ID)

# The name that source block resolves to. Pinned against `golden_set_runner`'s mapping and
# against `READONLY_TOOL_NAMES` by a test rather than imported, to keep this module's import
# graph one-directional (see the module docstring).
SEARCH_TOOL_NAME = "search_documentation"

# The three readings an item can get.
RECALL_PRECISION = "recall_precision"
BELOW_THRESHOLD = "below_threshold"
NOT_APPLICABLE = "not_applicable"

# The 2x2's four cells, named as Section 8's own table names them. Constants because two of
# them are the whole point of building the table and a reader has to be able to find them.
CELL_WORKING = "working as designed"
CELL_UNGROUNDED_CORRECT = "right answer, no evidence"
CELL_GENERATION_FAILURE = "generation failure"
CELL_RETRIEVAL_FAILURE = "retrieval failure"


# --------------------------------------------------------------------------------------
# Reading the ranked list off a run's payloads
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RankedChunk:
    """One retrieved chunk: the id Section 6's checks test membership against, and the
    similarity that retrieved it.

    `score` is optional because the envelope's own schema makes it optional
    (`src/agent/mcp/tools.py` writes `chunk.score`, and `deterministic.py` already treats a
    missing one as `None`). A scoreless chunk still counts as retrieved -- it is in the
    ranked list, which is what recall@k and precision@k ask about -- but it cannot be the
    top score, which is what the mirror metric asks about.
    """

    chunk_id: str
    score: float | None = None


@dataclass(frozen=True)
class SearchOutcome:
    """What a run's `search_documentation` calls produced, as the metrics need it.

    Three states a single "ranked list" cannot express, and all three occur in practice:

    - **never searched** (`searched=False`) -- no `search_documentation` result at all. Every
      tool-grounded item, and any corpus item whose model chose not to search.
    - **searched and failed** (`failed=True`) -- the documentation index was unreachable, so
      the envelope carried `error` and no `results`. This is CI's normal state and the state
      Section 8's cassettes are deliberately recorded in, which is exactly why it is a flag
      and not an empty list: an empty list would make the must-refuse mirror metric *pass*
      trivially, and a trivially-passing safety metric is worse than an absent one.
    - **searched and retrieved** -- `ranked` is non-empty.
    """

    ranked: tuple[RankedChunk, ...] = ()
    searched: bool = False
    failed: bool = False

    @property
    def retrieved_count(self) -> int:
        """Distinct chunks retrieved this turn, before truncation to `k`."""
        return len(self.ranked)

    @property
    def top_score(self) -> float | None:
        """The best similarity this turn retrieved, or `None` if it retrieved nothing."""
        scores = [chunk.score for chunk in self.ranked if chunk.score is not None]
        return max(scores) if scores else None

    def top_k(self, k: int = K) -> tuple[RankedChunk, ...]:
        return self.ranked[:k]


def is_search_payload(payload: Mapping[str, Any]) -> bool:
    """Whether one tool payload came from `search_documentation`.

    Matched on the `source` block the tool layer minted, never on anything a model wrote --
    Section 2's rule, and the same basis `golden_set_runner.resolve_tool_name` uses.
    """
    source = payload.get("source") or {}
    return (
        str(source.get("source_type", "")),
        str(source.get("source_id", "")),
    ) == SEARCH_SOURCE


def ranked_chunks(payloads: Iterable[Mapping[str, Any]]) -> tuple[RankedChunk, ...]:
    """Every chunk this turn retrieved, best score first, de-duplicated by `chunk_id`.

    De-duplicated because two searches in one turn legitimately return the same chunk, and a
    duplicate would inflate precision@k's denominator with a chunk the answerer saw once. The
    surviving copy keeps the **best** score, since that is the score the retrieval actually
    achieved for that chunk.

    The sort is stable, so chunks with equal scores keep the order the payloads listed them
    in -- which for a single search call is the retriever's own ranking, preserved rather
    than re-derived.
    """
    best: dict[str, RankedChunk] = {}
    for payload in payloads:
        if not is_search_payload(payload):
            continue
        data = payload.get("data")
        results = data.get("results", []) if isinstance(data, dict) else []
        for result in results:
            chunk_source = result.get("source") or {}
            chunk_id = chunk_source.get("source_id")
            if not chunk_id:
                continue
            raw_score = result.get("score")
            score = float(raw_score) if raw_score is not None else None
            previous = best.get(str(chunk_id))
            if previous is None:
                best[str(chunk_id)] = RankedChunk(str(chunk_id), score)
            elif score is not None and (previous.score is None or score > previous.score):
                best[str(chunk_id)] = RankedChunk(str(chunk_id), score)
    return tuple(
        sorted(best.values(), key=lambda chunk: -(chunk.score or 0.0))
    )


def search_outcome(payloads: Sequence[Mapping[str, Any]]) -> SearchOutcome:
    """A run's `search_documentation` results, as one `SearchOutcome`."""
    search_payloads = [payload for payload in payloads if is_search_payload(payload)]
    return SearchOutcome(
        ranked=ranked_chunks(search_payloads),
        searched=bool(search_payloads),
        failed=any("error" in payload for payload in search_payloads),
    )


# --------------------------------------------------------------------------------------
# The three metrics
# --------------------------------------------------------------------------------------


def recall_at_k(
    outcome: SearchOutcome, relevant: frozenset[str], *, k: int = K
) -> bool:
    """Did **at least one** relevant chunk appear in the top `k`?

    "At least one," not "all", straight from Section 8: Section 4's ~200-character overlap
    means a fact frequently appears in two adjacent chunks and either is a legitimate
    retrieval. A turn that retrieved nothing is a miss -- there was no relevant chunk in the
    top `k` because there was no top `k`.
    """
    return any(chunk.chunk_id in relevant for chunk in outcome.top_k(k))


def precision_at_k(
    outcome: SearchOutcome, relevant: frozenset[str], *, k: int = K
) -> float | None:
    """What fraction of the top `k` were relevant -- or `None` when nothing was retrieved.

    `None` rather than `0.0`, and the distinction is not pedantry: "the fraction of the
    ranked list that was relevant" has no value for an empty ranked list, and reporting one
    as `0.0` would drag a mean downward with items that measured nothing at all. The
    aggregate excludes them and says how many it excluded; recall, whose answer for the same
    turn is an unambiguous miss, does not.

    The denominator is the length of the list actually returned (capped at `k`), not `k`
    itself: a search that returns 3 chunks of which 3 are relevant has precision 1.0, not
    0.6.
    """
    top = outcome.top_k(k)
    if not top:
        return None
    return sum(1 for chunk in top if chunk.chunk_id in relevant) / len(top)


def top_score_below_tau(outcome: SearchOutcome, *, tau_top: float = TAU_TOP) -> bool:
    """The must-refuse category's mirror metric: did the top score stay below `TAU_TOP`?

    Section 8: for those 8 items "the correct retrieval outcome is that *nothing* clears
    `TAU_TOP`, so recall@k is not the question -- what is recorded is the top similarity
    score and whether it stayed below threshold. A refusal issued while a spuriously similar
    chunk cleared threshold is a refusal that came out right for the wrong reason."

    **Strictly the top score, not `assess_retrieval(...).passed`.** That function's verdict is
    a two-part gate (top >= `TAU_TOP` *and* at least two chunks >= `TAU_SUPPORT`), so its
    negation is also true when a single spuriously-similar chunk cleared `TAU_TOP` alone with
    no support behind it -- which is precisely the case Section 8 wants recorded as a
    failure. Reusing it would invert this metric on the one shape it exists to catch.

    "Nothing retrieved" passes, per Issue #152's own wording -- but `SearchOutcome.failed`
    and `.searched` travel with the verdict so a reader can tell a real below-threshold
    result from an unreachable index, and `RetrievalReport` counts the trivial ones out loud.
    """
    top = outcome.top_score
    return top is None or top < tau_top


# --------------------------------------------------------------------------------------
# One item's reading
# --------------------------------------------------------------------------------------


def retrieval_kind(item: GoldenSetItem, *, must_refuse_category: str) -> str:
    """Which of the three readings this item gets (see the module docstring's table).

    `must_refuse_category` is passed in rather than imported, so this module does not depend
    on `golden_set_runner`; the runner passes its own `MUST_REFUSE` constant and a test pins
    that the two agree.
    """
    if item.relevant_chunk_ids:
        return RECALL_PRECISION
    if item.category == must_refuse_category and SEARCH_TOOL_NAME in item.expected_tool_names:
        return BELOW_THRESHOLD
    return NOT_APPLICABLE


@dataclass(frozen=True)
class RetrievalQuality:
    """One item's retrieval reading, and the answer outcome it is tabulated against.

    Every metric field is `None` when it does not apply, and the three ways that happens are
    kept distinct on purpose: an item this reading does not apply to (`kind` is
    `NOT_APPLICABLE`), an item it applies to whose run produced no observation (`scored` is
    False -- an errored run, or an `ItemRun` built by hand), and a metric that is genuinely
    undefined for an observed run (`precision_at_k` with an empty ranked list).

    `answer_correct` is part 3a's `ItemScore.passed`, carried here **unchanged and unread by
    anything that produces a number above** -- it is one axis of the 2x2 and nothing else.
    """

    item_id: str
    category: str
    kind: str
    scored: bool = False
    searched: bool = False
    search_failed: bool = False
    retrieved_count: int = 0
    top_k_chunk_ids: tuple[str, ...] = ()
    top_score: float | None = None
    recall_at_k: bool | None = None
    precision_at_k: float | None = None
    top_below_tau: bool | None = None
    answer_correct: bool | None = None

    @property
    def applicable(self) -> bool:
        return self.kind != NOT_APPLICABLE

    @property
    def trivially_below_tau(self) -> bool:
        """A below-threshold pass that measured nothing: the index was unreachable, or no
        search happened at all. Counted separately in the report, never silently."""
        return self.top_below_tau is True and (self.search_failed or not self.searched)


def score_retrieval(
    item: GoldenSetItem,
    outcome: SearchOutcome | None,
    *,
    must_refuse_category: str,
    answer_correct: bool | None = None,
    k: int = K,
    tau_top: float = TAU_TOP,
) -> RetrievalQuality:
    """One item's reading. `outcome=None` means the run produced no observation."""
    kind = retrieval_kind(item, must_refuse_category=must_refuse_category)
    if kind == NOT_APPLICABLE or outcome is None:
        return RetrievalQuality(
            item_id=item.item_id,
            category=item.category,
            kind=kind,
            answer_correct=answer_correct,
        )

    common = {
        "item_id": item.item_id,
        "category": item.category,
        "kind": kind,
        "scored": True,
        "searched": outcome.searched,
        "search_failed": outcome.failed,
        "retrieved_count": outcome.retrieved_count,
        "top_k_chunk_ids": tuple(chunk.chunk_id for chunk in outcome.top_k(k)),
        "top_score": outcome.top_score,
        "answer_correct": answer_correct,
    }
    if kind == RECALL_PRECISION:
        return RetrievalQuality(
            **common,
            recall_at_k=recall_at_k(outcome, item.relevant_chunk_ids, k=k),
            precision_at_k=precision_at_k(outcome, item.relevant_chunk_ids, k=k),
        )
    return RetrievalQuality(
        **common, top_below_tau=top_score_below_tau(outcome, tau_top=tau_top)
    )


# --------------------------------------------------------------------------------------
# The 2x2 and the report
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalTable:
    """Section 8's 2x2: retrieval outcome x answer outcome, over the recall-scored items.

    Built over the items with `relevant_chunk_ids` only. The must-refuse items are not a row
    of this table -- their retrieval axis is a different question (did anything clear
    `TAU_TOP`, which *should* be "no"), and putting the two in one table would mean one axis
    labelled two ways. They are reported next to it instead.

    `not_scored` counts items whose run produced no observation at all; they are excluded
    from the four cells rather than assigned to one, for the same reason part 3a keeps an
    errored run distinct from a failing one: a failed item is a result, an errored item is
    the absence of one.
    """

    retrieved_correct: int = 0
    retrieved_wrong: int = 0
    missed_correct: int = 0
    missed_wrong: int = 0
    not_scored: int = 0

    @property
    def total(self) -> int:
        return (
            self.retrieved_correct
            + self.retrieved_wrong
            + self.missed_correct
            + self.missed_wrong
            + self.not_scored
        )

    @property
    def ungrounded_correct(self) -> int:
        """The top-right cell. Section 8: "the one an end-to-end score cannot see and would
        score as a win" -- a correct answer produced without the evidence for it, i.e. from
        parametric knowledge, "recorded as a **failure**, not a pass".

        Recorded as a failure **of this table**, and deliberately not folded into either of
        part 3a's gates (Issue #152's constraint): `format_retrieval_section` prints it as a
        failure and `no_ungrounded_correct_answers` is its own verdict, next to the two gates
        rather than inside them.
        """
        return self.missed_correct

    @property
    def no_ungrounded_correct_answers(self) -> bool:
        """The top-right cell's own verdict. Its own field, not part of `gates_passed`."""
        return self.missed_correct == 0


@dataclass(frozen=True)
class RetrievalReport:
    """Retrieval quality, as the second of Section 8's two numbers -- never averaged into the
    first. Carries `k` and `tau_top` on its face because both are what its numbers mean."""

    qualities: tuple[RetrievalQuality, ...] = ()
    k: int = K
    tau_top: float = TAU_TOP

    # --- recall / precision, over the items that declare relevant chunks ---------------

    @property
    def recall_items(self) -> tuple[RetrievalQuality, ...]:
        return tuple(q for q in self.qualities if q.kind == RECALL_PRECISION and q.scored)

    @property
    def recall_hits(self) -> int:
        return sum(1 for q in self.recall_items if q.recall_at_k)

    @property
    def recall_total(self) -> int:
        return len(self.recall_items)

    @property
    def recall_rate(self) -> float:
        return self.recall_hits / self.recall_total if self.recall_total else 0.0

    @property
    def precision_values(self) -> tuple[float, ...]:
        return tuple(
            q.precision_at_k for q in self.recall_items if q.precision_at_k is not None
        )

    @property
    def mean_precision(self) -> float | None:
        """Mean precision@k over the items where it is defined, or `None` if none are.

        A mean over a subset, and the subset size is reported next to it
        (`precision_undefined`) rather than left for a reader to infer -- the same reason
        part 3a prints both halves of every rate.
        """
        values = self.precision_values
        return sum(values) / len(values) if values else None

    @property
    def precision_undefined(self) -> int:
        """Recall-scored items that retrieved nothing, so precision has no value for them."""
        return sum(1 for q in self.recall_items if q.precision_at_k is None)

    # --- the must-refuse mirror -------------------------------------------------------

    @property
    def mirror_items(self) -> tuple[RetrievalQuality, ...]:
        return tuple(q for q in self.qualities if q.kind == BELOW_THRESHOLD and q.scored)

    @property
    def mirror_passed(self) -> int:
        return sum(1 for q in self.mirror_items if q.top_below_tau)

    @property
    def mirror_total(self) -> int:
        return len(self.mirror_items)

    @property
    def mirror_trivial(self) -> int:
        """Mirror passes that measured nothing (no search, or the index was unreachable)."""
        return sum(1 for q in self.mirror_items if q.trivially_below_tau)

    # --- the 2x2 ----------------------------------------------------------------------

    @property
    def table(self) -> RetrievalTable:
        scored = [q for q in self.qualities if q.kind == RECALL_PRECISION]
        cells = {
            (True, True): 0,
            (True, False): 0,
            (False, True): 0,
            (False, False): 0,
        }
        not_scored = 0
        for quality in scored:
            if not quality.scored or quality.answer_correct is None:
                not_scored += 1
                continue
            cells[(bool(quality.recall_at_k), bool(quality.answer_correct))] += 1
        return RetrievalTable(
            retrieved_correct=cells[(True, True)],
            retrieved_wrong=cells[(True, False)],
            missed_correct=cells[(False, True)],
            missed_wrong=cells[(False, False)],
            not_scored=not_scored,
        )

    @property
    def not_applicable(self) -> int:
        """Items with no retrieval reading, by design (the 14 tool-grounded ones)."""
        return sum(1 for q in self.qualities if not q.applicable)


def summarize_retrieval(
    qualities: Iterable[RetrievalQuality], *, k: int = K, tau_top: float = TAU_TOP
) -> RetrievalReport:
    return RetrievalReport(qualities=tuple(qualities), k=k, tau_top=tau_top)


def format_retrieval_section(report: RetrievalReport) -> list[str]:
    """The retrieval half of the printed report, as lines.

    Printed **below** part 3a's two gates and never inside them. The header says so, because
    the cheapest way to reintroduce the blended "RAG score" Section 8 forbids is for a reader
    to assume these numbers are already in the verdict above.
    """
    table = report.table
    lines = [
        "",
        f"retrieval quality (k={report.k}) -- Section 8's second number, reported next to "
        "the gates above and folded into neither:",
    ]
    if report.not_applicable:
        lines.append(
            f"  {report.not_applicable} item(s) get no retrieval reading by design "
            "(tool-grounded items perform no vector search)."
        )

    if report.recall_total:
        precision = report.mean_precision
        precision_text = "n/a" if precision is None else f"{precision:.2f}"
        lines += [
            f"  recall@{report.k}:    {report.recall_hits}/{report.recall_total} "
            f"({report.recall_rate:.0%})  -- at least one relevant chunk in the top "
            f"{report.k}",
            f"  precision@{report.k}: {precision_text} mean over "
            f"{len(report.precision_values)} item(s)"
            + (
                f"; {report.precision_undefined} retrieved nothing, so precision is "
                "undefined for them and they are excluded from this mean (their recall "
                "still counts as a miss)"
                if report.precision_undefined
                else ""
            ),
        ]
    else:
        lines.append(
            f"  recall@{report.k}/precision@{report.k}: no item produced a retrieval "
            "observation to score."
        )

    if report.mirror_total:
        lines += [
            "",
            f"  must-refuse mirror metric (top score < TAU_TOP = {report.tau_top}), its own "
            "pass/fail, not folded into recall/precision:",
            f"    {report.mirror_passed}/{report.mirror_total} stayed below threshold",
        ]
        if report.mirror_trivial:
            lines.append(
                f"    NOTE: {report.mirror_trivial} of those passed trivially -- nothing was "
                "retrieved at all (no search, or the documentation index was unreachable), "
                "so the threshold was never tested."
            )

    row_label = max(len("answer correct"), len("answer incorrect"))
    retrieved_column = [
        "relevant chunk retrieved",
        f"{table.retrieved_correct:>2} ({CELL_WORKING})",
        f"{table.retrieved_wrong:>2} ({CELL_GENERATION_FAILURE})",
    ]
    width = max(len(cell) for cell in retrieved_column)
    lines += [
        "",
        "  2x2 -- retrieval outcome x answer outcome (answer outcome is part 3a's gates, "
        "unchanged):",
        f"    {'':<{row_label}} | {retrieved_column[0]:<{width}} | not retrieved",
        f"    {'answer correct':<{row_label}} | {retrieved_column[1]:<{width}} | "
        f"{table.missed_correct:>2} ({CELL_UNGROUNDED_CORRECT})",
        f"    {'answer incorrect':<{row_label}} | {retrieved_column[2]:<{width}} | "
        f"{table.missed_wrong:>2} ({CELL_RETRIEVAL_FAILURE})",
    ]
    if table.not_scored:
        lines.append(
            f"    {table.not_scored} item(s) produced no retrieval observation and are in no "
            "cell."
        )
    if table.ungrounded_correct:
        lines.append(
            f"    {table.ungrounded_correct} item(s) in the "
            f"'{CELL_UNGROUNDED_CORRECT}' cell -- Section 8 records that as a FAILURE, not a "
            "pass: the answer was right without the evidence for it."
        )
    return lines


def retrieval_as_dict(report: RetrievalReport) -> dict[str, Any]:
    """The retrieval half as data. Its own key on the report dict, never merged into the
    gates'."""
    table = report.table
    return {
        "k": report.k,
        "tau_top": report.tau_top,
        "recall_at_k": {
            "hits": report.recall_hits,
            "total": report.recall_total,
            "rate": report.recall_rate,
        },
        "precision_at_k": {
            "mean": report.mean_precision,
            "measured_items": len(report.precision_values),
            "undefined_items": report.precision_undefined,
        },
        "must_refuse_mirror": {
            "below_threshold": report.mirror_passed,
            "total": report.mirror_total,
            "trivially_below_threshold": report.mirror_trivial,
        },
        "two_by_two": {
            "retrieved_answer_correct": table.retrieved_correct,
            "retrieved_answer_incorrect": table.retrieved_wrong,
            "not_retrieved_answer_correct": table.missed_correct,
            "not_retrieved_answer_incorrect": table.missed_wrong,
            "not_scored": table.not_scored,
            "cell_names": {
                "retrieved_answer_correct": CELL_WORKING,
                "not_retrieved_answer_correct": CELL_UNGROUNDED_CORRECT,
                "retrieved_answer_incorrect": CELL_GENERATION_FAILURE,
                "not_retrieved_answer_incorrect": CELL_RETRIEVAL_FAILURE,
            },
            "ungrounded_correct_is_a_failure": True,
            "no_ungrounded_correct_answers": table.no_ungrounded_correct_answers,
        },
        "not_applicable_items": report.not_applicable,
        "per_item": [
            {
                "item_id": q.item_id,
                "category": q.category,
                "kind": q.kind,
                "scored": q.scored,
                "searched": q.searched,
                "search_failed": q.search_failed,
                "retrieved_count": q.retrieved_count,
                "top_k_chunk_ids": list(q.top_k_chunk_ids),
                "top_score": q.top_score,
                "recall_at_k": q.recall_at_k,
                "precision_at_k": q.precision_at_k,
                "top_below_tau": q.top_below_tau,
                "answer_correct": q.answer_correct,
            }
            for q in report.qualities
            if q.applicable
        ],
    }
