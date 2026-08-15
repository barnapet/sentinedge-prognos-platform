"""Calibrate `TAU_TOP`/`TAU_SUPPORT` against the golden set's retrieval scores (Issue #163).

    python -m tests.fixtures.calibrate_retrieval

`src/agent/critic/retrieval_confidence.py`'s own docstring names its two thresholds
"starting values, not decisions", and `docs/agent_design.md` Section 6 fixes the procedure
that replaces them: sweep against Section 8's golden set, choose the pair that keeps every
must-refuse item refusing while maximizing pass rate on the answerable ones, and **publish
the measured values and the sweep**. This script is that procedure, run against real
retrieval scores.

**No Anthropic API call, anywhere.** Section 8 built `relevant_chunk_ids` and the
recall@k/precision@k reading specifically so retrieval can be scored without an end-to-end
answer, and this script uses that: it calls `src/agent/rag/retrieval.py`'s `search()`
directly on each item's own `question` text, at production's `DEFAULT_LIMIT`, and scores the
returned similarities. The model that writes an answer is never involved, so a full
calibration costs nothing and can be re-run whenever the corpus moves.

**It changes no production constant.** Issue #163 recommends a pair with evidence; editing
`TAU_TOP`/`TAU_SUPPORT` is a separate, reviewed follow-on. The current values are imported
and printed beside the recommendation purely so the two can be compared. `assess_retrieval`
already takes both thresholds as keyword arguments, so the sweep evaluates candidate pairs
through the **real** production predicate rather than a reimplementation of it -- a
reimplementation could agree with itself while disagreeing with what actually gates a turn.
`MIN_SUPPORTING_CHUNKS` is left at its imported value throughout: Section 6 fixes it at 2 and
it is not what this calibrates.

**It lives under `tests/` for Issue #122's reason**, the same one `golden_set_runner.py`
records: the golden set is test infrastructure, and no module under `src/agent/` may know
`tests/fixtures/golden_set_corpus.py` exists. The dependency runs one way only.

## What it needs

A Qdrant carrying the `prognos_docs` collection, which is what `search()` queries:

    docker compose --profile agent up -d qdrant
    python -m src.agent.rag.index

`golden_set_corpus.py`'s `relevant_chunk_ids` were read off exactly such an index (522
chunks, 22 documents). A calibration run against a *differently* built index is measuring a
different corpus, so the point count is printed in the header rather than assumed -- a
reader comparing two runs needs to see that they indexed the same thing.

## The one hard constraint, and why it is not a tunable

Every must-refuse item's top score must fall **below** `tau_top`, with zero tolerance. That
mirrors Section 8's gate 1 -- "every one of the 8 must-refuse items must pass individually.
100%, no aggregate" -- and it is the same reasoning: a must-refuse item whose retrieval
clears the bar is an out-of-corpus question the grounding contract will let through as
`partial` rather than refuse, and averaging that away is the precise failure
`docs/evaluation_protocol.md` §5 forbids. So it is a filter applied before pass rate is even
looked at, not a term traded off against it. If no swept pair satisfies it, this script says
so and recommends nothing; a best-effort compromise pair would be a threshold nobody
measured, which is the thing Section 6 wrote the procedure to prevent.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Sequence

from src.agent.critic.retrieval_confidence import (
    MIN_SUPPORTING_CHUNKS,
    TAU_SUPPORT,
    TAU_TOP,
    assess_retrieval,
)
from src.agent.rag.index import COLLECTION_NAME, DEFAULT_QDRANT_URL
from src.agent.rag.retrieval import DEFAULT_LIMIT, search
from tests.fixtures.golden_set_corpus import CORPUS_ITEMS
from tests.fixtures.golden_set_runner import MUST_REFUSE

# The swept grid. `tau_top` carries the fine step because it is the parameter the hard
# constraint binds on -- the feasible region begins immediately above the highest must-refuse
# top score, and a coarse grid would report the first grid point past that boundary as "the"
# answer while hiding how much room there actually is. `tau_support` is the corroboration
# floor and moves the result only at its high end (a second chunk has to clear it), so it is
# swept coarsely across a wide range.
TAU_TOP_CANDIDATES: tuple[float, ...] = tuple(round(0.40 + 0.01 * step, 4) for step in range(46))
TAU_SUPPORT_CANDIDATES: tuple[float, ...] = tuple(
    round(0.30 + 0.05 * step, 4) for step in range(10)
)


@dataclass(frozen=True)
class ItemScores:
    """One item's ranked retrieval scores, measured once and swept against many times.

    `search()` is called once per item and the scores are kept, rather than re-queried per
    candidate pair: embedding a query is the expensive part, the thresholds are applied to
    the scores afterwards, and re-querying would make an identical result look like 460
    independent measurements instead of 16.
    """

    item_id: str
    category: str
    scores: tuple[float, ...]

    @property
    def is_must_refuse(self) -> bool:
        return self.category == MUST_REFUSE

    @property
    def top(self) -> float | None:
        """The highest similarity, or `None` when the search returned nothing at all."""
        return self.scores[0] if self.scores else None

    def passes(self, tau_top: float, tau_support: float) -> bool:
        """Production's own predicate, at this candidate pair."""
        return assess_retrieval(
            self.scores,
            tau_top=tau_top,
            tau_support=tau_support,
            min_supporting=MIN_SUPPORTING_CHUNKS,
        ).passed


def collect_scores(
    *,
    limit: int = DEFAULT_LIMIT,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    collection_name: str = COLLECTION_NAME,
) -> tuple[ItemScores, ...]:
    """Retrieve once for each of the 16 corpus-grounded items, in the file's own order.

    Only `CORPUS_ITEMS` is read. The tool-grounded, inventory and similarity items reach
    their evidence through a live tool rather than the vector index, so a retrieval threshold
    calibrated on them would be calibrated on searches they never make.
    """
    return tuple(
        ItemScores(
            item_id=item.item_id,
            category=item.category,
            scores=tuple(
                hit.score
                for hit in search(
                    item.question,
                    limit=limit,
                    qdrant_url=qdrant_url,
                    collection_name=collection_name,
                )
            ),
        )
        for item in CORPUS_ITEMS
    )


@dataclass(frozen=True)
class SweepCell:
    """One candidate pair's full result: the hard constraint, then the pass rate."""

    tau_top: float
    tau_support: float
    refuse_below: int
    refuse_total: int
    answerable_passed: int
    answerable_total: int

    @property
    def feasible(self) -> bool:
        """Every must-refuse item's top score is below `tau_top`. Zero tolerance."""
        return self.refuse_total > 0 and self.refuse_below == self.refuse_total


def evaluate(
    measured: Sequence[ItemScores], tau_top: float, tau_support: float
) -> SweepCell:
    """Score one candidate pair against every measured item.

    The two halves are counted by **different** rules, because the two categories are asking
    different questions. An answerable item is scored on `assess_retrieval` -- the whole
    predicate, top score and corroboration together, exactly as a turn is gated. A
    must-refuse item is scored on its top score alone: Section 8 records for those items
    "the top similarity score and whether it stayed below `TAU_TOP`", and the corroboration
    count cannot rescue an item whose top hit already cleared the bar.
    """
    refusing = [item for item in measured if item.is_must_refuse]
    answerable = [item for item in measured if not item.is_must_refuse]
    return SweepCell(
        tau_top=tau_top,
        tau_support=tau_support,
        refuse_below=sum(1 for item in refusing if item.top is None or item.top < tau_top),
        refuse_total=len(refusing),
        answerable_passed=sum(1 for item in answerable if item.passes(tau_top, tau_support)),
        answerable_total=len(answerable),
    )


def sweep(measured: Sequence[ItemScores]) -> tuple[SweepCell, ...]:
    """Every candidate pair with `tau_support <= tau_top`.

    The inequality is not a search-space trim, it is the invariant the production module
    states outright ("the top chunk counts toward it, since `TAU_TOP` is above `TAU_SUPPORT`
    by construction"). A pair that inverted them would describe a corroboration floor
    stricter than the bar it corroborates.
    """
    return tuple(
        evaluate(measured, tau_top, tau_support)
        for tau_top in TAU_TOP_CANDIDATES
        for tau_support in TAU_SUPPORT_CANDIDATES
        if tau_support <= tau_top
    )


def recommend(
    cells: Sequence[SweepCell], measured: Sequence[ItemScores]
) -> SweepCell | None:
    """The pair to recommend, or `None` when the hard constraint is unsatisfiable in range.

    Ordered exactly as Section 6 words it, with two tie-breaks that only ever choose between
    pairs already tied on the things Section 6 names:

    1. **Feasible only.** Non-feasible pairs are removed, not ranked last.
    2. **Maximum answerable pass count.**
    3. **The most balanced `tau_top`**, by the larger of the two margins it sits between --
       distance down to the highest must-refuse top score, and distance up to the lowest
       answerable top score it still admits. Ties here are pairs that gate identically on
       today's 16 measurements; the balanced one is the one that keeps gating that way when
       the corpus shifts slightly in either direction.
    4. **The largest `tau_support`** among what remains -- the strongest corroboration
       requirement that costs no answerable item, rather than the loosest one that happens
       to sort first.
    """
    feasible = [cell for cell in cells if cell.feasible]
    if not feasible:
        return None
    best_rate = max(cell.answerable_passed for cell in feasible)
    return max(
        (cell for cell in feasible if cell.answerable_passed == best_rate),
        key=lambda cell: (_balance(cell, measured), cell.tau_support),
    )


def _balance(cell: SweepCell, measured: Sequence[ItemScores]) -> float:
    """How much room `tau_top` has on its tighter side, in similarity units.

    The smaller of: how far it sits above the highest must-refuse top score (the hard
    constraint's margin), and how far below the lowest top score among the answerable items
    it still admits (the pass rate's margin). Maximizing the smaller of the two is the
    max-margin choice between the two classes.
    """
    refuse_tops = [item.top for item in measured if item.is_must_refuse and item.top is not None]
    admitted = [
        item.top
        for item in measured
        if not item.is_must_refuse
        and item.passes(cell.tau_top, cell.tau_support)
        and item.top is not None
    ]
    below = cell.tau_top - max(refuse_tops) if refuse_tops else cell.tau_top
    above = min(admitted) - cell.tau_top if admitted else 0.0
    return min(below, above)


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def format_measurements(measured: Sequence[ItemScores]) -> list[str]:
    """Every item's ranked scores -- the evidence the sweep is computed from.

    Printed in full rather than summarized to a min/max, because a reader checking this
    calibration should be able to recompute any cell of the sweep table by hand from these
    numbers alone.
    """
    lines = ["measured retrieval scores (ranked, most similar first):", ""]
    for label, wanted in (("answerable from the docs", False), ("must refuse", True)):
        lines.append(f"  {label}:")
        for item in measured:
            if item.is_must_refuse is not wanted:
                continue
            ranked = "  ".join(f"{score:.4f}" for score in item.scores) or "(no hits)"
            lines.append(f"    {item.item_id:<48s} {ranked}")
        lines.append("")
    return lines


def format_separability(measured: Sequence[ItemScores]) -> list[str]:
    """Whether the two classes' top scores can be split by a threshold at all.

    This is the reading that decides whether a perfect score is even available, so it is
    stated before the sweep rather than left for a reader to infer from the winning row: any
    answerable item whose top score sits at or below the highest must-refuse top score cannot
    be admitted by *any* feasible `tau_top`, and the sweep's ceiling is therefore below 100%
    for a reason that no choice of threshold can fix.
    """
    refuse_tops = [item.top for item in measured if item.is_must_refuse and item.top is not None]
    answerable = [item for item in measured if not item.is_must_refuse and item.top is not None]
    if not refuse_tops or not answerable:
        return []

    highest_refuse = max(refuse_tops)
    blocked = sorted(
        (item for item in answerable if item.top is not None and item.top <= highest_refuse),
        key=lambda item: item.top or 0.0,
    )
    lines = [
        "separability of the two classes, by top score:",
        "",
        f"  highest must-refuse top score : {highest_refuse:.4f}"
        f"  ({_argmax_id(measured, must_refuse=True)})",
        f"  lowest answerable top score   : {min(item.top or 0.0 for item in answerable):.4f}"
        f"  ({_argmin_id(measured, must_refuse=False)})",
        "",
    ]
    if blocked:
        lines += [
            f"  The classes OVERLAP. {len(blocked)} answerable item(s) score at or below the",
            "  highest must-refuse item, so no feasible tau_top can admit them and a perfect",
            f"  {len(answerable)}/{len(answerable)} is unreachable at any threshold:",
            "",
        ]
        lines += [f"    {item.item_id:<48s} {item.top:.4f}" for item in blocked if item.top]
    else:
        lines.append("  The classes are separable: a feasible tau_top can admit every "
                     "answerable item.")
    lines.append("")
    return lines


def _argmax_id(measured: Sequence[ItemScores], *, must_refuse: bool) -> str:
    candidates = [
        item for item in measured if item.is_must_refuse is must_refuse and item.top is not None
    ]
    return max(candidates, key=lambda item: item.top or 0.0).item_id if candidates else "-"


def _argmin_id(measured: Sequence[ItemScores], *, must_refuse: bool) -> str:
    candidates = [
        item for item in measured if item.is_must_refuse is must_refuse and item.top is not None
    ]
    return min(candidates, key=lambda item: item.top or 0.0).item_id if candidates else "-"


def format_sweep(cells: Sequence[SweepCell]) -> list[str]:
    """The full sweep, every candidate pair, as a matrix.

    Section 8 requires the sweep itself to be published and not just its winner, so every
    swept pair appears: the losing candidates are what show that the recommendation sits on a
    plateau rather than on a spike, and that the rows below it fail for the stated reason.

    One row per `tau_top`, one column per `tau_support`, each cell the number of answerable
    items that pass there. `LEAK` marks a row where the hard constraint fails -- at least one
    must-refuse item's top score reached `tau_top` -- and those rows' pass counts are printed
    anyway, greyed by the marker rather than omitted, because "this threshold would score
    well if the constraint did not exist" is exactly the trade a reader is entitled to see
    being refused. A blank cell is a pair with `tau_support > tau_top`, which is not a
    threshold pair at all.
    """
    supports = sorted({cell.tau_support for cell in cells})
    by_pair = {(cell.tau_top, cell.tau_support): cell for cell in cells}
    tops = sorted({cell.tau_top for cell in cells})

    header = "  tau_top   must-refuse   " + " ".join(f"{support:>5.2f}" for support in supports)
    lines = [
        "full sweep -- answerable items passing assess_retrieval(), out of "
        f"{cells[0].answerable_total if cells else 0}:",
        "",
        "  (columns are tau_support; LEAK = a must-refuse item reached tau_top, "
        "so the pair is rejected)",
        "",
        header,
        "  " + "-" * (len(header) - 2),
    ]
    for tau_top in tops:
        row_cells = [by_pair.get((tau_top, support)) for support in supports]
        present = [cell for cell in row_cells if cell is not None]
        if not present:
            continue
        marker = "OK  " if present[0].feasible else "LEAK"
        refuse = f"{present[0].refuse_below}/{present[0].refuse_total} {marker}"
        counts = " ".join(
            f"{cell.answerable_passed:>5d}" if cell is not None else "     "
            for cell in row_cells
        )
        lines.append(f"  {tau_top:>7.2f}   {refuse:<11s}   {counts}")
    lines.append("")
    return lines


def format_recommendation(
    best: SweepCell | None, measured: Sequence[ItemScores]
) -> list[str]:
    """The recommendation, or the plain statement that there is none."""
    if best is None:
        return [
            "RECOMMENDATION: none.",
            "",
            "  No pair in the swept range keeps every must-refuse item below tau_top, so",
            "  there is no threshold to recommend. Deliberately not falling back to a",
            "  best-effort pair: the hard constraint mirrors Section 8's gate 1, which is",
            "  scored individually and never averaged, and a pair that violates it would be",
            "  a threshold nobody measured wearing a calibration's clothes.",
            "",
            "  Widening TAU_TOP_CANDIDATES is not the fix to reach for first -- a corpus in",
            "  which out-of-corpus questions retrieve as strongly as answerable ones is a",
            "  statement about the corpus and the embedding model, not about the threshold.",
            "",
        ]
    return [
        "RECOMMENDATION:",
        "",
        f"  tau_top     = {best.tau_top:.2f}   (currently {TAU_TOP})",
        f"  tau_support = {best.tau_support:.2f}   (currently {TAU_SUPPORT})",
        "",
        f"  must-refuse below tau_top : {best.refuse_below}/{best.refuse_total}  "
        "(the hard constraint, satisfied)",
        f"  answerable passing        : {best.answerable_passed}/{best.answerable_total}",
        f"  margin to nearest class   : {_balance(best, measured):.4f}",
        "",
        "  This script does not apply these values. Editing TAU_TOP/TAU_SUPPORT in",
        "  src/agent/critic/retrieval_confidence.py is a separate, reviewed change.",
        "",
    ]


def format_report(
    measured: Sequence[ItemScores],
    cells: Sequence[SweepCell],
    best: SweepCell | None,
    *,
    limit: int,
    collection_name: str,
    point_count: int | None,
) -> str:
    indexed = f"{point_count} chunks" if point_count is not None else "unknown chunk count"
    lines = [
        "docs/agent_design.md Section 6 / Section 8 -- TAU_TOP / TAU_SUPPORT calibration",
        f"  corpus items swept : {len(measured)} "
        f"({sum(1 for item in measured if not item.is_must_refuse)} answerable, "
        f"{sum(1 for item in measured if item.is_must_refuse)} must-refuse)",
        f"  k (DEFAULT_LIMIT)  : {limit}",
        f"  index              : {collection_name}, {indexed}",
        f"  min_supporting     : {MIN_SUPPORTING_CHUNKS} (Section 6 fixes this; not calibrated)",
        f"  candidate pairs    : {len(cells)}",
        "",
    ]
    lines += format_measurements(measured)
    lines += format_separability(measured)
    lines += format_sweep(cells)
    lines += format_recommendation(best, measured)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


def _point_count(qdrant_url: str, collection_name: str) -> int | None:
    """How many chunks the queried collection holds, for the header. `None` if unreadable.

    Reported rather than asserted: a calibration run against a differently built index is
    still a real measurement, it is just a measurement of a different corpus, and the number
    is what lets a reader tell the two apart.
    """
    try:
        from qdrant_client import QdrantClient

        return int(QdrantClient(url=qdrant_url).count(collection_name).count)
    except Exception:  # noqa: BLE001 -- a header detail must never fail the calibration
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--qdrant-url", default=DEFAULT_QDRANT_URL, help="URL of the Qdrant to query"
    )
    parser.add_argument(
        "--collection", default=COLLECTION_NAME, help="collection holding the indexed corpus"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="k for each search; defaults to production's DEFAULT_LIMIT",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    measured = collect_scores(
        limit=args.limit, qdrant_url=args.qdrant_url, collection_name=args.collection
    )
    cells = sweep(measured)
    best = recommend(cells, measured)
    print(
        format_report(
            measured,
            cells,
            best,
            limit=args.limit,
            collection_name=args.collection,
            point_count=_point_count(args.qdrant_url, args.collection),
        )
    )
    return 0 if best is not None else 1


if __name__ == "__main__":
    sys.exit(main())
