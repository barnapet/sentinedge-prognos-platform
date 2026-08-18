"""Is "no evidence at all" separable from "a borderline match"? (Issue #177)

    python -m tests.fixtures.measure_no_evidence_floor

`src/agent/critic/grounding.py`'s `_tier()` has one below-threshold outcome: `partial`. So a
must-refuse question that retrieved three tangential chunks, cited one, and made
numerically-faithful statements about it is *released* with a recommendation, and Section 8's
must-refuse gate -- "every one of the 8 items must pass individually, 100%, no aggregate" --
cannot be met by the estimator, the prompt, or the thresholds. Issue #177 asks whether there
is a **narrower** regime inside `below_threshold` that can be refused outright without
refusing the borderline-but-real matches that legitimately earn `partial` today.

This script is the measurement that question needs, and it is deliberately capable of
answering "no". `docs/agent_design.md` Section 6 fixed the procedure for a threshold in this
package -- measure against Section 8's golden set, publish the reading, and report a null
result as a null result (#173/PR #174's precedent) -- and the honest outcome here is either a
separation with a stated margin or the plain statement that the two classes overlap on every
axis measured, in which case nothing should be implemented.

**No Anthropic API call, anywhere.** The free path `tests/fixtures/calibrate_retrieval.py`
and `tests/fixtures/measure_hybrid_rerank.py` already use: each golden item's own `question`
goes straight to `search()` and the returned similarities are read. `collect_scores` is
imported from `calibrate_retrieval` rather than re-implemented, so both scripts are looking at
the same numbers retrieved the same way; a second copy of that call could drift from it.

**It changes no production constant, and the condition it evaluates introduces none.** The
axes swept below are read off `assess_retrieval`'s *existing* output -- `top_score` and
`supporting_count` -- at the calibrated `TAU_TOP`/`TAU_SUPPORT`/`MIN_SUPPORTING_CHUNKS`. That
is the point: a new number would be a threshold nobody calibrated, and #163 already fixed how
a threshold in this module gets one.

**It lives under `tests/` for Issue #122's reason**, the same one `golden_set_runner.py`,
`calibrate_retrieval.py` and `measure_hybrid_rerank.py` record: the golden set is test
infrastructure, and no module under `src/agent/` may know `tests/fixtures/golden_set_corpus.py`
exists. The dependency runs one way only.

## What it needs

A Qdrant carrying the `prognos_docs` collection, which is what `search()` queries:

    docker compose --profile agent up -d qdrant
    python -m src.agent.rag.index

The collection's point count is printed in the header. A run against a differently built index
is measuring a different corpus, so a reader comparing two runs needs to see that rather than
assume it.
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
from src.agent.rag.retrieval import DEFAULT_LIMIT
from tests.fixtures.calibrate_retrieval import ItemScores, collect_scores

# The candidate conditions, each a predicate on `assess_retrieval`'s own output. Named here
# rather than inline so the report can say which one it is rejecting and why, and so a reader
# can see that the list is short and exhaustive rather than a search that stopped when it
# found something agreeable.
#
# `supporting_count == 0` and `supporting_count < MIN_SUPPORTING_CHUNKS` are the only two
# conditions expressible from the existing constants that are *strictly stronger* than
# `below_threshold` itself. Anything else -- a second similarity threshold between
# `TAU_SUPPORT` and `TAU_TOP`, a rule on the spread of the returned scores -- introduces a
# number this project has not calibrated, which is exactly what #163 wrote its procedure to
# prevent.
CANDIDATE_CONDITIONS: tuple[tuple[str, str], ...] = (
    ("supporting_count == 0", "not one chunk reached TAU_SUPPORT"),
    (
        f"supporting_count < {MIN_SUPPORTING_CHUNKS}",
        "fewer than MIN_SUPPORTING_CHUNKS chunks reached TAU_SUPPORT",
    ),
)


@dataclass(frozen=True)
class Reading:
    """One item as the two axes a tier decision could read it on.

    `top` and `supporting` come from `assess_retrieval` itself, not from a re-derivation:
    the production predicate sorts the scores and applies the floors, and a reading computed
    beside it could agree with itself while disagreeing with what actually gates a turn.
    """

    item_id: str
    is_must_refuse: bool
    top: float | None
    supporting: int
    passed: bool

    @property
    def below_threshold(self) -> bool:
        return self.top is not None and not self.passed

    def fires(self, condition: str) -> bool:
        """Whether the named candidate condition would refuse this turn."""
        if not self.below_threshold:
            return False
        if condition == "supporting_count == 0":
            return self.supporting == 0
        return self.supporting < MIN_SUPPORTING_CHUNKS


def read(measured: Sequence[ItemScores]) -> tuple[Reading, ...]:
    """Every measured item, through production's own `assess_retrieval`."""
    readings = []
    for item in measured:
        confidence = assess_retrieval(item.scores)
        readings.append(
            Reading(
                item_id=item.item_id,
                is_must_refuse=item.is_must_refuse,
                top=confidence.top_score,
                supporting=confidence.supporting_count,
                passed=confidence.passed,
            )
        )
    return tuple(readings)


@dataclass(frozen=True)
class ConditionResult:
    """How one candidate condition splits the two classes.

    `false_refusals` is the number that decides it. A condition that fires on even one
    answerable item is refusing a question the corpus genuinely covers, and Section 6's
    "it never answers un-grounded" is not a licence to refuse what it can source.
    """

    condition: str
    description: str
    caught: int
    refuse_total: int
    false_refusals: int
    answerable_total: int
    nearest_answerable: float | None
    highest_caught_top: float | None

    @property
    def clean(self) -> bool:
        return self.caught > 0 and self.false_refusals == 0

    @property
    def margin(self) -> float | None:
        """Similarity units between the highest-scoring item the condition catches and the
        lowest-scoring answerable item it spares. `None` when either side is empty."""
        if self.nearest_answerable is None or self.highest_caught_top is None:
            return None
        return self.nearest_answerable - self.highest_caught_top


def evaluate(readings: Sequence[Reading], condition: str, description: str) -> ConditionResult:
    refusing = [r for r in readings if r.is_must_refuse]
    answerable = [r for r in readings if not r.is_must_refuse]
    caught = [r for r in refusing if r.fires(condition)]
    false_refusals = [r for r in answerable if r.fires(condition)]
    spared = [r for r in answerable if not r.fires(condition) and r.top is not None]
    return ConditionResult(
        condition=condition,
        description=description,
        caught=len(caught),
        refuse_total=len(refusing),
        false_refusals=len(false_refusals),
        answerable_total=len(answerable),
        nearest_answerable=min((r.top for r in spared if r.top is not None), default=None),
        highest_caught_top=max((r.top for r in caught if r.top is not None), default=None),
    )


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def format_readings(readings: Sequence[Reading]) -> list[str]:
    """Every item on both axes -- the evidence every line below is computed from.

    Printed in full rather than summarized, for `calibrate_retrieval.py`'s reason: a reader
    checking this should be able to recompute any conclusion by hand from these numbers.
    """
    lines = [
        f"measured at TAU_TOP={TAU_TOP}, TAU_SUPPORT={TAU_SUPPORT}, "
        f"MIN_SUPPORTING_CHUNKS={MIN_SUPPORTING_CHUNKS}:",
        "",
        f"  {'item':<48s} {'top':>7s} {'>=TAU_SUPPORT':>14s}  regime",
        "  " + "-" * 78,
    ]
    for wanted, label in ((True, "must refuse"), (False, "answerable from the docs")):
        lines.append(f"  {label}:")
        for reading in sorted(
            (r for r in readings if r.is_must_refuse is wanted),
            key=lambda r: -(r.top or 0.0),
        ):
            regime = (
                "passes"
                if reading.passed
                else ("NO EVIDENCE" if reading.supporting == 0 else "below_threshold")
            )
            top = f"{reading.top:.4f}" if reading.top is not None else "(no hits)"
            lines.append(
                f"    {reading.item_id:<46s} {top:>7s} {reading.supporting:>14d}  {regime}"
            )
        lines.append("")
    return lines


def format_top_score_axis(readings: Sequence[Reading]) -> list[str]:
    """The axis that does *not* separate, stated first.

    #163 already published this overlap for `TAU_TOP`; it is restated here because it is the
    reason a second threshold on the top score cannot be the answer, and a reader should not
    have to take that on trust from another document.
    """
    refuse_tops = [r.top for r in readings if r.is_must_refuse and r.top is not None]
    answerable_tops = [r.top for r in readings if not r.is_must_refuse and r.top is not None]
    if not refuse_tops or not answerable_tops:
        return []
    highest_refuse, lowest_answerable = max(refuse_tops), min(answerable_tops)
    above = [t for t in refuse_tops if t >= lowest_answerable]
    lines = [
        "axis 1 -- top score alone:",
        "",
        f"  highest must-refuse top : {highest_refuse:.4f}",
        f"  lowest answerable top   : {lowest_answerable:.4f}",
        "",
    ]
    if above:
        lines += [
            f"  OVERLAP. {len(above)} must-refuse item(s) score at or above the lowest",
            "  answerable item, so no threshold on the top score can separate the classes --",
            "  the same finding #163 published for TAU_TOP, unchanged. A second, lower",
            "  similarity threshold is therefore not available as a refusal condition.",
        ]
    else:
        lines.append("  The classes are separable by top score alone.")
    lines.append("")
    return lines


def format_conditions(results: Sequence[ConditionResult]) -> list[str]:
    """Each candidate condition's split, with the margin that decides between them."""
    lines = ["axis 2 -- how many chunks reached TAU_SUPPORT:", ""]
    for result in results:
        lines.append(f"  {result.condition}  ({result.description})")
        lines.append(
            f"    must-refuse caught  : {result.caught}/{result.refuse_total}"
        )
        lines.append(
            f"    answerable refused  : {result.false_refusals}/{result.answerable_total}"
            f"{'' if result.false_refusals == 0 else '   <-- refuses a sourceable question'}"
        )
        if result.margin is not None:
            lines.append(
                f"    margin              : {result.margin:+.4f}  "
                f"(highest caught top {result.highest_caught_top:.4f}, "
                f"nearest spared answerable {result.nearest_answerable:.4f})"
            )
        lines.append("")
    return lines


def format_verdict(results: Sequence[ConditionResult]) -> list[str]:
    """The reading, stated as a verdict rather than left for the reader to assemble.

    Ordered by margin among the clean conditions, not by how many must-refuse items each
    catches. Catching one more item is worth nothing if the condition sits on top of the
    nearest answerable question: the cost of a false refusal here is refusing something the
    corpus covers, and #163's own tie-break is the same max-margin reasoning.
    """
    clean = [result for result in results if result.clean]
    if not clean:
        return [
            "VERDICT: no separation. Recommend implementing nothing.",
            "",
            "  Every candidate condition either fires on an answerable item or catches no",
            "  must-refuse item at all. A cutoff chosen anyway would be a number nobody",
            "  measured, and `_tier()` is the core safety decision -- the honest outcome is",
            "  the null result, not a best-effort rule.",
            "",
        ]
    best = max(clean, key=lambda result: (result.margin or 0.0))
    lines = [
        f"VERDICT: `{best.condition}` is the condition to implement.",
        "",
        f"  It refuses {best.caught} of {best.refuse_total} must-refuse items and 0 of "
        f"{best.answerable_total} answerable ones.",
        "",
        f"  It is NOT a fix for the whole must-refuse gate: {best.refuse_total - best.caught}"
        " item(s) retrieve strongly enough",
        "  to be outside it, and axis 1 above says why no condition reaches them -- they",
        "  score above an answerable item, so catching them means refusing that item too.",
    ]
    if best.margin is not None:
        lines += [
            "",
            f"  Margin {best.margin:+.4f} in similarity units, between the strongest item it",
            "  refuses and the weakest it spares. Read that as thin rather than comfortable:",
            "  #163 chose TAU_TOP with margins of 0.0177 below and 0.0099 above, and this is",
            "  smaller than either. What makes it defensible anyway is that it adds no new",
            "  number -- it reads TAU_SUPPORT, already calibrated, and the count it compares",
            "  against is zero, which is not a tunable.",
        ]
    rejected = [result for result in results if result is not best]
    if rejected:
        lines += ["", "  Rejected:"]
        for result in rejected:
            if result.false_refusals:
                why = f"refuses {result.false_refusals} answerable item(s)"
            elif result.margin is not None and result.margin < 0:
                why = (
                    f"margin {result.margin:+.4f}: it refuses an item scoring ABOVE the "
                    "answerable one it spares"
                )
            elif result.margin is not None:
                why = f"margin {result.margin:+.4f} is smaller"
            else:
                why = "catches nothing"
            lines.append(f"    {result.condition:<24s} {why}")
    lines.append("")
    return lines


def format_report(
    readings: Sequence[Reading],
    results: Sequence[ConditionResult],
    *,
    collection_name: str,
    point_count: int | None,
    limit: int,
) -> str:
    points = "unknown" if point_count is None else str(point_count)
    header = [
        "=" * 82,
        "Is \"no evidence at all\" separable from \"a borderline match\"?  (Issue #177)",
        "=" * 82,
        "",
        f"collection : {collection_name} ({points} points)",
        f"k          : {limit}",
        f"items      : {len(readings)} corpus-grounded golden-set items",
        "",
    ]
    return "\n".join(
        header
        + format_readings(readings)
        + format_top_score_axis(readings)
        + format_conditions(results)
        + format_verdict(results)
    )


def _point_count(qdrant_url: str, collection_name: str) -> int | None:
    """How many chunks the queried collection holds, for the header. `None` if unreadable."""
    try:
        from qdrant_client import QdrantClient

        return int(QdrantClient(url=qdrant_url).count(collection_name).count)
    except Exception:  # noqa: BLE001 -- a header detail must never fail the measurement
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
        help="k for the reading; defaults to production's DEFAULT_LIMIT",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    measured = collect_scores(
        limit=args.limit, qdrant_url=args.qdrant_url, collection_name=args.collection
    )
    readings = read(measured)
    results = [
        evaluate(readings, condition, description)
        for condition, description in CANDIDATE_CONDITIONS
    ]
    print(
        format_report(
            readings,
            results,
            collection_name=args.collection,
            point_count=_point_count(args.qdrant_url, args.collection),
            limit=args.limit,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
