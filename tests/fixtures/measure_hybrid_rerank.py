"""Measure the hybrid reranker against the golden set's corpus items (Issue #175).

    python -m tests.fixtures.measure_hybrid_rerank

`src/agent/rag/retrieval.py` now fetches `INTERNAL_CANDIDATE_LIMIT` candidates and reranks
them by `(1 - LEXICAL_WEIGHT) * vector + LEXICAL_WEIGHT * lexical` before returning
`DEFAULT_LIMIT`. Both of those constants are tunables, and `docs/agent_design.md` Section 8
fixes how a tunable in this project gets its value: sweep it against the golden set, publish
the sweep and not only its winner, and report a null result as a null result (Issue #173's
precedent, PR #174). This script is that sweep for `LEXICAL_WEIGHT`, and the before/after
the issue asks for.

**No Anthropic API call, anywhere.** Same free path `tests/fixtures/calibrate_retrieval.py`
uses: each item's own `question` text goes straight to `search()`, and the returned chunk ids
are scored against the `relevant_chunk_ids` Section 8 already declares. The model that writes
an answer is never involved, so the whole measurement costs nothing and can be re-run
whenever the corpus moves.

**What "before" and "after" mean here.** Both are computed from the *same* pool of candidates,
retrieved once per item:

- **before** -- the top `DEFAULT_LIMIT` by vector similarity alone, which is exactly what
  `search()` returned prior to Issue #175 (a `limit=5` query returns the same five chunks a
  `limit=20` query's first five are).
- **after** -- the top `DEFAULT_LIMIT` by the combined score, through the **real**
  `rerank()`, not a reimplementation of it. A reimplementation could agree with itself while
  disagreeing with what actually ranks a search.

The metrics themselves are `tests/fixtures/golden_set_retrieval.py`'s own `recall_at_k`,
`precision_at_k` and `top_score_below_tau`, for the same reason -- these are the functions
the harness reports Section 8's numbers with.

**It lives under `tests/` for Issue #122's reason**, the same one `golden_set_runner.py` and
`calibrate_retrieval.py` record: the golden set is test infrastructure, and no module under
`src/agent/` may know `tests/fixtures/golden_set_corpus.py` exists. The dependency runs one
way only.

## What it needs

A Qdrant carrying the `prognos_docs` collection:

    docker compose --profile agent up -d qdrant
    python -m src.agent.rag.index

The collection's point count is printed in the header. `relevant_chunk_ids` are stable only
while each file's chunk count and order are (Section 8's own stated maintenance cost), so a
run against a differently built index is measuring a different corpus and a reader comparing
two runs needs to see that.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Sequence

from src.agent.critic.retrieval_confidence import TAU_TOP
from src.agent.rag.index import COLLECTION_NAME, DEFAULT_QDRANT_URL
from src.agent.rag.lexical import candidate_text, lexical_overlap
from src.agent.rag.retrieval import (
    DEFAULT_LIMIT,
    INTERNAL_CANDIDATE_LIMIT,
    LEXICAL_WEIGHT,
    RetrievedChunk,
    rerank,
    search,
)
from tests.fixtures.golden_set_corpus import CORPUS_ITEMS
from tests.fixtures.golden_set_retrieval import (
    RankedChunk,
    SearchOutcome,
    precision_at_k,
    recall_at_k,
    top_score_below_tau,
)
from tests.fixtures.golden_set_runner import MUST_REFUSE

# The swept grid: every weight from pure vector to pure lexical, finely near zero. The fine
# end is where the answer lives -- the plateau of weights that change nothing is only visible
# at a step small enough to see it end -- and the coarse end is kept in the report because
# "what happens if the lexical term is trusted completely" is the trade a reader is entitled
# to see being refused rather than asserted.
WEIGHT_CANDIDATES: tuple[float, ...] = tuple(
    sorted({round(0.01 * step, 4) for step in range(16)} | {round(0.05 * step, 4) for step in range(21)})
)


@dataclass(frozen=True)
class ItemCandidates:
    """One item's candidate pool, retrieved once and reranked against many times.

    `search()` is called once per item and the chunks are kept, rather than re-queried per
    candidate weight: embedding the query and reaching Qdrant is the expensive part, the
    weight is applied to the pool afterwards, and re-querying would make one measurement look
    like thirty-six independent ones.
    """

    item_id: str
    category: str
    question: str
    relevant: frozenset[str]
    candidates: tuple[RetrievedChunk, ...]

    @property
    def is_must_refuse(self) -> bool:
        return self.category == MUST_REFUSE

    @property
    def reachable(self) -> int:
        """Relevant chunks present in the pool at all -- the ceiling any reranking can hit."""
        return sum(1 for chunk in self.candidates if chunk.chunk_id in self.relevant)

    @property
    def deepest_relevant_rank(self) -> int | None:
        """The vector rank of the deepest relevant chunk, which is what sizes the pool."""
        ranks = [
            index
            for index, chunk in enumerate(self.candidates)
            if chunk.chunk_id in self.relevant
        ]
        return max(ranks) if ranks else None

    def top(self, weight: float | None, *, limit: int = DEFAULT_LIMIT) -> list[RetrievedChunk]:
        """The returned `limit`, either by vector alone (`weight is None`) or reranked."""
        if weight is None:
            return list(self.candidates[:limit])
        return rerank(self.question, self.candidates, limit=limit, lexical_weight=weight)

    def outcome(self, weight: float | None, *, limit: int = DEFAULT_LIMIT) -> SearchOutcome:
        """The same `SearchOutcome` shape a real run's payloads produce, so Section 8's own
        metric functions can score it unmodified."""
        return SearchOutcome(
            ranked=tuple(
                RankedChunk(chunk.chunk_id, chunk.score) for chunk in self.top(weight, limit=limit)
            ),
            searched=True,
        )


def collect_candidates(
    *,
    candidate_limit: int = INTERNAL_CANDIDATE_LIMIT,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    collection_name: str = COLLECTION_NAME,
) -> tuple[ItemCandidates, ...]:
    """Retrieve one pool per corpus item, in the file's own order.

    Only `CORPUS_ITEMS` is read: the tool-grounded, inventory and similarity items reach
    their evidence through a live tool rather than the vector index, so a reranker measured
    on them would be measured on searches they never make.

    `search()` is asked for the whole pool and its result is then **re-sorted by vector
    score**, which restores the pure similarity ordering Qdrant returned. Without that, the
    pool would arrive in the order the *current* `LEXICAL_WEIGHT` put it in, and the stable
    sort inside `rerank()` would resolve every tie by that order -- making the "before"
    column a function of the value being swept.
    """
    return tuple(
        ItemCandidates(
            item_id=item.item_id,
            category=item.category,
            question=item.question,
            relevant=item.relevant_chunk_ids,
            candidates=tuple(
                sorted(
                    search(
                        item.question,
                        limit=candidate_limit,
                        qdrant_url=qdrant_url,
                        collection_name=collection_name,
                    ),
                    key=lambda chunk: -chunk.score,
                )
            ),
        )
        for item in CORPUS_ITEMS
    )


@dataclass(frozen=True)
class Reading:
    """One weight's full result over the 16 items, as Section 8 reports retrieval quality."""

    weight: float | None
    recall_hits: int
    recall_total: int
    relevant_retrieved: int
    relevant_total: int
    mean_precision: float | None
    mirror_below: int
    mirror_total: int

    @property
    def label(self) -> str:
        return "vector only" if self.weight is None else f"w={self.weight:.2f}"


def measure(
    pools: Sequence[ItemCandidates], weight: float | None, *, limit: int = DEFAULT_LIMIT
) -> Reading:
    """Score one weight against every item, through the production metric functions.

    The two halves are counted by different rules because they are different questions, the
    same split `golden_set_retrieval.py` already makes: an item declaring `relevant_chunk_ids`
    gets recall@k and precision@k; a must-refuse item gets the mirror metric (did the top
    similarity stay below `TAU_TOP`), for which recall@k is not the question.
    """
    scored = [pool for pool in pools if pool.relevant]
    refusing = [pool for pool in pools if pool.is_must_refuse]
    precisions = [
        value
        for value in (
            precision_at_k(pool.outcome(weight, limit=limit), pool.relevant, k=limit)
            for pool in scored
        )
        if value is not None
    ]
    return Reading(
        weight=weight,
        recall_hits=sum(
            1
            for pool in scored
            if recall_at_k(pool.outcome(weight, limit=limit), pool.relevant, k=limit)
        ),
        recall_total=len(scored),
        relevant_retrieved=sum(
            sum(1 for chunk in pool.top(weight, limit=limit) if chunk.chunk_id in pool.relevant)
            for pool in scored
        ),
        relevant_total=sum(len(pool.relevant) for pool in scored),
        mean_precision=sum(precisions) / len(precisions) if precisions else None,
        mirror_below=sum(
            1
            for pool in refusing
            if top_score_below_tau(pool.outcome(weight, limit=limit), tau_top=TAU_TOP)
        ),
        mirror_total=len(refusing),
    )


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def format_pool_reach(pools: Sequence[ItemCandidates]) -> list[str]:
    """How much of the answer the candidate pool can even see.

    Stated before any sweep, because it bounds every row of it: a relevant chunk outside the
    pool cannot be reranked into the answer at any weight, so this is where the ceiling comes
    from and how `INTERNAL_CANDIDATE_LIMIT` was sized.
    """
    scored = [pool for pool in pools if pool.relevant]
    reachable = sum(pool.reachable for pool in scored)
    total = sum(len(pool.relevant) for pool in scored)
    deepest = [pool.deepest_relevant_rank for pool in scored]
    lines = [
        f"candidate pool (INTERNAL_CANDIDATE_LIMIT = {INTERNAL_CANDIDATE_LIMIT}):",
        "",
        f"  relevant chunks inside the pool : {reachable}/{total}",
        f"  deepest relevant chunk, by vector rank : "
        f"{max(rank for rank in deepest if rank is not None)}",
        "",
        "  per item (vector rank of each relevant chunk within the pool):",
    ]
    for pool in scored:
        ranks = [
            index for index, chunk in enumerate(pool.candidates) if chunk.chunk_id in pool.relevant
        ]
        lines.append(
            f"    {pool.item_id:<52s} {len(ranks)}/{len(pool.relevant)} at ranks "
            f"{ranks if ranks else '-'}"
        )
    lines.append("")
    return lines


def format_before_after(before: Reading, after: Reading, *, limit: int) -> list[str]:
    """The headline the issue asks for: precision@5/recall@5, before and after."""

    def _precision(reading: Reading) -> str:
        return "n/a" if reading.mean_precision is None else f"{reading.mean_precision:.3f}"

    return [
        f"before / after at LEXICAL_WEIGHT = {LEXICAL_WEIGHT} (k = {limit}):",
        "",
        f"  {'':<14s} {'recall@' + str(limit):>12s} {'precision@' + str(limit):>14s} "
        f"{'relevant in top-' + str(limit):>20s} {'must-refuse mirror':>20s}",
        f"  {before.label:<14s} {str(before.recall_hits) + '/' + str(before.recall_total):>12s} "
        f"{_precision(before):>14s} "
        f"{str(before.relevant_retrieved) + '/' + str(before.relevant_total):>20s} "
        f"{str(before.mirror_below) + '/' + str(before.mirror_total):>20s}",
        f"  {after.label:<14s} {str(after.recall_hits) + '/' + str(after.recall_total):>12s} "
        f"{_precision(after):>14s} "
        f"{str(after.relevant_retrieved) + '/' + str(after.relevant_total):>20s} "
        f"{str(after.mirror_below) + '/' + str(after.mirror_total):>20s}",
        "",
        f"  precision@{limit}'s ceiling is below 1.0 by construction: the "
        f"{before.recall_total} scored items declare",
        f"  {before.relevant_total} relevant chunks between them, 2-4 each, so the best "
        f"reachable mean is {_ceiling(before):.3f}, not 1.000.",
        "",
    ]


def _ceiling(reading: Reading) -> float:
    """The highest mean precision@k the labels allow, for the header line above."""
    return reading.relevant_total / (reading.recall_total * DEFAULT_LIMIT)


def format_sweep(readings: Sequence[Reading], baseline: Reading) -> list[str]:
    """Every swept weight, not only the chosen one (Section 8's publish-the-sweep rule)."""
    lines = [
        "full sweep -- one row per candidate weight, scored on the same pools:",
        "",
        f"  {'weight':>7s} {'recall':>8s} {'precision':>10s} {'relevant':>9s} {'mirror':>7s}   "
        "vs vector-only",
        "  " + "-" * 62,
    ]
    for reading in readings:
        precision = "n/a" if reading.mean_precision is None else f"{reading.mean_precision:.3f}"
        if reading.mean_precision is None or baseline.mean_precision is None:
            delta = "     -"
        else:
            delta = f"{reading.mean_precision - baseline.mean_precision:+.3f}"
        recall_note = (
            "" if reading.recall_hits == baseline.recall_hits else "  RECALL LOST"
        )
        lines.append(
            f"  {reading.weight:>7.2f} "
            f"{str(reading.recall_hits) + '/' + str(reading.recall_total):>8s} "
            f"{precision:>10s} "
            f"{str(reading.relevant_retrieved) + '/' + str(reading.relevant_total):>9s} "
            f"{str(reading.mirror_below) + '/' + str(reading.mirror_total):>7s}   "
            f"{delta}{recall_note}"
        )
    lines.append("")
    return lines


def format_displacement(pools: Sequence[ItemCandidates], *, limit: int) -> list[str]:
    """What is holding the top-`limit` slots that no relevant chunk holds.

    The reason this is in the report at all: it says whether the missing precision is a
    *ranking* problem (the right chunks are in the pool and something outranked them on the
    wrong grounds) or a *granularity* problem (the slots are held by neighbouring chunks of
    the same document, which share the query's vocabulary and its subject by construction).
    A lexical term can do something about the first. It cannot do anything about the second,
    because the words it counts are the same words on both sides.
    """
    scored = [pool for pool in pools if pool.relevant]
    missed = 0
    same_document = 0
    other_document = 0
    for pool in scored:
        relevant_documents = {chunk_id.split("::")[0] for chunk_id in pool.relevant}
        top = pool.top(None, limit=limit)
        missed += len(pool.relevant) - sum(1 for chunk in top if chunk.chunk_id in pool.relevant)
        for chunk in top:
            if chunk.chunk_id in pool.relevant:
                continue
            if chunk.source_id in relevant_documents:
                same_document += 1
            else:
                other_document += 1
    return [
        f"what occupies the top-{limit} slots that no relevant chunk holds (vector ranking):",
        "",
        f"  relevant chunks missed        : {missed}",
        f"  slots held by another chunk of the same document : {same_document}",
        f"  slots held by a different document               : {other_document}",
        "",
    ]


def raw_fusion_top(
    pool: ItemCandidates, weight: float, *, limit: int = DEFAULT_LIMIT
) -> list[RetrievedChunk]:
    """The rejected alternative, kept measurable rather than described.

    `(1 - w) * vector + w * lexical` on the **unnormalized** scores -- Issue #175's "a
    weighted sum is fine", taken literally. `_normalized`'s docstring in
    `src/agent/rag/retrieval.py` claims this variant hands the ranking to the lexical term at
    a much lower weight, because cosine similarities within one pool span ~0.05 while
    containment spans the unit interval. Same convention as `src/features/
    candidate_features.py` and `src/training/candidate_scalers.py`: the arm that lost stays
    in the repo, computable, so the claim that it lost can be re-checked.
    """
    scored = sorted(
        range(len(pool.candidates)),
        key=lambda index: -(
            (1.0 - weight) * pool.candidates[index].score
            + weight
            * lexical_overlap(
                pool.question,
                candidate_text(
                    pool.candidates[index].heading_path, pool.candidates[index].text
                ),
            )
        ),
    )
    return [pool.candidates[index] for index in scored[:limit]]


def format_raw_comparison(
    pools: Sequence[ItemCandidates], baseline: Reading, *, limit: int
) -> list[str]:
    """Normalized fusion against raw fusion, at the same handful of weights."""
    scored = [pool for pool in pools if pool.relevant]
    lines = [
        "normalization: the same weights under raw (unnormalized) fusion, for comparison:",
        "",
        f"  {'weight':>7s} {'raw recall':>11s} {'raw precision':>14s}",
        "  " + "-" * 36,
    ]
    for weight in (0.05, 0.10, 0.20, 0.25, 0.50):
        hits = 0
        precisions = []
        for pool in scored:
            top = raw_fusion_top(pool, weight, limit=limit)
            outcome = SearchOutcome(
                ranked=tuple(RankedChunk(chunk.chunk_id, chunk.score) for chunk in top),
                searched=True,
            )
            hits += recall_at_k(outcome, pool.relevant, k=limit)
            value = precision_at_k(outcome, pool.relevant, k=limit)
            if value is not None:
                precisions.append(value)
        mean = sum(precisions) / len(precisions) if precisions else None
        lines.append(
            f"  {weight:>7.2f} {str(hits) + '/' + str(len(scored)):>11s} "
            f"{('n/a' if mean is None else f'{mean:.3f}'):>14s}"
        )
    baseline_precision = (
        "n/a" if baseline.mean_precision is None else f"{baseline.mean_precision:.3f}"
    )
    lines += [
        "",
        f"  (vector only, for reference: {baseline.recall_hits}/{baseline.recall_total} "
        f"recall, {baseline_precision} precision)",
        "",
    ]
    return lines


def format_signal_separation(pools: Sequence[ItemCandidates]) -> list[str]:
    """Whether the lexical term separates relevant from irrelevant *within* a pool.

    The sweep says what the reranking does; this says why. Containment can look informative
    in aggregate and still be useless for ranking, because the ranking question is only ever
    asked inside one item's pool -- where every candidate already came back for the same
    query and therefore already shares its vocabulary.
    """
    lines = [
        "lexical containment, relevant vs irrelevant, within each item's own pool:",
        "",
        f"  {'item':<52s} {'relevant':>9s} {'other':>8s} {'delta':>8s}",
        "  " + "-" * 80,
    ]
    for pool in pools:
        if not pool.relevant:
            continue
        scores = [
            (
                chunk.chunk_id in pool.relevant,
                lexical_overlap(
                    pool.question, candidate_text(chunk.heading_path, chunk.text)
                ),
            )
            for chunk in pool.candidates
        ]
        relevant = [value for is_relevant, value in scores if is_relevant]
        other = [value for is_relevant, value in scores if not is_relevant]
        if not relevant or not other:
            continue
        mean_relevant = sum(relevant) / len(relevant)
        mean_other = sum(other) / len(other)
        lines.append(
            f"  {pool.item_id:<52s} {mean_relevant:>9.3f} {mean_other:>8.3f} "
            f"{mean_relevant - mean_other:>+8.3f}"
        )
    lines.append("")
    return lines


def format_report(
    pools: Sequence[ItemCandidates],
    readings: Sequence[Reading],
    before: Reading,
    after: Reading,
    *,
    limit: int,
    collection_name: str,
    point_count: int | None,
) -> str:
    indexed = f"{point_count} chunks" if point_count is not None else "unknown chunk count"
    lines = [
        "docs/agent_design.md Section 8 -- hybrid reranking, before/after and weight sweep",
        f"  corpus items       : {len(pools)} "
        f"({sum(1 for pool in pools if pool.relevant)} with relevant_chunk_ids, "
        f"{sum(1 for pool in pools if pool.is_must_refuse)} must-refuse)",
        f"  k (DEFAULT_LIMIT)  : {limit}",
        f"  candidate pool     : {INTERNAL_CANDIDATE_LIMIT}",
        f"  index              : {collection_name}, {indexed}",
        f"  tau_top            : {TAU_TOP} (the mirror metric's threshold; not swept here)",
        "",
    ]
    lines += format_pool_reach(pools)
    lines += format_before_after(before, after, limit=limit)
    lines += format_sweep(readings, before)
    lines += format_raw_comparison(pools, before, limit=limit)
    lines += format_displacement(pools, limit=limit)
    lines += format_signal_separation(pools)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


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
        help="k for every reading; defaults to production's DEFAULT_LIMIT",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=INTERNAL_CANDIDATE_LIMIT,
        help="internal candidate pool size; defaults to production's INTERNAL_CANDIDATE_LIMIT",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pools = collect_candidates(
        candidate_limit=args.candidates,
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
    )
    before = measure(pools, None, limit=args.limit)
    after = measure(pools, LEXICAL_WEIGHT, limit=args.limit)
    readings = [measure(pools, weight, limit=args.limit) for weight in WEIGHT_CANDIDATES]
    print(
        format_report(
            pools,
            readings,
            before,
            after,
            limit=args.limit,
            collection_name=args.collection,
            point_count=_point_count(args.qdrant_url, args.collection),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
