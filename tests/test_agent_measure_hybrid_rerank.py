"""Tier-1 tests for the hybrid-rerank measurement (Issue #175).

No model, no network, no Qdrant -- Section 8 tier 1's hard requirement. Every pool below is
built by hand, so the reading logic is exercised without an index. `collect_candidates` is the
one function not covered here: it is the thin layer that calls `search()`, and it needs the
real collection by definition.

What is worth pinning is not the arithmetic but the two places a measurement of this shape can
quietly lie about itself: a "before" column that is not actually the vector-only baseline, and
a precision mean that a defined-nowhere item silently drags down.
"""
from __future__ import annotations

import pytest

from src.agent.rag.retrieval import RetrievedChunk
from tests.fixtures.golden_set_runner import MUST_REFUSE
from tests.fixtures.measure_hybrid_rerank import (
    WEIGHT_CANDIDATES,
    ItemCandidates,
    format_before_after,
    format_displacement,
    format_pool_reach,
    measure,
    raw_fusion_top,
)

ANSWERABLE = "Answerable from the docs"


def _chunk(chunk_id: str, score: float, text: str) -> RetrievedChunk:
    source_id, _, chunk_index = chunk_id.partition("::")
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_type="decision_doc",
        source_id=source_id,
        source_ref=source_id,
        heading_path="",
        chunk_index=int(chunk_index),
        text=text,
        score=score,
    )


def _pool(
    item_id: str,
    *,
    question: str = "drift baseline",
    relevant: tuple[str, ...] = (),
    category: str = ANSWERABLE,
    candidates: tuple[RetrievedChunk, ...] = (),
) -> ItemCandidates:
    return ItemCandidates(
        item_id=item_id,
        category=category,
        question=question,
        relevant=frozenset(relevant),
        candidates=candidates,
    )


def _ladder(*specs: tuple[str, float, str]) -> tuple[RetrievedChunk, ...]:
    return tuple(_chunk(chunk_id, score, text) for chunk_id, score, text in specs)


# --- The baseline column ----------------------------------------------------------------


def test_the_before_column_is_the_vector_ranking_untouched():
    """`weight=None` must mean "the top k by similarity", not "the reranker at zero" -- they
    agree today, and a reading whose baseline moved with the thing being swept would compare
    the change against itself."""
    pool = _pool(
        "item",
        relevant=("doc.md::9",),
        candidates=_ladder(
            *((f"doc.md::{index}", 0.90 - 0.01 * index, "unrelated") for index in range(9)),
            ("doc.md::9", 0.50, "drift baseline"),
        ),
    )

    assert [chunk.chunk_id for chunk in pool.top(None, limit=3)] == [
        "doc.md::0",
        "doc.md::1",
        "doc.md::2",
    ]


def test_a_relevant_chunk_outside_the_pool_is_unreachable_at_every_weight():
    """The ceiling `format_pool_reach` exists to state: reranking can only reorder what was
    retrieved, so a label the pool never saw is a miss no weight can fix."""
    pool = _pool(
        "item",
        relevant=("doc.md::99",),
        candidates=_ladder(("doc.md::0", 0.80, "drift baseline")),
    )

    assert pool.reachable == 0
    assert pool.deepest_relevant_rank is None
    assert measure([pool], 1.0, limit=5).recall_hits == 0


def test_pool_reach_reports_the_deepest_relevant_rank():
    """The number `INTERNAL_CANDIDATE_LIMIT` is sized from -- printed rather than inferred, so
    a corpus that starts burying its relevant chunks deeper shows up as a moving figure."""
    pool = _pool(
        "item",
        relevant=("doc.md::0", "doc.md::3"),
        candidates=_ladder(
            ("doc.md::0", 0.90, "drift baseline"),
            ("doc.md::1", 0.85, "unrelated"),
            ("doc.md::2", 0.80, "unrelated"),
            ("doc.md::3", 0.75, "drift baseline"),
        ),
    )

    assert pool.deepest_relevant_rank == 3
    assert any("deepest relevant chunk" in line for line in format_pool_reach([pool]))


# --- What each half of the corpus is scored on ------------------------------------------


def test_must_refuse_items_are_scored_on_the_mirror_metric_only():
    """They declare no relevant chunks, so recall@k is not their question (Section 8). They
    must not appear in the recall denominator, and their mirror reading must still be taken."""
    scored = _pool(
        "answerable",
        relevant=("doc.md::0",),
        candidates=_ladder(("doc.md::0", 0.80, "drift baseline")),
    )
    refusing = _pool(
        "refuse",
        category=MUST_REFUSE,
        candidates=_ladder(("other.md::0", 0.60, "unrelated")),
    )

    reading = measure([scored, refusing], None, limit=5)

    assert reading.recall_total == 1
    assert reading.mirror_total == 1
    assert reading.mirror_below == 1


def test_a_must_refuse_item_whose_top_hit_clears_tau_is_counted_as_a_leak():
    refusing = _pool(
        "refuse",
        category=MUST_REFUSE,
        candidates=_ladder(("other.md::0", 0.99, "unrelated")),
    )

    assert measure([refusing], None, limit=5).mirror_below == 0


def test_precision_excludes_items_that_retrieved_nothing():
    """`precision_at_k` is `None` for an empty ranked list, and folding that in as 0.0 would
    drag the mean down with an item that measured nothing at all."""
    empty = _pool("empty", relevant=("doc.md::0",))
    full = _pool(
        "full",
        relevant=("doc.md::0",),
        candidates=_ladder(("doc.md::0", 0.80, "drift baseline")),
    )

    reading = measure([empty, full], None, limit=5)

    assert reading.mean_precision == pytest.approx(1.0)
    assert reading.recall_total == 2
    assert reading.recall_hits == 1


# --- The rejected alternative ------------------------------------------------------------


def test_raw_fusion_needs_a_far_higher_weight_to_move_anything():
    """`_normalized`'s claim, made checkable: cosine similarities within one pool span a few
    hundredths while containment spans the unit interval, so an unnormalized sum at a small
    weight cannot reorder what a normalized one already does."""
    pool = _pool(
        "item",
        candidates=_ladder(
            ("doc.md::0", 0.80, "unrelated"),
            ("doc.md::1", 0.79, "drift baseline"),
        ),
    )

    assert [chunk.chunk_id for chunk in raw_fusion_top(pool, 0.05, limit=2)] == [
        "doc.md::1",
        "doc.md::0",
    ]
    assert [chunk.chunk_id for chunk in pool.top(0.05, limit=2)] == ["doc.md::0", "doc.md::1"]


def test_displacement_separates_same_document_neighbours_from_other_documents():
    """The reading `docs/agent_design.md` Section 3's addendum quotes, so it is pinned: a slot
    held by another chunk of the *same* document is the granularity failure, and one held by a
    different document is the ranking failure. Counting them together would hide which."""
    pool = _pool(
        "item",
        relevant=("doc.md::0", "doc.md::9"),
        candidates=_ladder(
            ("doc.md::0", 0.90, "drift baseline"),
            ("doc.md::1", 0.85, "neighbouring chunk"),
            ("other.md::0", 0.80, "different document"),
            ("doc.md::9", 0.50, "drift baseline"),
        ),
    )

    block = "\n".join(format_displacement([pool], limit=3))

    assert "relevant chunks missed        : 1" in block
    assert "same document : 1" in block
    assert "different document               : 1" in block


# --- The grid and the report -------------------------------------------------------------


def test_the_swept_grid_covers_both_ends_and_is_fine_near_zero():
    """The plateau of weights that change nothing is only visible at a step small enough to
    see it end, and the far end is what shows the trade being refused rather than assumed."""
    assert WEIGHT_CANDIDATES[0] == 0.0
    assert WEIGHT_CANDIDATES[-1] == 1.0
    assert 0.08 in WEIGHT_CANDIDATES
    assert sorted(set(WEIGHT_CANDIDATES)) == list(WEIGHT_CANDIDATES)


def test_the_before_after_block_names_both_rows_and_the_ceiling():
    pool = _pool(
        "item",
        relevant=("doc.md::0", "doc.md::1"),
        candidates=_ladder(
            ("doc.md::0", 0.80, "drift baseline"),
            ("doc.md::1", 0.70, "drift baseline"),
        ),
    )
    before = measure([pool], None, limit=5)
    after = measure([pool], 0.05, limit=5)

    block = "\n".join(format_before_after(before, after, limit=5))

    assert "vector only" in block
    assert "w=0.05" in block
    # 2 relevant chunks over 1 item at k=5 -- the labels cannot allow more than 0.400.
    assert "0.400" in block
