"""Tier-1 tests for the pure, non-network parts of `src/agent/rag/index.py`: point-ID
determinism and loader composition. `build_index()` itself needs a live Qdrant and downloads
model weights on first use, so it is exercised manually (see the PR for issue #98), not
here -- importing this module is safe without either (fastembed/qdrant-client's own imports
touch no network; only instantiating `TextEmbedding`/calling the server does).

Issue #110 added the read half (`src/agent/rag/retrieval.py`); the same split applies to it,
so what is tested here is the pure payload-to-`RetrievedChunk` conversion, not the query.
"""
from __future__ import annotations

from src.agent.rag.index import _point_id, default_loaders
from src.agent.rag.retrieval import _to_chunk
from src.agent.rag.loaders.decision_doc import DecisionDocLoader
from src.agent.rag.loaders.public_reference import PublicReferenceLoader
from src.agent.rag.schema import Chunk, ChunkMetadata


def _make_chunk(source_id: str, chunk_index: int) -> Chunk:
    return Chunk(
        text="body",
        metadata=ChunkMetadata(
            source_type="decision_doc",
            source_id=source_id,
            source_ref=source_id,
            heading_path="doc.md",
            chunk_index=chunk_index,
            indexed_at="2026-01-01T00:00:00+00:00",
        ),
    )


def test_point_id_is_deterministic_across_calls():
    """Rebuilding the index from unchanged sources must reproduce the same point IDs
    (upsert, not accumulate) -- `_point_id` must be a pure function of the chunk's own
    `chunk_id`, not of `indexed_at` or anything else that changes between runs.
    """
    chunk_a = _make_chunk("docs/example.md", 2)
    chunk_b = _make_chunk("docs/example.md", 2)  # same source_id + chunk_index

    assert _point_id(chunk_a) == _point_id(chunk_b)


def test_point_id_differs_for_different_chunks():
    same_doc_different_index = _make_chunk("docs/example.md", 3)
    different_doc_same_index = _make_chunk("docs/other.md", 2)
    base = _make_chunk("docs/example.md", 2)

    assert _point_id(base) != _point_id(same_doc_different_index)
    assert _point_id(base) != _point_id(different_doc_same_index)


def test_default_loaders_is_both_launch_corpus_loaders():
    loaders = default_loaders()
    assert len(loaders) == 2
    assert any(isinstance(loader, DecisionDocLoader) for loader in loaders)
    assert any(isinstance(loader, PublicReferenceLoader) for loader in loaders)


def _index_payload(chunk: Chunk) -> dict:
    """The payload `build_index` upserts for a chunk, restated here so a change to either
    side of the write/read pair fails this test rather than passing silently."""
    return {
        "source_type": chunk.metadata.source_type,
        "source_id": chunk.metadata.source_id,
        "source_ref": chunk.metadata.source_ref,
        "heading_path": chunk.metadata.heading_path,
        "chunk_index": chunk.metadata.chunk_index,
        "text": chunk.text,
        "indexed_at": chunk.metadata.indexed_at,
    }


def test_retrieval_reconstructs_exactly_the_chunk_id_the_indexer_would_produce():
    """`chunk_id` is not a stored payload field -- `retrieval.py` rebuilds it from
    `source_id` and `chunk_index`. `docs/agent_design.md` Section 6 verifies citations by
    set membership over those strings, so a second, divergent id format would break
    grounding verification with no visible error (Issue #110).
    """
    chunk = _make_chunk("docs/model_training_decision.md", 7)

    retrieved = _to_chunk(_index_payload(chunk), score=0.5)

    assert retrieved.chunk_id == chunk.chunk_id == "docs/model_training_decision.md::7"


def test_retrieval_carries_every_field_the_grounding_check_needs():
    chunk = _make_chunk("docs/example.md", 3)

    retrieved = _to_chunk(_index_payload(chunk), score=0.42)

    assert retrieved.source_type == "decision_doc"
    assert retrieved.source_ref == "docs/example.md"
    assert retrieved.heading_path == "doc.md"
    # Verbatim, so Section 6's numeric-fidelity check has the real characters to match.
    assert retrieved.text == chunk.text
    assert retrieved.score == 0.42
