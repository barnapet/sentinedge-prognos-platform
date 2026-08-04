"""Tier-1 tests for the pure, non-network parts of `src/agent/rag/index.py`: point-ID
determinism and loader composition. `build_index()` itself needs a live Qdrant and downloads
model weights on first use, so it is exercised manually (see the PR for issue #98), not
here -- importing this module is safe without either (fastembed/qdrant-client's own imports
touch no network; only instantiating `TextEmbedding`/calling the server does).
"""
from __future__ import annotations

from src.agent.rag.index import _point_id, default_loaders
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
