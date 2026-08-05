"""Query side of the `prognos_docs` collection (Issue #110, `docs/agent_design.md`
Sections 3 and 4).

Issue #98/#99 built the *write* half -- loaders, chunking, and `index.py`'s upsert -- and
stopped there, because nothing consumed the index yet. `search_documentation`
(`docs/agent_design.md` Section 2) is the first consumer, so the read half lands here
rather than inside the MCP tool: the tool layer wraps a backing module the same way
`check_inventory` wraps `src/agent/inventory/query.py`, and this module knows nothing about
MCP.

Section 3's collection schema is the only contract this shares with `index.py`: same
collection name, same embedding model, same payload fields. `chunk_id` is not a payload
field -- it is reconstructed here exactly as `Chunk.chunk_id` builds it
(`source_id::chunk_index`), so an id returned by a search is the same string the indexer
would produce for that chunk. Section 6's citation-existence check compares those strings,
so a second, divergent id format would silently break grounding verification.

The embedding model is constructed lazily and cached per process: `fastembed` loads an
ONNX model on first use (~seconds, and a download if the model is not already cached), and
paying that at import time would make every `import` of the MCP server slow -- including
the ones that never search.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.agent.rag.index import (
    COLLECTION_NAME,
    DEFAULT_QDRANT_URL,
    EMBEDDING_MODEL_NAME,
)

DEFAULT_LIMIT = 5

_embedder: Any = None


def _get_embedder() -> Any:
    """The `fastembed` model, built once per process. Same `threads=1` sizing note as
    `index.py`: the default thread pool scales with available cores, not with how small
    this corpus is."""
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding

        _embedder = TextEmbedding(model_name=EMBEDDING_MODEL_NAME, threads=1)
    return _embedder


@dataclass(frozen=True)
class RetrievedChunk:
    """One search hit, carrying everything Section 6's grounding contract needs: the
    stable `chunk_id` to cite, the `source_type`/`source_ref` to attribute it to, and the
    verbatim `text` its numeric-fidelity check reads."""

    chunk_id: str
    source_type: str
    source_id: str
    source_ref: str
    heading_path: str
    chunk_index: int
    text: str
    score: float


def _to_chunk(payload: dict, score: float) -> RetrievedChunk:
    source_id = str(payload["source_id"])
    chunk_index = int(payload["chunk_index"])
    return RetrievedChunk(
        # Rebuilt, not stored -- must match `Chunk.chunk_id` exactly (see module docstring).
        chunk_id=f"{source_id}::{chunk_index}",
        source_type=str(payload["source_type"]),
        source_id=source_id,
        source_ref=str(payload["source_ref"]),
        heading_path=str(payload["heading_path"]),
        chunk_index=chunk_index,
        text=str(payload["text"]),
        score=float(score),
    )


def search(
    query: str,
    limit: int = DEFAULT_LIMIT,
    source_type: str | None = None,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    collection_name: str = COLLECTION_NAME,
) -> list[RetrievedChunk]:
    """Return the `limit` nearest chunks to `query`, most similar first.

    `source_type` filters on the indexed payload field (Section 3: "`source_type` is an
    indexed payload field so it is filterable") -- the mechanism that lets a later
    `loaders/manual.py` be retrieved separately without a schema change.

    Raises whatever the embedder or the Qdrant client raises; the MCP tool layer is what
    turns a failure into a tool result (`docs/agent_design.md` Section 2), not this module.
    """
    vector = next(iter(_get_embedder().embed([query])))
    query_filter = (
        Filter(must=[FieldCondition(key="source_type", match=MatchValue(value=source_type))])
        if source_type is not None
        else None
    )
    client = QdrantClient(url=qdrant_url)
    response = client.query_points(
        collection_name=collection_name,
        query=vector.tolist(),
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    )
    return [_to_chunk(point.payload or {}, point.score) for point in response.points]
