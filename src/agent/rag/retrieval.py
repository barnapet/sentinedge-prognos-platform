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

**Retrieve-then-rerank (Issue #175).** `search()` asks Qdrant for `INTERNAL_CANDIDATE_LIMIT`
candidates, scores each one lexically as well as by vector similarity, and returns the best
`limit` by the combination. The public signature, `DEFAULT_LIMIT`, and the `RetrievedChunk`
shape are all unchanged -- every caller (the MCP tool, `golden_set_retrieval.py`,
`calibrate_retrieval.py`) sees the same contract, the same fields, and the same number of
results as before. What changes is *which* chunks and in what order. See `_combined_scores`
for the weighting and `score` below for the one thing that deliberately did **not** become
the combined number.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.agent.rag.index import (
    COLLECTION_NAME,
    DEFAULT_QDRANT_URL,
    EMBEDDING_MODEL_NAME,
)
from src.agent.rag.lexical import candidate_text, lexical_overlap

DEFAULT_LIMIT = 5

# How many candidates the vector search fetches internally, before reranking down to
# `limit`. Chosen from measurement rather than convention: against the committed corpus,
# **all 23 chunks the golden set declares relevant sit inside the top 20** by vector
# similarity, the deepest at rank 15 (`tests/fixtures/measure_hybrid_rerank.py`). 15 would
# therefore reach every one of them with no margin at all, and a pool that cannot see a
# chunk can never rerank it into the answer. 20 keeps that headroom while staying at the
# top of Issue #175's suggested 15-20 range; going deeper adds candidates whose similarity
# is already well below the returned band, at one payload fetch and one tokenization each.
#
# `limit` wins when a caller asks for more than this, so a caller requesting 30 results
# still gets 30 rather than being silently capped at the reranking pool.
INTERNAL_CANDIDATE_LIMIT = 20

# How much of the ranking score the lexical containment term carries, against `1 -
# LEXICAL_WEIGHT` for the vector term (both min-max normalized within the candidate pool
# first -- see `_combined_scores`).
#
# **0.05 is chosen as measured-harmless, not as measured-better, and the PR for Issue #175
# reports that plainly.** Swept on the golden set's 16 corpus items over the free path
# (`tests/fixtures/measure_hybrid_rerank.py`, no model calls): recall@5 is 8/8 and mean
# precision@5 is 0.425 for every weight from 0.00 to 0.08 inclusive, and precision falls
# from 0.09 upward (0.400 by w=0.09, 0.325 by w=0.60, and recall itself breaks at w=0.65).
# There is no weight at which the lexical term adds a relevant chunk the vector ranking had
# missed. 0.05 sits inside the no-loss plateau with room on the side that costs something,
# which is the same max-margin tie-break `tests/fixtures/calibrate_retrieval.py` applies to
# `TAU_TOP`; set it to 0.0 to return the pure vector ranking without removing the mechanism.
LEXICAL_WEIGHT = 0.05

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
    verbatim `text` its numeric-fidelity check reads.

    `score` is the **vector similarity**, and it stays that after Issue #175's reranking --
    it is not the combined ranking score. Two callers read it as a cosine similarity against
    thresholds calibrated as cosine similarities: `src/agent/critic/retrieval_confidence.py`
    (`TAU_TOP`/`TAU_SUPPORT`, measured by Issue #163's sweep) and Section 8's must-refuse
    mirror metric. Publishing a blended number in the same field would silently reinterpret
    both against a scale nobody measured. The combination decides *order and membership*;
    `score` keeps its meaning.
    """

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


def _normalized(values: Sequence[float]) -> list[float]:
    """Min-max the values into [0, 1] across the candidate pool, or all-zero if they are flat.

    Normalization is not decoration here, it is what makes a weighted sum mean anything.
    Cosine similarities from `bge-small-en-v1.5` on this corpus arrive in a narrow band
    (roughly 0.65-0.83 overall, and ~0.05 wide *within* one query's candidate pool) while
    containment spans 0.0-1.0 in steps of one query word. Summed raw, a weight as low as
    0.25 lets the lexical term decide the entire ranking -- measured, and the reason the raw
    variant degrades faster than this one in `tests/fixtures/measure_hybrid_rerank.py`'s
    sweep. Normalizing first makes `LEXICAL_WEIGHT` a share of the *spread* each signal
    actually has among the candidates being compared.

    A flat pool (one candidate, or several with identical scores) maps to all zeros rather
    than all ones: it contributes nothing to the ordering, which leaves the other term -- and
    then the stable sort -- to decide, instead of manufacturing a distinction that is not in
    the numbers.
    """
    if not values:
        return []
    low, high = min(values), max(values)
    span = high - low
    if span <= 0.0:
        return [0.0] * len(values)
    return [(value - low) / span for value in values]


def _combined_scores(
    query: str, chunks: Sequence[RetrievedChunk], *, lexical_weight: float
) -> list[float]:
    """One ranking score per candidate: `(1 - w) * vector + w * lexical`, both normalized.

    A weighted sum rather than a filter or a cascade, so a chunk is never dropped for
    scoring zero on one signal -- the lexical term moves a candidate up or down within the
    pool the vector search already chose, and cannot veto it.
    """
    vector_scores = _normalized([chunk.score for chunk in chunks])
    lexical_scores = _normalized(
        [
            lexical_overlap(query, candidate_text(chunk.heading_path, chunk.text))
            for chunk in chunks
        ]
    )
    return [
        (1.0 - lexical_weight) * vector + lexical_weight * lexical
        for vector, lexical in zip(vector_scores, lexical_scores)
    ]


def rerank(
    query: str,
    chunks: Sequence[RetrievedChunk],
    *,
    limit: int = DEFAULT_LIMIT,
    lexical_weight: float = LEXICAL_WEIGHT,
) -> list[RetrievedChunk]:
    """The best `limit` candidates by combined score, best first.

    Pure: no client, no embedder, no network -- which is what lets the reranking be tested
    against hand-built `RetrievedChunk`s rather than against an index.

    The sort is **stable**, and that is load-bearing rather than incidental. `chunks` arrives
    in Qdrant's own similarity order, so candidates that tie on the combined score keep the
    vector ranking's verdict on them. Containment over a question of five to eleven content
    words takes only a handful of distinct values, so ties are common rather than exotic, and
    resolving them by anything other than "what the vector search already thought" would make
    the result depend on dictionary iteration order.
    """
    scores = _combined_scores(query, chunks, lexical_weight=lexical_weight)
    ranked = sorted(range(len(chunks)), key=lambda index: -scores[index])
    return [chunks[index] for index in ranked[:limit]]


def search(
    query: str,
    limit: int = DEFAULT_LIMIT,
    source_type: str | None = None,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    collection_name: str = COLLECTION_NAME,
) -> list[RetrievedChunk]:
    """Return the `limit` best chunks for `query`, best first.

    Retrieve-then-rerank (Issue #175): Qdrant is asked for `INTERNAL_CANDIDATE_LIMIT`
    candidates by vector similarity, each is additionally scored by lexical containment
    against the query, and the top `limit` by the combination come back. The `limit` a
    caller passes still means exactly what it meant before -- how many results it gets --
    and `RetrievedChunk.score` still carries the vector similarity, not the combined score.

    `source_type` filters on the indexed payload field (Section 3: "`source_type` is an
    indexed payload field so it is filterable") -- the mechanism that lets a later
    `loaders/manual.py` be retrieved separately without a schema change. It is applied to
    the **wider internal query**, so the reranking never sees a chunk the caller filtered
    out: widening the pool must not widen what the filter admits.

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
        # Never below `limit`: a caller asking for more results than the reranking pool gets
        # the number it asked for, with the pool widened to match rather than the request
        # quietly truncated.
        limit=max(limit, INTERNAL_CANDIDATE_LIMIT),
        query_filter=query_filter,
        with_payload=True,
    )
    candidates = [_to_chunk(point.payload or {}, point.score) for point in response.points]
    return rerank(query, candidates, limit=limit)
