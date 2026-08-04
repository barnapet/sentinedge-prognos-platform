"""Qdrant collection builder (Issue #98, `docs/agent_design.md` Section 3).

Generic over loaders (`src.agent.rag.loaders.base.Loader`): builds the `prognos_docs`
collection from whatever loaders it is given, embedding each chunk's `embedding_text()`
with `fastembed`'s `BAAI/bge-small-en-v1.5` (384-dim, cosine distance) and upserting one
Qdrant point per chunk with Section 3's payload schema. Knows nothing about any specific
loader -- Section 4's hypothetical future `loaders/manual.py` would need no change here.

    python -m src.agent.rag.index

connects to `QDRANT_URL` (default `http://localhost:6333`, i.e. the `agent` compose
profile's published port) and rebuilds the collection from scratch.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Iterable

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

from src.agent.rag.loaders.base import Loader
from src.agent.rag.loaders.decision_doc import DecisionDocLoader
from src.agent.rag.loaders.public_reference import PublicReferenceLoader
from src.agent.rag.schema import Chunk

COLLECTION_NAME = "prognos_docs"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384
DEFAULT_QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

# Namespace for deriving a point ID from a chunk's own `chunk_id` (source_id + position),
# so rebuilding the index from unchanged sources reproduces the same point IDs -- an
# idempotent upsert, not an accumulating one. Fixed and arbitrary; only its stability
# across runs matters, not its value.
_POINT_ID_NAMESPACE = uuid.UUID("6f1f9b1a-6b3e-4b8a-9b7e-2a6b1f9c9d3e")


def _point_id(chunk: Chunk) -> str:
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk.chunk_id))


@dataclass(frozen=True)
class IndexStats:
    n_chunks: int
    n_decision_doc_chunks: int
    n_public_reference_chunks: int
    n_documents: int


def default_loaders() -> list[Loader]:
    return [DecisionDocLoader(), PublicReferenceLoader()]


def build_index(
    loaders: Iterable[Loader] | None = None,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    collection_name: str = COLLECTION_NAME,
) -> IndexStats:
    """(Re)build `collection_name` from scratch against every chunk `loaders` yields.

    A full rebuild, not an incremental diff -- the corpus is small (Section 3: fewer than
    twenty documents) and rebuildable from committed sources in seconds. Section 7 makes
    the same "cheap to regenerate, so don't bother making it incremental" choice for the
    inventory SQLite database, a later issue in this sequence.
    """
    loaders = list(loaders) if loaders is not None else default_loaders()

    chunks: list[Chunk] = []
    for loader in loaders:
        chunks.extend(loader.iter_chunks())

    client = QdrantClient(url=qdrant_url)
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    # Section 3: "source_type is an indexed payload field so it is filterable."
    client.create_payload_index(
        collection_name=collection_name,
        field_name="source_type",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    if chunks:
        # `threads=1` and a small `batch_size`: measured (this issue) at ~3.1 GB resident
        # for the default single-process onnxruntime path against this corpus's ~450
        # chunks -- onnxruntime's default intra-op thread pool plus a large internal batch
        # both scale with available cores/batch size, not with how small this corpus
        # actually is, and the unconstrained combination triggered an OOM kill in a
        # constrained environment. `parallel` is deliberately left at its default (`None`):
        # fastembed's own dispatch treats `parallel=0` as "spawn os.cpu_count() worker
        # processes, each loading its own copy of the model" -- worse, not better, for a
        # corpus this size, and was ruled out the same way.
        embedder = TextEmbedding(model_name=EMBEDDING_MODEL_NAME, threads=1)
        vectors = embedder.embed(
            [chunk.embedding_text() for chunk in chunks], batch_size=16
        )
        points = [
            PointStruct(
                id=_point_id(chunk),
                vector=vector.tolist(),
                payload={
                    "source_type": chunk.metadata.source_type,
                    "source_id": chunk.metadata.source_id,
                    "source_ref": chunk.metadata.source_ref,
                    "heading_path": chunk.metadata.heading_path,
                    "chunk_index": chunk.metadata.chunk_index,
                    "text": chunk.text,
                    "indexed_at": chunk.metadata.indexed_at,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        client.upsert(collection_name=collection_name, points=points)

    return IndexStats(
        n_chunks=len(chunks),
        n_decision_doc_chunks=sum(1 for c in chunks if c.metadata.source_type == "decision_doc"),
        n_public_reference_chunks=sum(
            1 for c in chunks if c.metadata.source_type == "public_reference"
        ),
        n_documents=len({c.metadata.source_id for c in chunks}),
    )


def main() -> None:
    stats = build_index()
    print(
        f"Indexed {stats.n_chunks} chunks from {stats.n_documents} documents into "
        f"'{COLLECTION_NAME}' ({stats.n_decision_doc_chunks} decision_doc, "
        f"{stats.n_public_reference_chunks} public_reference)."
    )


if __name__ == "__main__":
    main()
