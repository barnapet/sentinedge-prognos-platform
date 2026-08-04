"""Chunk + ChunkMetadata (Issue #98, `docs/agent_design.md` Section 4).

The only contract every `src/agent/rag/loaders/*` module and `index.py` share: a loader
knows how to walk one source and emit `Chunk` objects; `index.py` knows how to embed and
upsert `Chunk` objects and nothing about any specific loader. Adding a new source type later
(Section 4's "adding `loaders/manual.py`" test) is one new loader module emitting this same
type -- no change here, in `chunking.py`, or in `index.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

# This issue's two loaders only. Section 2's tool-result `source` block reuses the same
# `source_type` namespace for values this issue does not mint ("live_endpoint",
# "inventory", "trajectory_match") -- those are minted by later MCP-tool-server issues, so
# they are deliberately not listed here; this constant validates what THIS issue's loaders
# produce, not Section 2's full eventual vocabulary.
KNOWN_SOURCE_TYPES = frozenset({"decision_doc", "public_reference"})


@dataclass(frozen=True)
class ChunkMetadata:
    """Section 3's Qdrant payload fields, minus `text` (kept on `Chunk` itself, not
    duplicated here, so there is exactly one place a chunk's text lives).
    """

    source_type: str
    source_id: str
    source_ref: str
    heading_path: str
    chunk_index: int
    indexed_at: str  # ISO 8601 -- a str, not datetime, so it serializes into a Qdrant payload unchanged

    def __post_init__(self) -> None:
        if self.source_type not in KNOWN_SOURCE_TYPES:
            raise ValueError(
                f"unknown source_type {self.source_type!r}; expected one of {sorted(KNOWN_SOURCE_TYPES)}"
            )


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit.

    `text` is the loader's verbatim source text -- never authored, never prefixed -- so
    Section 8's tier-1 verbatim assertion can check it directly against the source file with
    a plain substring test. The heading-path prefix Section 4 requires for retrieval is
    added separately by `embedding_text()`, precisely so that prefix (synthesized, not
    source text) never has to be stripped back out for that test.
    """

    text: str
    metadata: ChunkMetadata

    @property
    def chunk_id(self) -> str:
        """Stable id, deterministic across re-indexing runs: same source + same position
        in it always yields the same id, so rebuilding the collection upserts rather than
        accumulating duplicates.
        """
        return f"{self.metadata.source_id}::{self.metadata.chunk_index}"

    def embedding_text(self) -> str:
        """Section 4: "every chunk's text is prefixed with its heading path" -- the text
        actually sent to the embedding model and stored as the payload's `text` field.
        """
        return f"{self.metadata.heading_path}\n\n{self.text}"
