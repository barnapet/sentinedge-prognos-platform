"""`public_reference` loader (Issue #98, `docs/agent_design.md` Section 4).

Reads `references/public_references.json`, a committed, URL-stamped file -- not a live
fetch of anything at index time. Every entry's `text` is already the exact content to index
(a citation alone, or a citation plus that source's own published Scope/abstract text); see
that file's per-entry `provenance` field for where each `text` came from and why.

`source_ref` is the entry's external URL here, not a repo-relative path -- Section 3's
payload schema documents `source_ref` as "repo-relative path or URL" and a public reference
has no repo file of its own to point at; `source_id` is the stable internal slug instead.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from src.agent.rag.chunking import build_heading_path, chunk_document
from src.agent.rag.schema import Chunk, ChunkMetadata

REFERENCES_PATH = Path(__file__).resolve().parents[1] / "references" / "public_references.json"


def load_references(path: Path = REFERENCES_PATH) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["references"]


@dataclass(frozen=True)
class PublicReferenceLoader:
    references_path: Path = REFERENCES_PATH
    # Fixed per instance (not per entry) so every chunk from one `index.py` run shares one
    # build timestamp -- same convention as `DecisionDocLoader`.
    indexed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def iter_chunks(self) -> Iterator[Chunk]:
        for entry in load_references(self.references_path):
            for chunk_index, raw in enumerate(chunk_document(entry["text"])):
                metadata = ChunkMetadata(
                    source_type="public_reference",
                    source_id=entry["id"],
                    source_ref=entry["url"],
                    heading_path=build_heading_path(entry["label"], raw.heading_path_parts),
                    chunk_index=chunk_index,
                    indexed_at=self.indexed_at,
                )
                yield Chunk(text=raw.text, metadata=metadata)
