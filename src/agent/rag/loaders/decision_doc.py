"""`decision_doc` loader (Issue #98, `docs/agent_design.md` Section 4).

Launch corpus rule: every `docs/*.md` file except `docs/CONTRIBUTING.md`, plus `README.md`
-- `docs/CONTRIBUTING.md` is excluded because commit conventions have no bearing on a
technician question. File list is enumerated at call time (`glob`), not hard-coded, so the
count reported in a PR is always the real one, not a stale copy of "eighteen at time of
writing."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from src.agent.rag.chunking import build_heading_path, chunk_document
from src.agent.rag.schema import Chunk, ChunkMetadata

REPO_ROOT = Path(__file__).resolve().parents[4]
EXCLUDED_FILENAMES = frozenset({"CONTRIBUTING.md"})


def decision_doc_paths(repo_root: Path = REPO_ROOT) -> list[Path]:
    """`docs/*.md` (minus the excluded names) plus `README.md`, repo-root-relative."""
    docs = sorted(
        p for p in (repo_root / "docs").glob("*.md") if p.name not in EXCLUDED_FILENAMES
    )
    return [*docs, repo_root / "README.md"]


@dataclass(frozen=True)
class DecisionDocLoader:
    repo_root: Path = REPO_ROOT
    # Fixed per instance (not per file) so every chunk from one `index.py` run shares one
    # build timestamp -- reindexing later is a new instance, a new timestamp.
    indexed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def iter_chunks(self) -> Iterator[Chunk]:
        for path in decision_doc_paths(self.repo_root):
            relative_path = path.relative_to(self.repo_root).as_posix()
            text = path.read_text(encoding="utf-8")
            for chunk_index, raw in enumerate(chunk_document(text)):
                metadata = ChunkMetadata(
                    source_type="decision_doc",
                    source_id=relative_path,
                    source_ref=relative_path,
                    heading_path=build_heading_path(path.name, raw.heading_path_parts),
                    chunk_index=chunk_index,
                    indexed_at=self.indexed_at,
                )
                yield Chunk(text=raw.text, metadata=metadata)
