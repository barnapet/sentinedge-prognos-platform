"""Loader protocol (Issue #98, `docs/agent_design.md` Section 4).

`index.py` depends on this and nothing else: it can build a Qdrant collection from any
object satisfying `iter_chunks() -> Iterable[Chunk]`, without importing or special-casing
any specific loader. This is what makes "adding `loaders/manual.py` later" a change confined
to that one new file plus a registration line.
"""
from __future__ import annotations

from typing import Iterable, Protocol

from src.agent.rag.schema import Chunk


class Loader(Protocol):
    def iter_chunks(self) -> Iterable[Chunk]: ...
