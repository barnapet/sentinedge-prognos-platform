"""Corpus-grounded golden-set items (Issue #144's follow-up "part 2b").

Grouped separately from `golden_set_tools.py` by shared grounding source -- the docs corpus
(`docs/*.md` plus `README.md`, Section 4) rather than a live tool, inventory, or the
trajectory archive -- so the two follow-up issues that populate them can append in parallel
without a merge conflict on shared file content. Expected to carry most or all of Section 8's
8 "Answerable from the docs" items, and whichever of the 8 "Must refuse" items are refusals of
an out-of-corpus documentation question rather than of a tool-grounded one; the issue that
populates this file makes that split, not this one.

Empty until that issue lands. `tests/test_agent_golden_set.py`'s category-table check skips
until this file and `golden_set_tools.py` are both non-empty (Issue #144).
"""
from __future__ import annotations

from tests.fixtures.golden_set import GoldenSetItem

CORPUS_ITEMS: tuple[GoldenSetItem, ...] = ()
