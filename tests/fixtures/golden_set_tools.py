"""Tool/inventory/archive-grounded golden-set items (Issue #144's follow-up "part 2c").

Grouped separately from `golden_set_corpus.py` by shared grounding source -- a live tool call
(`get_bearing_status`, `predict_health_state`, `check_inventory`,
`find_similar_historical_pattern`) rather than the docs corpus -- so the two follow-up issues
that populate them can append in parallel without a merge conflict on shared file content.
Expected to carry Section 8's 6 "Requires a live tool", 4 "Inventory", and 4 "Historical
similarity" items, and whichever of the 8 "Must refuse" items are refusals grounded in a tool
result (e.g. an unknown `bearing_id`) rather than an out-of-corpus documentation question; the
issue that populates this file makes that split, not this one.

Empty until that issue lands. `tests/test_agent_golden_set.py`'s category-table check skips
until this file and `golden_set_corpus.py` are both non-empty (Issue #144).
"""
from __future__ import annotations

from tests.fixtures.golden_set import GoldenSetItem

TOOL_ITEMS: tuple[GoldenSetItem, ...] = ()
