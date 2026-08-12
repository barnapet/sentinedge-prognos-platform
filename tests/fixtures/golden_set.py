"""Golden-set item schema, loader, and structural check for Section 8's tier 2 (Issue #144).

`docs/agent_design.md` Section 8 fixes a 30-item golden set across five categories, scored on
three independent binary sub-scores per item (correct tool call, source-grounded answer,
correct refusal where applicable). This module is scaffolding only: the item shape, a loader
that concatenates the two content files this issue ships empty, and a structural check that
the categories and counts match Section 8's table. **No item content and no scoring/pass-fail
logic live here** -- those belong to `tests/fixtures/golden_set_corpus.py` /
`golden_set_tools.py` (two later issues, split by shared grounding source so each can append
without a merge conflict on the other's content) and a further issue after that, respectively.

    from tests.fixtures.golden_set import load_golden_set

    items = load_golden_set()

Matches the shape of `tests/fixtures/adversarial_payloads.py` (flat, heavily-commented
module-level data) for the two content files, and of `tests/fixtures/cassette.py` (loader +
its own exception for a structural failure) for this one.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

# Section 8's tier-2 table, verbatim, in the table's own row order: category name -> required
# item count. 8 + 6 + 4 + 4 + 8 = 30.
CATEGORY_COUNTS: dict[str, int] = {
    "Answerable from the docs": 8,
    "Requires a live tool": 6,
    "Inventory": 4,
    "Historical similarity": 4,
    'Must refuse / "I don\'t know"': 8,
}

KNOWN_CATEGORIES = frozenset(CATEGORY_COUNTS)

TOTAL_ITEM_COUNT = sum(CATEGORY_COUNTS.values())


class GoldenSetStructureError(ValueError):
    """The golden set (or a subset passed to `check_category_counts`) does not match Section
    8's fixed category table, or carries a duplicate `item_id`.

    A plain exception rather than a bare `assert`, so both `pytest` and any later non-pytest
    caller get the same message and the same type to catch.
    """


@dataclass(frozen=True)
class GoldenSetItem:
    """One Section 8 tier-2 item and the scoring contract it must satisfy.

    `expected_tool_names` is compared by set equality on names, not on arguments -- arguments
    legitimately vary run to run (Section 8). `allowed_source_ids` is the set at least one of
    which a grounded answer must cite. `relevant_chunk_ids` is Issue #102's retrieval-quality
    addition (recall@k / precision@k against the same *k* the answerer retrieves with) and is
    declared only for corpus-dependent items -- empty is correct for an item with no corpus
    dependency, not a missing value.

    Every field but the first three defaults to empty so a follow-up issue's item only states
    what actually applies to it; `__post_init__` validates `category` alone, since Section 8's
    scoring contract is exactly what a *later* issue implements, not this one.
    """

    item_id: str
    category: str
    question: str
    expected_tool_names: frozenset[str] = frozenset()
    allowed_source_ids: frozenset[str] = frozenset()
    required_substrings: tuple[str, ...] = ()
    forbidden_substrings: tuple[str, ...] = ()
    relevant_chunk_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.category not in KNOWN_CATEGORIES:
            raise ValueError(
                f"unknown golden-set category {self.category!r}; expected one of "
                f"{sorted(KNOWN_CATEGORIES)}"
            )


def load_golden_set() -> tuple[GoldenSetItem, ...]:
    """Every golden-set item: `golden_set_corpus.py`'s items then `golden_set_tools.py`'s, in
    each file's own declaration order.

    Imported here rather than at module top, the same way `cassette.py`'s
    `current_fingerprint()` reaches into `src/agent/` -- `golden_set_corpus.py` and
    `golden_set_tools.py` both import `GoldenSetItem` from this module, so a top-level import
    back into either of them would be circular.
    """
    from tests.fixtures.golden_set_corpus import CORPUS_ITEMS
    from tests.fixtures.golden_set_tools import TOOL_ITEMS

    return CORPUS_ITEMS + TOOL_ITEMS


def check_category_counts(items: Iterable[GoldenSetItem]) -> None:
    """Raise `GoldenSetStructureError` unless `items`' categories and per-category counts
    match `CATEGORY_COUNTS` exactly, and every `item_id` is unique.

    Checked as `category -> count` equality, not a bare `len(items) == 30`, so a 30-item set
    with the wrong category mix still fails -- the same "don't let an aggregate hide a failing
    subgroup" discipline Section 8 states for the golden set's own pass/fail rule, one level
    down at the scaffolding this check itself is part of.
    """
    materialized = tuple(items)

    actual_counts: dict[str, int] = {}
    for item in materialized:
        actual_counts[item.category] = actual_counts.get(item.category, 0) + 1
    if actual_counts != CATEGORY_COUNTS:
        raise GoldenSetStructureError(
            "golden set does not match docs/agent_design.md Section 8's category table; "
            f"expected {CATEGORY_COUNTS}, got {actual_counts}"
        )

    item_ids = [item.item_id for item in materialized]
    duplicates = sorted({item_id for item_id in item_ids if item_ids.count(item_id) > 1})
    if duplicates:
        raise GoldenSetStructureError(f"duplicate golden-set item_id(s): {duplicates}")
