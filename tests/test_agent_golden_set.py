"""Tier-1 tests for the golden-set schema, loader, and structural check (Issue #144).

Scaffolding only, matching what `tests/fixtures/golden_set.py` implements -- no item content
exists yet, so `test_golden_set_matches_category_table` below is **skipped** until both
`golden_set_corpus.py` and `golden_set_tools.py` are non-empty, rather than asserted against
zero items (the issue's constraint: the check must not fail on its own PR shipping both files
empty). The other tests here exercise the schema and the check directly, against small
in-test item sets, so they run -- and mean something -- from this PR onward.
"""
from __future__ import annotations

import pytest

from tests.fixtures.golden_set import (
    CATEGORY_COUNTS,
    GoldenSetItem,
    GoldenSetStructureError,
    check_category_counts,
    load_golden_set,
)
from tests.fixtures.golden_set_corpus import CORPUS_ITEMS
from tests.fixtures.golden_set_tools import TOOL_ITEMS


def _item(item_id: str, category: str) -> GoldenSetItem:
    return GoldenSetItem(item_id=item_id, category=category, question="?")


def _matching_set() -> tuple[GoldenSetItem, ...]:
    """One item set with exactly Section 8's category/count table, distinct `item_id`s."""
    return tuple(
        _item(f"{category}-{i}", category)
        for category, count in CATEGORY_COUNTS.items()
        for i in range(count)
    )


def test_load_golden_set_concatenates_both_files_in_order():
    assert load_golden_set() == CORPUS_ITEMS + TOOL_ITEMS


def test_both_content_files_ship_empty_in_this_pr():
    assert CORPUS_ITEMS == ()
    assert TOOL_ITEMS == ()


def test_item_accepts_a_known_category():
    item = _item("x", "Inventory")
    assert item.category == "Inventory"
    assert item.expected_tool_names == frozenset()
    assert item.relevant_chunk_ids == frozenset()


def test_item_rejects_an_unknown_category():
    with pytest.raises(ValueError, match="unknown golden-set category"):
        _item("x", "Not A Real Category")


def test_check_category_counts_accepts_a_table_matching_set():
    check_category_counts(_matching_set())  # does not raise


def test_check_category_counts_rejects_wrong_counts():
    items = (_item("a", "Inventory"),)
    with pytest.raises(GoldenSetStructureError, match="category table"):
        check_category_counts(items)


def test_check_category_counts_rejects_duplicate_item_ids():
    items = tuple(
        GoldenSetItem(item_id="dup", category=category, question="?")
        for category, count in CATEGORY_COUNTS.items()
        for _ in range(count)
    )
    with pytest.raises(GoldenSetStructureError, match="duplicate"):
        check_category_counts(items)


@pytest.mark.skipif(
    not CORPUS_ITEMS or not TOOL_ITEMS,
    reason=(
        "golden_set_corpus.py / golden_set_tools.py are still empty -- Issue #144 ships the "
        "schema and loader only. Runs once the two follow-up issues populate both files."
    ),
)
def test_golden_set_matches_category_table():
    check_category_counts(load_golden_set())
