"""Tier-1 tests for the golden-set schema, loader, and structural check (Issue #144).

Matches what `tests/fixtures/golden_set.py` implements. `test_golden_set_matches_category_table`
below is **skipped** until both `golden_set_corpus.py` and `golden_set_tools.py` are non-empty,
rather than asserted against a partial set (Issue #144's constraint: the check must not fail on
its own PR shipping both files empty). Issue #146 populated the first of those two files, so the
skip now rests on `golden_set_tools.py` alone and lifts on its own once part 2c lands -- nothing
here is unskipped by hand. The other tests exercise the schema and the check directly, against
small in-test item sets, so they run -- and mean something -- independently of either.
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


def test_tools_content_file_still_ships_empty():
    """`CORPUS_ITEMS` was populated by Issue #146 (part 2b), so the half of Issue #144's
    original "both content files ship empty" assertion that named it is gone. The other half
    still holds and still means something -- `golden_set_tools.py` is empty until part 2c --
    and it is what keeps `test_golden_set_matches_category_table`'s skip below honest: that
    check is skipped *because* this is still true, not for a forgotten reason. This test
    goes away with 2c, when the skip guard stops skipping."""
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
