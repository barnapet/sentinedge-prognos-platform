"""Tier-1 tests (Issue #100, `docs/agent_design.md` Section 7 / Section 8) for the agent
layer's inventory data layer: the CSV-seed-to-SQLite materialization, the schema's own
enforcement of the approval gate and the no-oversell invariant, and the transactional
`place_order` path. No API key, no network -- everything here is local SQLite against a
`tmp_path` database, never the repo's real (gitignored) `data/agent/inventory.db`.
"""
from __future__ import annotations

import sqlite3

import pytest

from src.agent.inventory.build_db import (
    ORDERS_COLUMNS,
    ORDERS_SEED_PATH,
    PARTS_COLUMNS,
    PARTS_SEED_PATH,
    build_db,
)
from src.agent.inventory.orders import (
    OrderRejectedError,
    UnknownPartError,
    place_order,
)
from src.agent.inventory.query import (
    get_order_history,
    get_stock_level,
    list_parts_below_reorder_threshold,
)


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "inventory.db"
    build_db(path)
    return path


@pytest.fixture
def conn(db_path):
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
    finally:
        connection.close()


def _non_comment_lines(path):
    with path.open(encoding="utf-8") as f:
        return [line for line in f if not line.startswith("#")]


# --- Seed files themselves -----------------------------------------------------------


def test_parts_seed_has_za2115_as_real_primary_row_plus_a_real_seed():
    """A real seed, not a one-row placeholder (Issue #100's own phrasing)."""
    lines = _non_comment_lines(PARTS_SEED_PATH)
    part_numbers = [line.split(",")[0] for line in lines[1:] if line.strip()]

    assert "ZA-2115" in part_numbers
    assert len(part_numbers) >= 5, "seed must be more than a one-row placeholder"

    raw_text = PARTS_SEED_PATH.read_text(encoding="utf-8")
    assert "Rexnord ZA-2115" in raw_text


def test_parts_seed_header_comment_discloses_invented_demo_values():
    """docs/agent_design.md Section 7: stock/price/lead-time/location are invented demo
    values for every row, including ZA-2115 -- and the seed file must say so explicitly,
    in these terms, not leave it implicit."""
    raw_text = PARTS_SEED_PATH.read_text(encoding="utf-8")
    comment_lines = [line for line in raw_text.splitlines() if line.startswith("#")]
    comment_block = "\n".join(comment_lines)

    assert "invented demo" in comment_block.lower()
    assert "no warehouse" in comment_block.lower()


def test_orders_seed_is_header_only():
    lines = _non_comment_lines(ORDERS_SEED_PATH)
    non_blank = [line for line in lines if line.strip()]
    assert len(non_blank) == 1
    assert non_blank[0].strip().split(",") == ORDERS_COLUMNS


# --- build_db: schema and seeded rows -------------------------------------------------


def test_build_db_creates_expected_schema(db_path):
    connection = sqlite3.connect(db_path)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {"parts", "orders"} <= tables

        parts_cols = [row[1] for row in connection.execute("PRAGMA table_info(parts)")]
        orders_cols = [row[1] for row in connection.execute("PRAGMA table_info(orders)")]
        assert parts_cols == PARTS_COLUMNS
        assert orders_cols == ORDERS_COLUMNS
    finally:
        connection.close()


def test_build_db_seeds_expected_number_of_parts_and_no_orders(conn):
    expected_count = len(
        [line for line in _non_comment_lines(PARTS_SEED_PATH)[1:] if line.strip()]
    )
    (n_parts,) = conn.execute("SELECT COUNT(*) FROM parts").fetchone()
    (n_orders,) = conn.execute("SELECT COUNT(*) FROM orders").fetchone()

    assert n_parts == expected_count
    assert n_orders == 0


def test_build_db_seeds_za2115_row_correctly(conn):
    row = get_stock_level(conn, "ZA-2115")
    assert row is not None
    assert "Rexnord ZA-2115" in row["description"]
    assert row["quantity_on_hand"] == 4
    assert row["unit_price_usd"] == pytest.approx(850.00)
    assert row["lead_time_days"] == 21
    assert row["location"] == "Warehouse A - Shelf 12"


def test_build_db_is_idempotent_by_default(db_path, conn):
    """Calling build_db again on an existing database must not touch it -- runtime orders
    placed during a demo must survive a repeat call at startup."""
    order_id = place_order(
        conn,
        part_number="BRG-6205-2RS",
        quantity=1,
        requested_by="agent",
        approved_by="plant-manager",
        approved_at="2026-08-04T12:00:00+00:00",
    )
    conn.close()

    n_seeded = build_db(db_path)
    assert n_seeded == 0, "an existing database must be left untouched by default"

    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute(
            "SELECT order_id FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        assert row is not None, "the order placed before the repeat build_db call must survive"
    finally:
        connection.close()


def test_build_db_reset_rebuilds_from_seed_and_discards_orders(db_path, conn):
    place_order(
        conn,
        part_number="BRG-6205-2RS",
        quantity=1,
        requested_by="agent",
        approved_by="plant-manager",
        approved_at="2026-08-04T12:00:00+00:00",
    )
    conn.close()

    n_seeded = build_db(db_path, reset=True)
    assert n_seeded > 0

    connection = sqlite3.connect(db_path)
    try:
        (n_orders,) = connection.execute("SELECT COUNT(*) FROM orders").fetchone()
        assert n_orders == 0
        row = get_stock_level(connection, "BRG-6205-2RS")
        assert row["quantity_on_hand"] == 25, "reset must restore the seed's original quantity"
    finally:
        connection.close()


# --- place_order: in-stock success -----------------------------------------------------


def test_place_order_within_stock_succeeds_and_persists(conn):
    before = get_stock_level(conn, "BRG-6205-2RS")
    assert before["quantity_on_hand"] == 25

    order_id = place_order(
        conn,
        part_number="BRG-6205-2RS",
        quantity=5,
        bearing_id="1st_test-bearing3",
        requested_by="agent",
        approved_by="plant-manager",
        approved_at="2026-08-04T12:00:00+00:00",
    )
    assert order_id is not None

    after = get_stock_level(conn, "BRG-6205-2RS")
    assert after["quantity_on_hand"] == 20

    history = get_order_history(conn, "BRG-6205-2RS")
    assert len(history) == 1
    assert history[0]["order_id"] == order_id
    assert history[0]["quantity"] == 5
    assert history[0]["bearing_id"] == "1st_test-bearing3"
    assert history[0]["approved_by"] == "plant-manager"
    assert history[0]["status"] == "placed"


# --- place_order: oversell rejection, with re-query evidence --------------------------


def test_place_order_oversell_is_rejected_and_leaves_quantity_unchanged(conn):
    """The core acceptance criterion (Issue #100): an overselling order must be rejected,
    and the rejection must be proven by re-querying persisted state afterward, not merely
    by asserting that an exception was raised."""
    before = get_stock_level(conn, "TOOL-BRG-PULLER-KIT")
    assert before["quantity_on_hand"] == 3

    with pytest.raises(OrderRejectedError):
        place_order(
            conn,
            part_number="TOOL-BRG-PULLER-KIT",
            quantity=4,  # one more than in stock
            requested_by="agent",
            approved_by="plant-manager",
            approved_at="2026-08-04T12:00:00+00:00",
        )

    # Re-query persisted state on a fresh read -- this is the evidence, not the exception.
    after = get_stock_level(conn, "TOOL-BRG-PULLER-KIT")
    assert after["quantity_on_hand"] == 3
    assert after["quantity_on_hand"] == before["quantity_on_hand"]

    history = get_order_history(conn, "TOOL-BRG-PULLER-KIT")
    assert history == [], "the rejected order must not have inserted an orders row"


def test_place_order_oversell_rejection_is_visible_on_a_second_connection(db_path):
    """Same as above, but reopening the database from disk -- proves the rollback was
    actually committed to the file, not just visible within the same connection/transaction."""
    conn1 = sqlite3.connect(db_path)
    try:
        with pytest.raises(OrderRejectedError):
            place_order(
                conn1,
                part_number="TOOL-BRG-PULLER-KIT",
                quantity=100,
                requested_by="agent",
                approved_by="plant-manager",
                approved_at="2026-08-04T12:00:00+00:00",
            )
    finally:
        conn1.close()

    conn2 = sqlite3.connect(db_path)
    try:
        row = get_stock_level(conn2, "TOOL-BRG-PULLER-KIT")
        assert row["quantity_on_hand"] == 3
        assert get_order_history(conn2, "TOOL-BRG-PULLER-KIT") == []
    finally:
        conn2.close()


def test_place_order_unknown_part_raises_and_touches_nothing(conn):
    (n_orders_before,) = conn.execute("SELECT COUNT(*) FROM orders").fetchone()

    with pytest.raises(UnknownPartError):
        place_order(
            conn,
            part_number="NOT-A-REAL-PART",
            quantity=1,
            requested_by="agent",
            approved_by="plant-manager",
            approved_at="2026-08-04T12:00:00+00:00",
        )

    (n_orders_after,) = conn.execute("SELECT COUNT(*) FROM orders").fetchone()
    assert n_orders_after == n_orders_before


def test_place_order_rejects_non_positive_quantity(conn):
    with pytest.raises(ValueError):
        place_order(
            conn,
            part_number="ZA-2115",
            quantity=0,
            requested_by="agent",
            approved_by="plant-manager",
            approved_at="2026-08-04T12:00:00+00:00",
        )


# --- Schema-level enforcement, independent of place_order -----------------------------


def test_orders_insert_without_approved_by_fails_at_schema_level(conn):
    """docs/agent_design.md Section 7: an order row without a recorded human approval must
    be impossible to construct in the database, independent of anything the agent layer
    does -- so this bypasses place_order entirely and inserts raw SQL."""
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO orders (
                part_number, quantity, bearing_id, requested_by,
                approved_by, approved_at, created_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("ZA-2115", 1, None, "agent", None, None, "2026-08-04T12:00:00+00:00", "placed"),
        )


def test_orders_insert_without_approved_at_fails_at_schema_level(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO orders (
                part_number, quantity, bearing_id, requested_by,
                approved_by, approved_at, created_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("ZA-2115", 1, None, "agent", "plant-manager", None, "2026-08-04T12:00:00+00:00", "placed"),
        )


def test_parts_quantity_on_hand_check_constraint_rejects_negative_directly(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE parts SET quantity_on_hand = -1 WHERE part_number = 'ZA-2115'"
        )


def test_orders_foreign_key_to_parts_is_enforced(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO orders (
                part_number, quantity, bearing_id, requested_by,
                approved_by, approved_at, created_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "NOT-A-REAL-PART",
                1,
                None,
                "agent",
                "plant-manager",
                "2026-08-04T12:00:00+00:00",
                "2026-08-04T12:00:00+00:00",
                "placed",
            ),
        )


# --- Query module ------------------------------------------------------------------------


def test_get_stock_level_returns_none_for_unknown_part(conn):
    assert get_stock_level(conn, "NOT-A-REAL-PART") is None


def test_list_parts_below_reorder_threshold(conn):
    below = list_parts_below_reorder_threshold(conn, threshold=5)
    part_numbers = {row["part_number"] for row in below}

    assert "TOOL-BRG-PULLER-KIT" in part_numbers  # quantity_on_hand == 3
    assert "ZA-2115" in part_numbers  # quantity_on_hand == 4
    assert "BRG-6205-2RS" not in part_numbers  # quantity_on_hand == 25

    for row in below:
        assert row["quantity_on_hand"] < 5

    # Most-depleted first.
    quantities = [row["quantity_on_hand"] for row in below]
    assert quantities == sorted(quantities)


def test_get_order_history_is_empty_before_any_orders(conn):
    assert get_order_history(conn, "ZA-2115") == []


def test_get_order_history_is_chronological(conn):
    first_id = place_order(
        conn,
        part_number="ZA-2115",
        quantity=1,
        requested_by="agent",
        approved_by="plant-manager",
        approved_at="2026-08-04T12:00:00+00:00",
        created_at="2026-08-04T12:00:00+00:00",
    )
    second_id = place_order(
        conn,
        part_number="ZA-2115",
        quantity=1,
        requested_by="agent",
        approved_by="plant-manager",
        approved_at="2026-08-04T13:00:00+00:00",
        created_at="2026-08-04T13:00:00+00:00",
    )

    history = get_order_history(conn, "ZA-2115")
    assert [row["order_id"] for row in history] == [first_id, second_id]
