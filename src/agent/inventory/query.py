"""Read-only queries against the real SQLite inventory database (Issue #100,
`docs/agent_design.md` Section 7).

These are the query surface `check_inventory` (`docs/agent_design.md` Section 2's read-only
MCP tool) will eventually wrap -- ordinary functions against the real database, with no MCP
framing, per this issue's data-layer-only scope. Every function takes an open `sqlite3.
Connection` rather than opening its own, so callers (tests, a future tool wrapper, `build_db`'s
own verification) control the connection's lifetime.
"""
from __future__ import annotations

import sqlite3

PART_COLUMNS = [
    "part_number",
    "description",
    "bearing_type",
    "quantity_on_hand",
    "unit_price_usd",
    "lead_time_days",
    "location",
]

ORDER_COLUMNS = [
    "order_id",
    "part_number",
    "quantity",
    "bearing_id",
    "requested_by",
    "approved_by",
    "approved_at",
    "created_at",
    "status",
]


def _row_to_dict(columns: list[str], row: tuple) -> dict:
    return dict(zip(columns, row))


def get_stock_level(conn: sqlite3.Connection, part_number: str) -> dict | None:
    """Return the part row for `part_number`, or `None` if it does not exist."""
    row = conn.execute(
        f"SELECT {', '.join(PART_COLUMNS)} FROM parts WHERE part_number = ?",
        (part_number,),
    ).fetchone()
    return _row_to_dict(PART_COLUMNS, row) if row is not None else None


def list_parts_below_reorder_threshold(conn: sqlite3.Connection, threshold: int) -> list[dict]:
    """Return every part with `quantity_on_hand < threshold`, ordered by the most depleted
    first. There is no `reorder_threshold` column in the schema (`docs/agent_design.md`
    Section 7 does not define one) -- the threshold is supplied by the caller."""
    rows = conn.execute(
        f"SELECT {', '.join(PART_COLUMNS)} FROM parts "
        "WHERE quantity_on_hand < ? ORDER BY quantity_on_hand ASC, part_number ASC",
        (threshold,),
    ).fetchall()
    return [_row_to_dict(PART_COLUMNS, row) for row in rows]


def get_order_history(conn: sqlite3.Connection, part_number: str) -> list[dict]:
    """Return every order ever placed for `part_number`, oldest first."""
    rows = conn.execute(
        f"SELECT {', '.join(ORDER_COLUMNS)} FROM orders "
        "WHERE part_number = ? ORDER BY order_id ASC",
        (part_number,),
    ).fetchall()
    return [_row_to_dict(ORDER_COLUMNS, row) for row in rows]
