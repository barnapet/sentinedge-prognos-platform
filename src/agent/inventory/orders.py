"""The transactional `place_order` path (Issue #100, `docs/agent_design.md` Section 7).

This is the data-layer half of the executor's one tool (Section 5's `place_order`) -- the MCP
tool wrapper itself is explicitly out of scope for this issue. What lives here is the actual
database transaction: the decrement of `quantity_on_hand` and the `INSERT` into `orders`
happen inside one SQLite transaction, so an order that would oversell stock aborts the whole
transaction rather than half-applying. `approval_token` minting/validation (Section 5) belongs
to the agent layer calling this function, not to this module -- `place_order` here requires
`approved_by`/`approved_at` as ordinary required arguments, and the schema's `NOT NULL`
constraints on those two columns are the second, independent enforcement of the approval gate
(`docs/agent_design.md` Section 7), regardless of what any caller passes.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


class InventoryError(Exception):
    """Base class for `place_order` failures."""


class UnknownPartError(InventoryError):
    """Raised when `part_number` does not exist in `parts`."""


class OrderRejectedError(InventoryError):
    """Raised when the database rejects the transaction -- most commonly because it would
    oversell stock (the `quantity_on_hand >= 0` CHECK constraint), but also covers any other
    schema constraint violation (e.g. a NULL `approved_by`/`approved_at`). The original
    `sqlite3.IntegrityError` text is preserved in the message so the actual cause is never
    hidden behind a generic label."""


def place_order(
    conn: sqlite3.Connection,
    *,
    part_number: str,
    quantity: int,
    requested_by: str,
    approved_by: str,
    approved_at: str,
    bearing_id: str | None = None,
    created_at: str | None = None,
    status: str = "placed",
) -> int:
    """Place an order: decrement `parts.quantity_on_hand` and insert one `orders` row, in one
    transaction. Returns the new `order_id`.

    Raises `UnknownPartError` if `part_number` is not in `parts`, or `OrderRejectedError` if
    the transaction violates a schema constraint (oversell, or a NULL approval field) -- in
    either case no partial write is left behind: the decrement and the insert either both
    happen or neither does.
    """
    if quantity <= 0:
        raise ValueError(f"quantity must be positive, got {quantity!r}")

    created_at = created_at or datetime.now(timezone.utc).isoformat()

    try:
        with conn:
            cur = conn.execute(
                "UPDATE parts SET quantity_on_hand = quantity_on_hand - ? WHERE part_number = ?",
                (quantity, part_number),
            )
            if cur.rowcount == 0:
                raise UnknownPartError(f"no part with part_number={part_number!r}")

            cur = conn.execute(
                """
                INSERT INTO orders (
                    part_number, quantity, bearing_id, requested_by,
                    approved_by, approved_at, created_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    part_number,
                    quantity,
                    bearing_id,
                    requested_by,
                    approved_by,
                    approved_at,
                    created_at,
                    status,
                ),
            )
            return cur.lastrowid
    except sqlite3.IntegrityError as exc:
        raise OrderRejectedError(
            f"order for {quantity} x {part_number!r} rejected by the database: {exc}"
        ) from exc
