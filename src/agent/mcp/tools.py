"""The five tool bodies (Issue #110, `docs/agent_design.md` Section 2).

Registration lives in `readonly_server.py` and `write_server.py`; what lives here is what
each tool *does*, as an ordinary function with its dependencies as explicit arguments. The
split is what makes Section 8's tier-1 tests possible: they call these directly, with no
model, no API key, and no network beyond localhost, and check the envelope shape and the
`is_error` behaviour rather than checking that some model chose to call them.

Every function here returns a `CallToolResult` and raises nothing. That is Section 2's
error rule, applied at the only layer that can honour it -- a failure that escaped this
module would reach the MCP server, be stringified into `Error executing tool <name>: ...`,
and arrive at the model as a stack trace instead of something it can degrade from.

Each tool's `source` block is minted here from a hard-coded `source_type`/`source_id`
pair. Neither value is ever taken from an argument, so no input -- including one written
by an injected instruction inside a retrieved chunk or an inventory `description` field --
can influence what a result claims to be sourced from.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Callable, Sequence

from mcp.types import CallToolResult

from src.agent.inventory.build_db import DB_PATH
from src.agent.inventory.orders import (
    InventoryError,
    OrderRejectedError,
    UnknownPartError,
    place_order as place_order_row,
)
from src.agent.inventory.query import find_parts
from src.agent.mcp.results import (
    DOCS_INDEX_UNREACHABLE,
    INVENTORY_UNAVAILABLE,
    ORDER_FAILED,
    SERVICE_UNREACHABLE,
    failed,
    ok,
)
from src.agent.mcp.serving_client import (
    DEFAULT_BASE_URL,
    DRIFT_ENDPOINT,
    PREDICT_ENDPOINT,
    ServingRejected,
    ServingUnreachable,
    get_drift,
    post_predict,
)
from src.agent.rag.index import COLLECTION_NAME
from src.agent.rag.retrieval import DEFAULT_LIMIT, RetrievedChunk
from src.agent.rag.retrieval import search as default_search

INVENTORY_SOURCE_ID = "data/agent/inventory.db"
# `search_documentation` queries a live Qdrant service at request time, so `live_endpoint`
# is the vocabulary value that describes *the retrieval*; the citable per-chunk ids each
# carry their own `decision_doc`/`public_reference` block inside `data.results` (Section 6:
# "every retrieved chunk carries a stable `chunk_id`"). Flagged in the PR for #110, since
# Section 2's five-value vocabulary has no term for a search index and this is the closest
# reading rather than a new value.
DOCS_SOURCE_ID = f"SEARCH {COLLECTION_NAME}"

SearchFn = Callable[..., Sequence[RetrievedChunk]]


# --------------------------------------------------------------------------------------
# Read-only tools
# --------------------------------------------------------------------------------------


def get_bearing_status(
    bearing_id: str | None = None, *, base_url: str = DEFAULT_BASE_URL
) -> CallToolResult:
    """Current monitoring state for one bearing, or for every tracked bearing.

    Wraps `GET /monitoring/drift` over HTTP (never `create_app()` -- see
    `serving_client.py`). An unknown `bearing_id` is a **structured not-found**, not an
    error: `docs/agent_design.md` Section 10 case 1 is precisely the failure of inventing a
    state for a bearing nobody is tracking, so the result says `found: false` and lists
    which bearings *are* tracked, and the answer built from it has no health-state label to
    invent one from.
    """
    try:
        body = get_drift(base_url=base_url)
    except ServingUnreachable:
        return failed("live_endpoint", DRIFT_ENDPOINT, SERVICE_UNREACHABLE)
    except ServingRejected as exc:
        return failed(
            "live_endpoint",
            DRIFT_ENDPOINT,
            f"the prediction service rejected the request (HTTP {exc.status_code})",
        )

    bearings: dict[str, Any] = body.get("bearings", {})
    if bearing_id is None:
        return ok("live_endpoint", DRIFT_ENDPOINT, {"bearings": bearings})
    if bearing_id not in bearings:
        return ok(
            "live_endpoint",
            DRIFT_ENDPOINT,
            {
                "bearing_id": bearing_id,
                "found": False,
                "tracked_bearings": sorted(bearings),
            },
        )
    return ok(
        "live_endpoint",
        DRIFT_ENDPOINT,
        {"bearing_id": bearing_id, "found": True, "status": bearings[bearing_id]},
    )


def predict_health_state(
    bearing_id: str, signal: list[float], *, base_url: str = DEFAULT_BASE_URL
) -> CallToolResult:
    """Score one raw single-window signal for one bearing.

    Deliberately awkward to call, and `docs/agent_design.md` Section 2 says so: it needs a
    full 20,480-point raw window, which a technician asking a question does not have. That
    awkwardness is a direct consequence of `docs/serving_design.md` Section 1 giving the
    server sole ownership of feature computation, and papering over it with an agent-side
    wrapper that fabricates or reuses a signal would reintroduce exactly the
    second-copy-of-the-feature-logic problem Section 1 exists to prevent. In practice the
    answerer reaches for `get_bearing_status`.
    """
    if not signal:
        return failed(
            "live_endpoint", PREDICT_ENDPOINT, "a signal window is required, and none was given"
        )
    try:
        body = post_predict(bearing_id, list(signal), base_url=base_url)
    except ServingUnreachable:
        return failed("live_endpoint", PREDICT_ENDPOINT, SERVICE_UNREACHABLE)
    except ServingRejected as exc:
        return failed(
            "live_endpoint",
            PREDICT_ENDPOINT,
            f"the prediction service rejected the request (HTTP {exc.status_code})",
        )
    return ok("live_endpoint", PREDICT_ENDPOINT, {"bearing_id": bearing_id, **body})


def check_inventory(
    part_number: str | None = None,
    bearing_type: str | None = None,
    *,
    db_path: Path = DB_PATH,
) -> CallToolResult:
    """Matching part rows from the real SQLite inventory (`docs/agent_design.md` Section 7).

    Read-only by construction: this opens the database in SQLite's read-only URI mode, so
    the read tool on the read-only server cannot write a row even if something in its
    argument path went wrong. Placing an order is `place_order`, on the other server.

    No matches is a successful empty result, not an error -- "we do not stock that part"
    is an answer, and returning it as a failure would push a legitimate answer onto Section
    6's degraded path.
    """
    try:
        conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    except sqlite3.Error:
        return failed("inventory", INVENTORY_SOURCE_ID, INVENTORY_UNAVAILABLE)
    try:
        parts = find_parts(conn, part_number=part_number, bearing_type=bearing_type)
    except sqlite3.Error:
        return failed("inventory", INVENTORY_SOURCE_ID, INVENTORY_UNAVAILABLE)
    finally:
        conn.close()

    return ok(
        "inventory",
        INVENTORY_SOURCE_ID,
        {
            "query": {"part_number": part_number, "bearing_type": bearing_type},
            "match_count": len(parts),
            "parts": parts,
        },
    )


def search_documentation(
    query: str,
    limit: int = DEFAULT_LIMIT,
    source_type: str | None = None,
    *,
    search: SearchFn | None = None,
) -> CallToolResult:
    """Retrieve chunks from the `prognos_docs` collection (Sections 3 and 4).

    Each hit carries its **own** `source` block -- `decision_doc` or `public_reference`,
    with the chunk's stable `chunk_id` as `source_id` -- because those are the ids Section
    6's citation-existence check tests membership against. The chunk text is returned
    verbatim, exactly as indexed, so the numeric-fidelity check has the real characters to
    substring-match a claim's numbers against.

    Every failure in the retrieval path becomes one message. The caller cannot act on the
    difference between "the embedding model would not load" and "Qdrant refused the
    connection" -- in both cases the index is unavailable for this question -- and reporting
    local infrastructure detail into the model's context invites it to speculate about
    machinery it cannot see.
    """
    if source_type is not None and source_type not in {"decision_doc", "public_reference"}:
        return failed(
            "live_endpoint",
            DOCS_SOURCE_ID,
            "source_type must be 'decision_doc' or 'public_reference' when given",
        )
    search = search if search is not None else default_search
    try:
        chunks = search(query, limit=limit, source_type=source_type)
    except Exception:  # noqa: BLE001 -- one failure mode by design, see docstring
        return failed("live_endpoint", DOCS_SOURCE_ID, DOCS_INDEX_UNREACHABLE)

    results = [
        {
            "source": {
                "source_type": chunk.source_type,
                "source_id": chunk.chunk_id,
                "source_ref": chunk.source_ref,
            },
            "heading_path": chunk.heading_path,
            "score": chunk.score,
            "text": chunk.text,
        }
        for chunk in chunks
    ]
    return ok(
        "live_endpoint",
        DOCS_SOURCE_ID,
        {"query": query, "result_count": len(results), "results": results},
    )


# --------------------------------------------------------------------------------------
# Write-capable tool
# --------------------------------------------------------------------------------------


def place_order(
    part_number: str,
    quantity: int,
    requested_by: str,
    approved_by: str,
    approved_at: str,
    bearing_id: str | None = None,
    *,
    db_path: Path = DB_PATH,
) -> CallToolResult:
    """Place one order: decrement stock and insert an `orders` row, in one transaction.

    Wraps `src/agent/inventory/orders.place_order` (#101) unchanged -- the oversell abort
    and the `NOT NULL` constraints on `approved_by`/`approved_at` stay where they are, in
    the schema, which `docs/agent_design.md` Section 7 calls the second, independent
    enforcement of the approval gate.

    **The approval gate itself is not built here, and this tool does not pretend to be
    one.** Section 5's `approval_token` -- minted out-of-band by the harness, scoped to one
    order, single-use, short-lived -- belongs to the agent layer that calls this tool, and
    Issue #110 excludes it explicitly. What this tool requires is what the *database*
    requires: a recorded approver and approval time. A caller that has not been through a
    gate can still reach this function; what it cannot do is write a row with no recorded
    approval, and that is the property the schema owns.
    """
    if quantity <= 0:
        return failed("inventory", INVENTORY_SOURCE_ID, "quantity must be a positive whole number")

    db_path = Path(db_path)
    if not db_path.exists():
        return failed("inventory", INVENTORY_SOURCE_ID, INVENTORY_UNAVAILABLE)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        order_id = place_order_row(
            conn,
            part_number=part_number,
            quantity=quantity,
            requested_by=requested_by,
            approved_by=approved_by,
            approved_at=approved_at,
            bearing_id=bearing_id,
        )
    except UnknownPartError:
        return failed(
            "inventory", INVENTORY_SOURCE_ID, f"no part with part number {part_number!r} exists"
        )
    except OrderRejectedError as exc:
        # The database's own reason is preserved: an oversell and a missing approval field
        # are both rejections here, and which one it was is exactly what the caller needs.
        return failed("inventory", INVENTORY_SOURCE_ID, f"{ORDER_FAILED}: {exc}")
    except (InventoryError, sqlite3.Error, ValueError) as exc:
        return failed("inventory", INVENTORY_SOURCE_ID, f"{ORDER_FAILED}: {exc}")
    finally:
        conn.close()

    return ok(
        "inventory",
        INVENTORY_SOURCE_ID,
        {
            "order_id": order_id,
            "part_number": part_number,
            "quantity": quantity,
            "bearing_id": bearing_id,
            "requested_by": requested_by,
            "approved_by": approved_by,
            "approved_at": approved_at,
            "status": "placed",
        },
    )
