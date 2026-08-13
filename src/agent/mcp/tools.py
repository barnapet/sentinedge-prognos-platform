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

from src.agent.executor import approval
from src.agent.executor.approval import ApprovalTokenStore, TokenError
from src.agent.inventory.build_db import DB_PATH
from src.agent.inventory.orders import (
    InventoryError,
    OrderRejectedError,
    UnknownPartError,
    place_order as place_order_row,
)
from src.agent.inventory.query import find_parts
from src.agent.mcp.results import (
    ARCHIVE_UNAVAILABLE,
    DOCS_INDEX_UNREACHABLE,
    INVENTORY_UNAVAILABLE,
    ORDER_FAILED,
    SERVICE_UNREACHABLE,
    TRAJECTORY_UNUSABLE,
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
    get_history,
    post_predict,
)
from src.agent.rag.index import COLLECTION_NAME
from src.agent.rag.retrieval import DEFAULT_LIMIT, RetrievedChunk
from src.agent.rag.retrieval import search as default_search
from src.agent.similarity.archive import (
    CAVEAT,
    archive_source_id,
    best_match_or_none,
    load_archive,
    query_matrix,
    rank_against_archive,
)

INVENTORY_SOURCE_ID = "data/agent/inventory.db"
# `search_documentation` queries a live Qdrant service at request time, so `live_endpoint`
# is the vocabulary value that describes *the retrieval*; the citable per-chunk ids each
# carry their own `decision_doc`/`public_reference` block inside `data.results` (Section 6:
# "every retrieved chunk carries a stable `chunk_id`"). Flagged in the PR for #110, since
# Section 2's five-value vocabulary has no term for a search index and this is the closest
# reading rather than a new value.
DOCS_SOURCE_ID = f"SEARCH {COLLECTION_NAME}"

SearchFn = Callable[..., Sequence[RetrievedChunk]]

# `docs/agent_design.md` Section 12's "the last 50 requests".
QUERY_WINDOW = 50
# Below `src.features.extraction.ROLLING_WINDOW` (10) a trajectory is shorter than the
# smoothing window that produced it, so it carries less than one window's worth of
# independent shape -- a distance computed from it would be a number without a meaning.
MIN_QUERY_WINDOW = 10
# Used only when the archive itself could not be read, so its content hash -- which the
# real `source_id` is derived from -- is exactly what is unavailable. The envelope still
# needs a `source_id` to say *which* source failed (`results.py`'s "one shape" rule).
ARCHIVE_SOURCE_ID_FALLBACK = "trajectory_archive@unavailable"


def _call_retrying_once_on_unreachable(call: Callable[[], Any]) -> Any:
    """Call `call()`; if it raises `ServingUnreachable`, call it exactly once more.

    `docs/agent_design.md` Section 8's tier-3 table: "Tool times out -> Exactly one retry,
    then degrade; not an infinite loop." This is that retry, shared by all three
    live-serving call sites (Issue #154) so "all three sites get the same policy" is a
    property of using this function rather than three hand-copied try/excepts that could
    drift apart.

    Only `ServingUnreachable` is retried -- "nothing answered" is the transient case Section
    8 means by "tool times out". `ServingRejected` (the service is up and said no) and every
    other exception propagate from the first attempt unchanged: retrying an actual rejection
    doesn't help, and this function does not decide what a site does with either failure --
    that stays each site's own `except` clause, exactly as before this change.

    No backoff, no jitter, no budget beyond the one extra attempt: the second call's own
    `ServingUnreachable` (or anything else) propagates straight out, uncaught here.
    """
    try:
        return call()
    except ServingUnreachable:
        return call()


# Plain-language rejections for each of approval.py's four closed `TokenError` reasons
# (Issue #124). Keyed off the module's own constants rather than the literal strings, so a
# typo here would be a `KeyError` at test time, not a silently-wrong message.
_APPROVAL_TOKEN_ERROR_MESSAGES = {
    approval.UNKNOWN: "the approval token is not recognized",
    approval.EXPIRED: "the approval token has expired",
    approval.ALREADY_USED: "the approval token has already been used",
    approval.SCOPE_MISMATCH: (
        "the approval token does not match this order's part number, quantity, and bearing"
    ),
}


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
        body = _call_retrying_once_on_unreachable(lambda: get_drift(base_url=base_url))
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
        body = _call_retrying_once_on_unreachable(
            lambda: post_predict(bearing_id, list(signal), base_url=base_url)
        )
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


def trajectory_source_id() -> str:
    """The archive's citable id, or a fallback when the archive cannot be read.

    Needed outside the tool body too: `readonly_server.py` charges the budget *before*
    running the body, and that refusal envelope still has to name a source. Never raises,
    because a missing artifact must reach the model as a readable failure rather than as
    `Error executing tool ...` (`results.py`'s rule).
    """
    try:
        return archive_source_id()
    except (OSError, ValueError, KeyError):
        return ARCHIVE_SOURCE_ID_FALLBACK


def find_similar_historical_pattern(
    bearing_id: str,
    window: int = QUERY_WINDOW,
    *,
    base_url: str = DEFAULT_BASE_URL,
) -> CallToolResult:
    """Rank a live bearing's recent trajectory against the three archived experiments
    (`docs/agent_design.md` Section 12, Issue #140).

    Two halves, from two places: the query is the bearing's last `window` values of
    Section 12's three channels, read over HTTP from the running serving process; the
    references are `models/trajectory_archive.parquet`, committed. Neither half is invented
    when it is missing, which is what the four non-error early returns below are for.

    **`best_match` is `null` unless the closest reference clears the calibrated threshold.**
    Section 12: "always returning a winner out of three is how 'most resembles' quietly
    becomes a false claim about an unfamiliar bearing." `ranked` still carries all three
    distances in that case, so the answer can say *what* it was closest to and that it was
    not close enough -- a refusal with its evidence attached, not a blank.

    `n_references` and `caveat` are present on every successful result, per Section 12's
    output contract; the critic's risky-claim check (Section 6) needs the sample size to be
    on the result rather than remembered.
    """
    source_id = trajectory_source_id()
    try:
        trajectories = load_archive()
    except (OSError, ValueError, KeyError):
        # No archive means no comparison. Reported as a failure rather than as an empty
        # ranking: "nothing resembles this" and "I could not look" are different answers,
        # and only one of them should reach a technician as information.
        return failed("trajectory_match", source_id, ARCHIVE_UNAVAILABLE)

    if window < MIN_QUERY_WINDOW:
        return failed(
            "trajectory_match",
            source_id,
            f"window must be at least {MIN_QUERY_WINDOW} points, got {window}",
        )

    try:
        body = _call_retrying_once_on_unreachable(
            lambda: get_history(bearing_id, window, base_url=base_url)
        )
    except ServingUnreachable:
        return failed("trajectory_match", source_id, SERVICE_UNREACHABLE)
    except ServingRejected as exc:
        return failed(
            "trajectory_match",
            source_id,
            f"the prediction service rejected the request (HTTP {exc.status_code})",
        )

    if not body.get("found", False):
        # Same structured not-found `get_bearing_status` returns, for the same reason
        # (Section 10 case 1): an untracked bearing must arrive as a fact about which
        # bearings exist, never as a comparison against a trajectory nobody recorded.
        return ok(
            "trajectory_match",
            source_id,
            {
                "bearing_id": bearing_id,
                "found": False,
                "tracked_bearings": body.get("tracked_bearings", []),
                "n_references": len(trajectories),
                "caveat": CAVEAT,
            },
        )

    channels = body.get("channels", {})
    n_points = int(body.get("n_points", 0))
    common = {
        "bearing_id": bearing_id,
        "found": True,
        "n_references": len(trajectories),
        "query_window": n_points,
        "baseline_status": body.get("baseline_status"),
        "caveat": CAVEAT,
    }
    if n_points < MIN_QUERY_WINDOW:
        # A real bearing with too little history yet. Not an error -- the honest answer is
        # "ask again later", and inventing a comparison from 6 points would be worse than
        # saying so.
        return ok(
            "trajectory_match",
            source_id,
            {
                **common,
                "best_match": None,
                "no_match_reason": (
                    f"only {n_points} window(s) recorded for this bearing; at least "
                    f"{MIN_QUERY_WINDOW} are needed for a shape comparison"
                ),
                "ranked": [],
            },
        )

    try:
        query = query_matrix(channels)
        ranked = rank_against_archive(query, trajectories)
    except (KeyError, ValueError) as exc:
        return failed("trajectory_match", source_id, f"{TRAJECTORY_UNUSABLE}: {exc}")

    best, no_match_reason = best_match_or_none(ranked)
    return ok(
        "trajectory_match",
        source_id,
        {**common, "best_match": best, "no_match_reason": no_match_reason, "ranked": ranked},
    )


# --------------------------------------------------------------------------------------
# Write-capable tool
# --------------------------------------------------------------------------------------


def place_order(
    part_number: str,
    quantity: int,
    requested_by: str,
    approval_token: str,
    bearing_id: str | None = None,
    *,
    token_store: ApprovalTokenStore,
    db_path: Path = DB_PATH,
) -> CallToolResult:
    """Place one order: decrement stock and insert an `orders` row, in one transaction.

    Wraps `src/agent/inventory/orders.place_order` (#101) unchanged -- the oversell abort
    and the `NOT NULL` constraints on `approved_by`/`approved_at` stay where they are, in
    the schema, which `docs/agent_design.md` Section 7 calls the second, independent
    enforcement of the approval gate.

    **The approval gate is now enforced here (Issue #125).** `approved_by`/`approved_at`
    are no longer caller-supplied arguments -- Issue #110 and `docs/agent_design.md`
    Section 10 named that gap explicitly: a tool schema that lets the model fill in its own
    approver/timestamp turns a successful injection into a satisfied constraint. Instead the
    caller supplies `approval_token`, which is validated -- exists, unexpired, scope match,
    unconsumed -- against `token_store` (`src/agent/executor/approval.py`, Issue #124,
    unmodified here) via `token_store.consume(...)`. `approved_by`/`approved_at` are then
    *derived* from the `ApprovedOrder` that validation returns, never from an argument. A
    rejected token writes nothing -- validation happens before the database is touched.
    `requested_by` is unaffected by any of this and stays an ordinary caller-supplied
    argument.
    """
    if quantity <= 0:
        return failed("inventory", INVENTORY_SOURCE_ID, "quantity must be a positive whole number")

    db_path = Path(db_path)
    if not db_path.exists():
        return failed("inventory", INVENTORY_SOURCE_ID, INVENTORY_UNAVAILABLE)

    consumed = token_store.consume(approval_token, part_number, quantity, bearing_id)
    if isinstance(consumed, TokenError):
        return failed(
            "inventory",
            INVENTORY_SOURCE_ID,
            f"{ORDER_FAILED}: {_APPROVAL_TOKEN_ERROR_MESSAGES[consumed.reason]}",
        )
    # `ApprovedOrder.approved_at` is a `datetime` (Issue #124); `orders.place_order`'s
    # `approved_at` column is `TEXT NOT NULL` (Section 7) -- the one `.isoformat()` seam
    # PR #128 flagged as needed here.
    approved_by = consumed.approved_by
    approved_at = consumed.approved_at.isoformat()

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
