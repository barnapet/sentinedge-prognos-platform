"""Tier-1 tests for Agent C's client (Issue #126, `docs/agent_design.md` Sections 1, 5 and
10). No API key, no model call, no network beyond a local stdio pipe -- see the module
docstring in `src/agent/executor/client.py` for why there is no model call to gate here at
all.

What is asserted here is the wiring and the two least-privilege claims Section 10 case 6
makes about this agent, tested against real processes the same way `test_agent_mcp_servers.py`
and `test_agent_answerer.py` already test the other two agents: the client's tool surface is
exactly `place_order`, it cannot reach any read-only tool, and a well-formed order is refused
for a substantive reason (never "Unknown tool"). The other half of Section 10 case 6 --
that the read-only clients (Agent A, and Agent B which holds no tools at all) cannot reach
`place_order` -- is already covered by `test_agent_answerer.py::
test_the_answerers_own_session_cannot_reach_place_order` and `test_agent_mcp_servers.py::
test_a_client_of_the_read_only_server_cannot_reach_place_order`; it is not duplicated here.

A real, successful order, and each of Issue #125's four token-rejection reasons, are shown
here against `write_server.build_server()`'s in-process `MCPServer`, sharing a real
`ApprovalTokenStore` with the test -- see `client.py`'s docstring on why `execute_order` takes
a session rather than opening its own.

**Why in-process here, when Issue #132 made the subprocess route work.** When this module was
written it was the only route: the subprocess built its own empty store and nothing a test
minted was visible to it. #132's `--token-bridge` removed that wall, and
`tests/test_agent_token_bridge.py` proves a real token now crosses a real process boundary.
These tests stay in-process anyway, because what they are about is `execute_order`'s own
behaviour -- that it returns `OrderPlaced` for a good order and `OrderRejected` carrying the
tool's reason for each rejection -- and a shared in-process store is the shortest path to
that. Spawning a subprocess and a socket per case would test the bridge a second time and
`execute_order` no better.
"""
from __future__ import annotations

import ast
import asyncio
import dataclasses
import sqlite3
import typing
from pathlib import Path

import pytest

from src.agent.executor import client
from src.agent.executor.approval import ApprovalTokenStore
from src.agent.executor.client import (
    OrderPlaced,
    OrderRejected,
    OrderRequest,
    _assert_write_only_surface,
    execute_order,
    execute_order_sync,
    execute_order_via_write_server,
    write_server_params,
    write_tools,
)
from src.agent.inventory.build_db import build_db
from src.agent.mcp.write_server import WRITE_TOOL_NAMES
from src.agent.mcp.write_server import build_server as build_write_server

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIENT_SOURCE = REPO_ROOT / "src" / "agent" / "executor" / "client.py"


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "inventory.db"
    build_db(path)
    return path


def _some_part(db_path: Path) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT part_number, bearing_type FROM parts "
            "WHERE bearing_type IS NOT NULL ORDER BY part_number LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    return {"part_number": row[0], "bearing_type": row[1]}


# --------------------------------------------------------------------------------------
# Task 2/3: the tool surface, against a real server process
# --------------------------------------------------------------------------------------


def test_the_executor_launches_the_write_server_and_only_that():
    """Static half of the least-privilege claim: the four read-only tools live on a
    different module, and this agent never names it."""
    params = write_server_params()

    assert params.args[:2] == ["-m", "src.agent.mcp.write_server"]
    assert not any("readonly_server" in arg for arg in params.args)


def test_the_executors_tools_are_exactly_place_order(db_path):
    """The real thing, over a real stdio transport to a real subprocess -- the pattern
    `answerer.py`'s equivalent test already sets for the read-only client, mirrored here."""

    async def run() -> list[str]:
        async with write_tools(db_path=db_path) as session:
            return [tool.name for tool in (await session.list_tools()).tools]

    names = asyncio.run(run())

    assert names == list(WRITE_TOOL_NAMES) == ["place_order"]


def test_the_executors_own_session_cannot_reach_a_read_only_tool(db_path):
    """Section 5's "Sees" row, the direction that matters for Agent C: the order parameters
    and the approval token, never the fleet's monitoring state. `get_bearing_status` is not
    filtered out of this connection -- it does not exist on it, so a well-formed call comes
    back "Unknown tool", exactly the shape `test_agent_mcp_servers.py`'s bare-subprocess
    version of this claim already asserts, now through this module's own client."""

    async def run() -> dict:
        async with write_tools(db_path=db_path) as session:
            result = await session.call_tool("get_bearing_status", {})
            return {
                "is_error": bool(result.is_error),
                "text": result.content[0].text if result.content else "",
            }

    seen = asyncio.run(run())

    assert seen["is_error"] is True
    assert "Unknown tool: get_bearing_status" in seen["text"]


def test_a_well_formed_order_over_the_real_subprocess_is_refused_for_a_substantive_reason(
    db_path,
):
    """Distinguishes "reachable but refused" from "not reachable at all" -- without this,
    the read-only-tool test above would also pass if `place_order` were broken everywhere.
    The subprocess's token store is its own and empty (see the module docstring), so this
    cannot show a full order succeeding -- what it shows is that the call reaches real
    validation and is rejected for `unknown`, not for the tool not existing here."""
    order = OrderRequest(
        part_number="BRG-6205-2RS",
        quantity=1,
        approval_token="no-such-token-was-ever-minted-here",
        requested_by="tech-01",
    )

    result = asyncio.run(execute_order_via_write_server(order, db_path=db_path))

    assert isinstance(result, OrderRejected)
    assert "not recognized" in result.reason

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0
    finally:
        conn.close()


def test_the_write_only_surface_check_passes_for_place_order_alone():
    _assert_write_only_surface(["place_order"])  # must not raise


def test_the_write_only_surface_check_rejects_anything_else():
    """Task 3's "raises if the connected server offers anything outside {place_order}",
    tested directly: there is no way to make the real, unmodified `write_server.py` offer a
    second tool to probe this against, so the check is exercised in isolation instead of
    inside `write_tools()`."""
    with pytest.raises(RuntimeError, match="get_bearing_status"):
        _assert_write_only_surface(["place_order", "get_bearing_status"])


# --------------------------------------------------------------------------------------
# Task 4: the fixed-schema input, and that nothing else can reach the entry point
# --------------------------------------------------------------------------------------


def test_order_request_has_exactly_the_fixed_schema_fields():
    fields = {f.name for f in dataclasses.fields(OrderRequest)}

    assert fields == {"part_number", "quantity", "approval_token", "requested_by", "bearing_id"}


def test_order_request_is_frozen():
    order = OrderRequest(
        part_number="BRG-6205-2RS", quantity=1, approval_token="t", requested_by="tech-01"
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        order.quantity = 2  # type: ignore[misc]


def test_bearing_id_is_optional_and_defaults_to_none():
    order = OrderRequest(part_number="X", quantity=1, approval_token="t", requested_by="tech-01")

    assert order.bearing_id is None
    assert order.as_tool_arguments()["bearing_id"] is None


def test_the_entry_points_only_content_carrying_parameter_is_the_fixed_schema_record():
    """The entry point's only parameter that could carry a `Draft`, a question, or any other
    free text is `order: OrderRequest` -- every other parameter is infrastructure (`db_path`,
    `token_bridge`, a session), asserted by resolved type hints rather than by behaviour.

    The infrastructure set is enumerated rather than pattern-matched **so that adding a
    parameter to this entry point cannot pass silently** -- it must be looked at and named
    here. `token_bridge` (Issue #132) was added that way: it is a `Path` to a Unix socket,
    a rendezvous address that carries no order content and no credential.
    """
    hints = typing.get_type_hints(execute_order_via_write_server)

    assert hints["order"] is OrderRequest
    other = {name: hint for name, hint in hints.items() if name not in ("order", "return")}
    assert other == {"db_path": Path | None, "token_bridge": Path | None}


def test_the_module_imports_nothing_from_the_answerer_or_the_critic():
    """No code path in this module can reach a `Draft` or a critic verdict type -- checked at
    the import level, since that is the level at which "this module never sees Agent A's
    output" is actually true or false."""
    tree = ast.parse(CLIENT_SOURCE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)

    assert not any("answerer" in name or "critic" in name for name in imported)


# --------------------------------------------------------------------------------------
# Task 5/6: the call itself, against a real shared ApprovalTokenStore
# --------------------------------------------------------------------------------------


def test_given_a_valid_unconsumed_token_the_order_succeeds_and_matches_the_record(db_path):
    part = _some_part(db_path)
    server, _, token_store = build_write_server(db_path=db_path)
    token = token_store.mint(part["part_number"], 1, "2nd_test-demo", "supervisor-02")
    order = OrderRequest(
        part_number=part["part_number"],
        quantity=1,
        approval_token=token.token,
        requested_by="tech-01",
        bearing_id="2nd_test-demo",
    )

    result = asyncio.run(execute_order(order, server))

    assert isinstance(result, OrderPlaced)
    assert result.part_number == order.part_number
    assert result.quantity == order.quantity
    assert result.bearing_id == order.bearing_id
    assert result.requested_by == order.requested_by
    assert result.approved_by == "supervisor-02"
    assert result.approved_at == token.approved_at.isoformat()
    assert isinstance(result.order_id, int)

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1
    finally:
        conn.close()


def test_an_unknown_token_is_reported_as_a_rejection_not_raised(db_path):
    part = _some_part(db_path)
    server, _, _ = build_write_server(db_path=db_path)
    order = OrderRequest(
        part_number=part["part_number"],
        quantity=1,
        approval_token="never-minted",
        requested_by="tech-01",
    )

    result = asyncio.run(execute_order(order, server))

    assert isinstance(result, OrderRejected)
    assert "not recognized" in result.reason

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0
    finally:
        conn.close()


def test_an_already_used_token_is_reported_as_a_rejection_not_raised(db_path):
    part = _some_part(db_path)
    server, _, token_store = build_write_server(db_path=db_path)
    token = token_store.mint(part["part_number"], 1, None, "supervisor-02")
    order = OrderRequest(
        part_number=part["part_number"],
        quantity=1,
        approval_token=token.token,
        requested_by="tech-01",
    )

    first = asyncio.run(execute_order(order, server))
    second = asyncio.run(execute_order(order, server))

    assert isinstance(first, OrderPlaced)
    assert isinstance(second, OrderRejected)
    assert "already been used" in second.reason

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1
    finally:
        conn.close()


def test_a_scope_mismatched_token_is_reported_as_a_rejection_not_raised(db_path):
    """A token minted for a different quantity than the record presents -- otherwise
    identical and otherwise valid -- must still be refused, mirroring
    `test_agent_mcp_tools.py::test_a_scope_mismatched_token_is_rejected_and_orders_is_unchanged`
    at this module's own layer."""
    part = _some_part(db_path)
    server, _, token_store = build_write_server(db_path=db_path)
    token = token_store.mint(part["part_number"], 1, None, "supervisor-02")
    mismatched = OrderRequest(
        part_number=part["part_number"],
        quantity=2,  # minted for 1
        approval_token=token.token,
        requested_by="tech-01",
    )

    result = asyncio.run(execute_order(mismatched, server))

    assert isinstance(result, OrderRejected)
    assert "does not match" in result.reason

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0
    finally:
        conn.close()


def test_an_expired_token_is_reported_as_a_rejection_not_raised(db_path):
    from datetime import datetime, timedelta

    from src.agent.executor.approval import TOKEN_LIFETIME

    part = _some_part(db_path)
    now = datetime(2026, 8, 5, 9, 0, 0)
    clock = [now]
    server, _, token_store = build_write_server(
        db_path=db_path, token_store=ApprovalTokenStore(clock=lambda: clock[0])
    )
    token = token_store.mint(part["part_number"], 1, None, "supervisor-02")
    clock[0] = now + TOKEN_LIFETIME + timedelta(seconds=1)
    order = OrderRequest(
        part_number=part["part_number"],
        quantity=1,
        approval_token=token.token,
        requested_by="tech-01",
    )

    result = asyncio.run(execute_order(order, server))

    assert isinstance(result, OrderRejected)
    assert "expired" in result.reason

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0
    finally:
        conn.close()


def test_execute_order_sync_wraps_the_async_path(db_path, monkeypatch):
    """The synchronous entry point is the thin `asyncio.run` wrapper it claims to be, not a
    second implementation that could drift from the one every other test exercises."""
    called = {}

    async def fake_execute_order_via_write_server(order, **kwargs):
        called["order"] = order
        called["kwargs"] = kwargs
        return OrderRejected(reason="stub")

    monkeypatch.setattr(
        client, "execute_order_via_write_server", fake_execute_order_via_write_server
    )
    order = OrderRequest(
        part_number="X", quantity=1, approval_token="t", requested_by="tech-01"
    )

    result = execute_order_sync(order, db_path=db_path)

    assert result == OrderRejected(reason="stub")
    assert called["order"] is order
    assert called["kwargs"] == {"db_path": db_path}
