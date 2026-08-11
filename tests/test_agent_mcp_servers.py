"""Tier-1 tests for the two MCP servers themselves (Issue #110, `docs/agent_design.md`
Sections 2, 8 and 10; the `place_order` argument surface fixed by Issue #125).

Two claims are under test here, and both are claims about *structure* rather than about
behaviour a model chose:

1. **The servers are genuinely separate.** Section 2 chose MCP over plain Python callables
   so that least privilege is a process boundary. The strongest form of that test is the
   real one: spawn each server as an actual subprocess over stdio, connect a real client,
   and show that a client configured for the read-only server cannot reach `place_order` --
   the same "verify against real processes, not just an in-process harness" discipline
   `src/serving/single_worker.py`'s tests already set for the single-worker constraint
   (Issue #84).
2. **The 8-tool-call cap fires at the server**, not in an agent loop that could forget to
   count or be talked out of counting.

No API key and no network: the stdio transport is a pair of pipes, the serving API is never
started (its tools fail closed against a dead port, which is fine -- what is being measured
here is the cap and the tool surface, not the tools' payloads), and the inventory database
is built in `tmp_path`.

**Issue #125's approval-token wrinkle for the subprocess tests.** `ApprovalTokenStore` is
in-process, no-durable-state by design (#124) -- exactly right for the in-process
`server.call_tool()` tests below, where the test and the server share one Python process and
can mint into the same store `build_write_server` returns. It cannot be right for `_probe`,
which spawns `write_server.py` as a real OS subprocess (Section 2's actual security
boundary): that subprocess builds its own fresh, empty store in `main()`, and nothing this
test process mints is ever visible to it. Wiring a token minted by one process into a store
consumed by another is exactly the kind of orchestrator question Issue #125 excludes
("do not build the orchestrator wiring"), so `test_a_client_of_the_write_server_can_reach_
place_order_and_only_that` below asserts what *can* be shown across a real process boundary
without that wiring -- that the tool is genuinely reachable and executes real validation
(an `unknown`-token rejection), not that a full order can be placed through a bare subprocess
with no orchestrator behind it. Flagged here rather than silently narrowed elsewhere.

**Issue #132 lifted that restriction, and these tests deliberately keep exercising the
un-bridged shape.** `write_server.py` now takes `--token-bridge`, so a token minted in one
process *can* be consumed by a subprocess -- `tests/test_agent_token_bridge.py` is where that
is proved. `_probe` below still launches the server without the flag, which is exactly the
"its own fresh, empty store" behaviour described above, and asserting it stays that way is
worth a test in its own right: the bridge must be something a caller opts into, not the
default a bare `python -m src.agent.mcp.write_server` silently acquires.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters, stdio_client

from src.agent.executor.approval import ApprovalTokenStore
from src.agent.inventory.build_db import build_db
from src.agent.mcp.budget import BUDGET_EXHAUSTED, MAX_TOOL_CALLS, ToolCallBudget
from src.agent.mcp.readonly_server import READONLY_TOOL_NAMES
from src.agent.mcp.readonly_server import build_server as build_readonly_server
from src.agent.mcp.results import payload_of
from src.agent.mcp.write_server import WRITE_TOOL_NAMES
from src.agent.mcp.write_server import build_server as build_write_server

REPO_ROOT = Path(__file__).resolve().parents[1]
CLOSED_PORT_URL = "http://127.0.0.1:9"

# No `approved_by`/`approved_at` (Issue #125 -- those are no longer arguments at all) and no
# `approval_token` (minted per-test, per-store, by `_approved_order` below).
ORDER_ARGS = {
    "part_number": "BRG-6205-2RS",
    "quantity": 1,
    "requested_by": "tech-01",
}


def _approved_order(token_store: ApprovalTokenStore, approved_by: str = "supervisor-02") -> dict:
    """`ORDER_ARGS` plus a token freshly minted, in `token_store`, for exactly that order."""
    token = token_store.mint(
        ORDER_ARGS["part_number"], ORDER_ARGS["quantity"], None, approved_by
    )
    return {**ORDER_ARGS, "approval_token": token.token}


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "inventory.db"
    build_db(path)
    return path


def _no_hits(*args, **kwargs):
    return []


# --------------------------------------------------------------------------------------
# The tool surface of each server
# --------------------------------------------------------------------------------------


def test_the_read_only_server_registers_exactly_the_five_read_only_tools(db_path):
    server, _ = build_readonly_server(
        base_url=CLOSED_PORT_URL, db_path=db_path, search=_no_hits
    )
    names = [tool.name for tool in asyncio.run(server.list_tools())]

    assert names == list(READONLY_TOOL_NAMES)
    assert "find_similar_historical_pattern" in names, (
        "Section 2's fifth read-only tool, registered as of Issue #140 now that Section "
        "12's trajectory archive exists and is committed"
    )
    assert len(names) == 5


def test_the_write_server_registers_place_order_and_nothing_else(db_path):
    server, _, _ = build_write_server(db_path=db_path)

    assert [tool.name for tool in asyncio.run(server.list_tools())] == list(WRITE_TOOL_NAMES)


def test_the_two_servers_tool_sets_are_disjoint(db_path):
    readonly, _ = build_readonly_server(
        base_url=CLOSED_PORT_URL, db_path=db_path, search=_no_hits
    )
    write, _, _ = build_write_server(db_path=db_path)

    readonly_names = {tool.name for tool in asyncio.run(readonly.list_tools())}
    write_names = {tool.name for tool in asyncio.run(write.list_tools())}

    assert readonly_names & write_names == set()
    assert "place_order" not in readonly_names


def test_no_tool_schema_exposes_an_injected_dependency(db_path):
    """A tool's backing URL, database path and retrieval function are bound at registration
    time, so they are not arguments -- and an argument that is not in the schema is an
    argument no prompt can supply. `token_store` (Issue #125) joins `budget` here for the
    same reason -- the store is injected at registration, `approval_token` is the only
    approval-related thing a caller ever supplies."""
    readonly, _ = build_readonly_server(
        base_url=CLOSED_PORT_URL, db_path=db_path, search=_no_hits
    )
    write, _, _ = build_write_server(db_path=db_path)

    for server in (readonly, write):
        for tool in asyncio.run(server.list_tools()):
            properties = set(tool.input_schema.get("properties", {}))
            assert properties.isdisjoint(
                {"base_url", "db_path", "search", "budget", "token_store"}
            ), f"{tool.name} exposes an injected dependency: {properties}"


def test_place_orders_schema_no_longer_accepts_approved_by_or_approved_at(db_path):
    """Issue #125: `approved_by`/`approved_at` must be gone from the schema itself, not just
    ignored at runtime -- the whole point is that a model filling them in can no longer
    produce a satisfied-looking call. `approval_token` replaces them."""
    write, _, _ = build_write_server(db_path=db_path)

    (place_order_tool,) = asyncio.run(write.list_tools())
    properties = set(place_order_tool.input_schema.get("properties", {}))

    assert properties.isdisjoint({"approved_by", "approved_at"})
    assert "approval_token" in properties
    assert "approval_token" in place_order_tool.input_schema.get("required", [])


def test_calling_place_order_with_the_old_approved_by_shape_is_a_schema_level_rejection(
    db_path,
):
    """Not just 'the value is ignored': a call built the old way (no `approval_token`) never
    reaches the tool body at all -- it fails argument validation, which is a different kind
    of failure than the tool's own `is_error=True` result for a rejected order."""
    write, _, _ = build_write_server(db_path=db_path)

    old_style_call = {
        **ORDER_ARGS,
        "approved_by": "supervisor-02",
        "approved_at": "2026-08-05T09:00:00+00:00",
    }

    with pytest.raises(Exception) as exc_info:
        asyncio.run(write.call_tool("place_order", old_style_call))

    # A schema-validation failure, not a normal tool result -- distinguishes this from
    # `is_error=True`, which is a return value, not an exception.
    assert "approval_token" in str(exc_info.value)

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0
    finally:
        conn.close()


def test_no_tool_can_reset_the_budget(db_path):
    """A model that can reset its own cap does not have a cap, so `ToolCallBudget.reset`
    is reachable in-process only and is registered on neither server."""
    readonly, _ = build_readonly_server(
        base_url=CLOSED_PORT_URL, db_path=db_path, search=_no_hits
    )
    write, _, _ = build_write_server(db_path=db_path)

    names = [t.name for t in asyncio.run(readonly.list_tools())]
    names += [t.name for t in asyncio.run(write.list_tools())]
    assert not any("reset" in name or "budget" in name for name in names)


# --------------------------------------------------------------------------------------
# The 8-tool-call cap, enforced at the server
# --------------------------------------------------------------------------------------


def test_the_ninth_tool_call_is_refused_by_the_server(db_path):
    calls: list[str] = []

    def counting_search(query, limit=5, source_type=None):
        calls.append(query)
        return []

    server, budget = build_readonly_server(
        base_url=CLOSED_PORT_URL, db_path=db_path, search=counting_search
    )

    async def run() -> list:
        return [
            await server.call_tool("search_documentation", {"query": f"q{i}"})
            for i in range(MAX_TOOL_CALLS + 1)
        ]

    results = asyncio.run(run())

    assert MAX_TOOL_CALLS == 8
    assert all(r.is_error is False for r in results[:MAX_TOOL_CALLS])
    refused = results[MAX_TOOL_CALLS]
    assert refused.is_error is True
    assert payload_of(refused)["error"] == BUDGET_EXHAUSTED
    # The refusal happened *instead of* the work, not after it.
    assert len(calls) == MAX_TOOL_CALLS
    assert budget.remaining == 0


def test_the_cap_counts_across_every_tool_on_a_server_not_per_tool(db_path):
    """Section 10 case 7's unbounded loop does not have to hammer one tool. A per-tool
    counter would give it 8 calls each."""
    server, _ = build_readonly_server(base_url=CLOSED_PORT_URL, db_path=db_path, search=_no_hits)

    async def run() -> list:
        names = ["search_documentation", "check_inventory", "get_bearing_status"]
        return [
            await server.call_tool(
                names[i % len(names)], {"query": "q"} if i % len(names) == 0 else {}
            )
            for i in range(MAX_TOOL_CALLS + 1)
        ]

    results = asyncio.run(run())

    assert payload_of(results[MAX_TOOL_CALLS])["error"] == BUDGET_EXHAUSTED


def test_the_write_server_refuses_a_ninth_order_and_writes_no_ninth_row(db_path):
    server, _, token_store = build_write_server(db_path=db_path)

    async def run() -> list:
        # A fresh token per call -- each is single-use, so the ninth refusal must be the
        # budget firing, not the eighth token being replayed.
        return [
            await server.call_tool("place_order", _approved_order(token_store))
            for _ in range(MAX_TOOL_CALLS + 1)
        ]

    results = asyncio.run(run())

    assert payload_of(results[MAX_TOOL_CALLS])["error"] == BUDGET_EXHAUSTED

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == MAX_TOOL_CALLS
    finally:
        conn.close()


def test_a_refused_call_still_costs_a_call(db_path):
    """Otherwise a caller that keeps hammering a refused server walks the counter backwards
    by failing on purpose."""
    budget = ToolCallBudget(limit=1)
    budget.guard("inventory", "x")
    budget.guard("inventory", "x")

    assert budget.used == 2 and budget.exhausted


def test_reset_starts_a_new_questions_budget(db_path):
    server, budget = build_readonly_server(
        base_url=CLOSED_PORT_URL, db_path=db_path, search=_no_hits
    )

    async def call() -> object:
        return await server.call_tool("search_documentation", {"query": "q"})

    for _ in range(MAX_TOOL_CALLS + 1):
        asyncio.run(call())
    assert budget.exhausted

    budget.reset()

    assert asyncio.run(call()).is_error is False


# --------------------------------------------------------------------------------------
# The structural claim, against real processes
# --------------------------------------------------------------------------------------


def _server_params(module: str, db_path: Path) -> StdioServerParameters:
    args = ["-m", module, "--db-path", str(db_path)]
    if module.endswith("readonly_server"):
        args += ["--url", CLOSED_PORT_URL]
    return StdioServerParameters(
        command=sys.executable,
        args=args,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )


async def _probe(module: str, db_path: Path, tool_name: str, arguments: dict) -> dict:
    """Start `module` as a real subprocess, speak MCP over its stdio pipes, and report what
    the client can see and do."""
    async with stdio_client(_server_params(module, db_path)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = [tool.name for tool in (await session.list_tools()).tools]
            result = await session.call_tool(tool_name, arguments)
            return {
                "tools": listed,
                "is_error": bool(result.is_error),
                "text": result.content[0].text if result.content else "",
            }


def test_a_client_of_the_read_only_server_cannot_reach_place_order(db_path):
    """The structural claim Section 2 makes, tested the way Section 10 case 6 asks for it:
    on the client's actual connection, not on whether a model happened not to try.

    A real subprocess, a real stdio transport, a real MCP client. `place_order` is not
    hidden or disabled on this connection -- it does not exist on it, so the server answers
    "Unknown tool" to a perfectly well-formed order.
    """
    seen = asyncio.run(
        _probe(
            "src.agent.mcp.readonly_server",
            db_path,
            "place_order",
            {**ORDER_ARGS, "approval_token": "irrelevant-the-tool-does-not-exist-here"},
        )
    )

    assert seen["tools"] == list(READONLY_TOOL_NAMES)
    assert "place_order" not in seen["tools"]
    assert seen["is_error"] is True
    assert "Unknown tool: place_order" in seen["text"]

    # And nothing was written: the order the read-only client asked for does not exist.
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0
    finally:
        conn.close()


def test_a_client_of_the_write_server_can_reach_place_order_and_only_that(db_path):
    """The other half of the same claim -- without it, the test above would also pass if
    `place_order` were broken everywhere.

    This subprocess's `ApprovalTokenStore` is its own, built fresh inside `main()`, and
    nothing this test process mints is visible to it (see the module docstring's note on
    Issue #125's process-boundary wrinkle) -- so this cannot show a full order succeeding.
    What it *can* show, and what actually distinguishes "reachable" from "broken", is that
    the call reaches real validation logic and is rejected for a substantive reason
    (`unknown` token) rather than for not existing on this connection at all -- the
    read-only-server test above gets "Unknown tool: place_order"; this one must not.
    """
    seen = asyncio.run(
        _probe(
            "src.agent.mcp.write_server",
            db_path,
            "place_order",
            {**ORDER_ARGS, "approval_token": "no-such-token-was-ever-minted-here"},
        )
    )

    assert seen["tools"] == list(WRITE_TOOL_NAMES)
    assert seen["is_error"] is True
    assert "Unknown tool" not in seen["text"]
    assert "approval token is not recognized" in seen["text"]

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0
    finally:
        conn.close()


def test_a_client_of_the_write_server_cannot_reach_the_read_only_tools(db_path):
    """Least privilege runs both ways: the executor sees the order parameters and the
    approval, not the fleet's monitoring state (Section 5's "Sees" row)."""
    seen = asyncio.run(
        _probe("src.agent.mcp.write_server", db_path, "get_bearing_status", {})
    )

    assert seen["is_error"] is True
    assert "Unknown tool: get_bearing_status" in seen["text"]
