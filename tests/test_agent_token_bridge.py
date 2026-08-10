"""Tier-1 tests for the approval-token bridge (Issue #132, `docs/agent_design.md` Sections 0,
2 and 5). No API key, no model call, no network beyond a local Unix socket and a local stdio
pipe.

**The claim this module exists to prove, which nothing in this repo demonstrated before:
a token minted in one process can be spent by a `place_order` running in another.** Until
#132, `write_server.py`'s `main()` built its own empty store, so the only thing a real
subprocess could show was an `unknown`-token rejection --
`tests/test_agent_mcp_servers.py`'s module docstring says exactly that, and
`tests/test_agent_executor_client.py`'s says it again. The test named
`test_a_token_minted_here_is_consumed_by_a_place_order_in_another_process` below is the one
that was missing.

Everything here runs against real processes and real objects: a real
`ApprovalTokenStore` (#124, unmodified), a real `python -m src.agent.mcp.write_server`
subprocess over real stdio, a real Unix socket between them, and a real `tmp_path` SQLite
database whose rows are read back to check what actually happened.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import sqlite3
import stat
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.agent.executor.approval import (
    ALREADY_USED,
    EXPIRED,
    SCOPE_MISMATCH,
    TOKEN_LIFETIME,
    UNKNOWN,
    ApprovalTokenStore,
    ApprovedOrder,
    TokenError,
)
from src.agent.executor.client import (
    OrderPlaced,
    OrderRejected,
    OrderRequest,
    execute_order_via_write_server,
)
from src.agent.executor.token_bridge import (
    RemoteTokenStore,
    TokenConsumer,
    serve_token_store,
    token_store_for_bridge,
)
from src.agent.inventory.build_db import build_db


@pytest.fixture
def db_path(tmp_path) -> Path:
    path = tmp_path / "inventory.db"
    build_db(path)
    return path


def _real_part(db_path: Path) -> str:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT part_number FROM parts ORDER BY part_number LIMIT 1"
        ).fetchone()[0]
    finally:
        conn.close()


def _orders(db_path: Path) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT part_number, quantity, bearing_id, requested_by, approved_by FROM orders"
        ).fetchall()
    finally:
        conn.close()


# --------------------------------------------------------------------------------------
# The claim: a real token across a real process boundary
# --------------------------------------------------------------------------------------


def test_a_token_minted_here_is_consumed_by_a_place_order_in_another_process(db_path):
    """**Issue #132's central deliverable.** Mint in this process; place the order through a
    `python -m src.agent.mcp.write_server` subprocess reached over stdio; read the row back
    out of the database.

    Every part of this is the real thing: the subprocess is a separate OS process with its
    own memory, it holds a `RemoteTokenStore` and not the store the token lives in, and the
    only thing that crosses between them is one `consume` call over a Unix socket.
    """
    part = _real_part(db_path)
    store = ApprovalTokenStore()
    token = store.mint(part, 2, "2nd_test-demo", "supervisor-02")

    async def run():
        async with serve_token_store(store) as socket_path:
            order = OrderRequest(
                part_number=part,
                quantity=2,
                approval_token=token.token,
                requested_by="tech-01",
                bearing_id="2nd_test-demo",
            )
            return await execute_order_via_write_server(
                order, db_path=db_path, token_bridge=socket_path
            )

    result = asyncio.run(run())

    assert isinstance(result, OrderPlaced), getattr(result, "reason", result)
    assert result.approved_by == "supervisor-02"
    assert result.approved_at == token.approved_at.isoformat()
    assert _orders(db_path) == [(part, 2, "2nd_test-demo", "tech-01", "supervisor-02")]


def test_the_token_is_marked_consumed_in_this_process_not_in_a_copy(db_path):
    """The single-use invariant stays where #124 put it: on the one authoritative record.

    This is the property that rules out the rejected "pre-seed the subprocess with a copy of
    the token" alternative -- there, `consumed` would flip on the subprocess's copy and this
    assertion would fail.
    """
    part = _real_part(db_path)
    store = ApprovalTokenStore()
    token = store.mint(part, 1, None, "supervisor-02")
    assert token.consumed is False

    async def run():
        async with serve_token_store(store) as socket_path:
            order = OrderRequest(part, 1, token.token, "tech-01")
            return await execute_order_via_write_server(
                order, db_path=db_path, token_bridge=socket_path
            )

    assert isinstance(asyncio.run(run()), OrderPlaced)
    assert token.consumed is True


def test_a_second_order_on_the_same_token_is_refused_across_the_boundary(db_path):
    """Single-use survives the round trip: the second subprocess asks the same store, which
    has already spent the token."""
    part = _real_part(db_path)
    store = ApprovalTokenStore()
    token = store.mint(part, 1, None, "supervisor-02")

    async def run():
        async with serve_token_store(store) as socket_path:
            order = OrderRequest(part, 1, token.token, "tech-01")
            first = await execute_order_via_write_server(
                order, db_path=db_path, token_bridge=socket_path
            )
            second = await execute_order_via_write_server(
                order, db_path=db_path, token_bridge=socket_path
            )
            return first, second

    first, second = asyncio.run(run())

    assert isinstance(first, OrderPlaced)
    assert isinstance(second, OrderRejected)
    assert "already been used" in second.reason
    assert len(_orders(db_path)) == 1


def test_a_scope_mismatched_token_is_refused_across_the_boundary(db_path):
    """The scope tuple is compared by the authoritative store, so altering a parameter on the
    way to the subprocess cannot help -- Section 5's "an approval for one part cannot
    authorize another", now demonstrated across processes rather than in one."""
    part = _real_part(db_path)
    store = ApprovalTokenStore()
    token = store.mint(part, 1, None, "supervisor-02")

    async def run():
        async with serve_token_store(store) as socket_path:
            order = OrderRequest(part, 2, token.token, "tech-01")  # minted for 1
            return await execute_order_via_write_server(
                order, db_path=db_path, token_bridge=socket_path
            )

    result = asyncio.run(run())

    assert isinstance(result, OrderRejected)
    assert "does not match" in result.reason
    assert _orders(db_path) == []


def test_without_a_bridge_the_subprocess_still_refuses_every_token(db_path):
    """The pre-#132 behaviour, deliberately unchanged when the flag is absent: a server
    started without `--token-bridge` validates against its own empty store, so a perfectly
    real token is unknown to it. The flag is additive, not a change to the default."""
    part = _real_part(db_path)
    store = ApprovalTokenStore()
    token = store.mint(part, 1, None, "supervisor-02")

    result = asyncio.run(
        execute_order_via_write_server(
            OrderRequest(part, 1, token.token, "tech-01"), db_path=db_path
        )
    )

    assert isinstance(result, OrderRejected)
    assert "not recognized" in result.reason
    assert _orders(db_path) == []


# --------------------------------------------------------------------------------------
# What the socket does and does not expose
# --------------------------------------------------------------------------------------


def test_the_bridge_exposes_consume_and_offers_no_way_to_mint():
    """Section 5's out-of-band property, across the boundary: the subprocess can spend an
    approval a human already gave and can create none. `TokenConsumer` is the whole surface,
    and `mint` is not on it."""
    assert hasattr(RemoteTokenStore, "consume")
    assert not hasattr(RemoteTokenStore, "mint")
    assert "mint" not in TokenConsumer.__protocol_attrs__
    assert "consume" in TokenConsumer.__protocol_attrs__


def test_the_socket_lives_in_a_private_directory_and_is_removed_afterwards():
    seen: dict = {}

    async def run():
        async with serve_token_store(ApprovalTokenStore()) as socket_path:
            seen["path"] = socket_path
            seen["dir_mode"] = stat.S_IMODE(os.stat(socket_path.parent).st_mode)
            seen["exists"] = socket_path.exists()

    asyncio.run(run())

    assert seen["exists"] is True
    assert seen["dir_mode"] == 0o700
    assert not seen["path"].exists()
    assert not seen["path"].parent.exists()


def test_no_token_value_is_ever_passed_on_the_command_line():
    """The socket path is a rendezvous address, not a credential -- which is the reason this
    mechanism was chosen over pre-seeding the subprocess with the minted token through argv
    or the environment, where it would be readable by other processes."""
    from src.agent.executor.client import write_server_params

    store = ApprovalTokenStore()
    token = store.mint("ZA-2115", 1, None, "supervisor-02")
    params = write_server_params(db_path=Path("/tmp/x.db"), token_bridge=Path("/tmp/s.sock"))

    rendered = " ".join(params.args)
    assert token.token not in rendered
    assert "--token-bridge" in rendered


# --------------------------------------------------------------------------------------
# The wire format, and failing closed
# --------------------------------------------------------------------------------------


def _consume_over_bridge(store: TokenConsumer, **kwargs) -> ApprovedOrder | TokenError:
    """Drive one `consume` through a real socket, from a thread that is not the event loop --
    which is exactly how the write server calls it (a blocking client, an async server)."""

    async def run():
        async with serve_token_store(store) as socket_path:
            remote = RemoteTokenStore(socket_path)
            return await asyncio.to_thread(remote.consume, **kwargs)

    return asyncio.run(run())


def test_a_successful_consume_round_trips_the_approved_order_exactly():
    store = ApprovalTokenStore()
    token = store.mint("ZA-2115", 3, "2nd_test-demo", "supervisor-02")

    outcome = _consume_over_bridge(
        store,
        token=token.token,
        part_number="ZA-2115",
        quantity=3,
        bearing_id="2nd_test-demo",
    )

    assert outcome == ApprovedOrder(
        part_number="ZA-2115",
        quantity=3,
        bearing_id="2nd_test-demo",
        approved_by="supervisor-02",
        approved_at=token.approved_at,
    )


@pytest.mark.parametrize("reason", [UNKNOWN, EXPIRED, ALREADY_USED, SCOPE_MISMATCH])
def test_every_rejection_reason_survives_the_round_trip(reason, db_path):
    """All four of #124's closed vocabulary, reconstructed on the far side as the same typed
    value -- not flattened into one generic failure."""
    now = datetime(2026, 8, 10, 9, 0, 0)
    clock = [now]
    store = ApprovalTokenStore(clock=lambda: clock[0])
    token = store.mint("ZA-2115", 1, None, "supervisor-02")
    kwargs = {
        "token": token.token,
        "part_number": "ZA-2115",
        "quantity": 1,
        "bearing_id": None,
    }

    if reason == UNKNOWN:
        kwargs["token"] = "never-minted"
    elif reason == EXPIRED:
        clock[0] = now + TOKEN_LIFETIME + timedelta(seconds=1)
    elif reason == ALREADY_USED:
        store.consume(token.token, "ZA-2115", 1, None)
    elif reason == SCOPE_MISMATCH:
        kwargs["quantity"] = 2

    outcome = _consume_over_bridge(store, **kwargs)

    assert outcome == TokenError(reason)


def test_an_unreachable_bridge_fails_closed(tmp_path):
    """A bridge that is not there refuses the order rather than raising into a tool body.
    The reason is `UNKNOWN` -- see `token_bridge.py`'s docstring on why that wording is
    imprecise and why it is left that way."""
    remote = RemoteTokenStore(tmp_path / "nothing-is-listening.sock", timeout=1.0)

    outcome = remote.consume("some-token", "ZA-2115", 1, None)

    assert outcome == TokenError(UNKNOWN)


def test_a_malformed_request_spends_nothing_and_is_refused():
    """A caller that speaks nonsense to the socket must not be able to move the store."""
    store = ApprovalTokenStore()
    token = store.mint("ZA-2115", 1, None, "supervisor-02")

    async def run():
        async with serve_token_store(store) as socket_path:
            def talk() -> bytes:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.settimeout(5.0)
                    sock.connect(str(socket_path))
                    sock.sendall(b"this is not json\n")
                    return sock.recv(4096)

            return await asyncio.to_thread(talk)

    reply = json.loads(asyncio.run(run()))

    assert reply["error"]["reason"] == UNKNOWN
    assert token.consumed is False


def test_token_store_for_bridge_returns_nothing_without_a_flag():
    """`None` in, `None` out, so `build_server` falls through to its own default store and a
    server launched the old way behaves exactly as Issue #110 built it."""
    assert token_store_for_bridge(None) is None
    assert isinstance(token_store_for_bridge("/tmp/some.sock"), RemoteTokenStore)


def test_the_real_store_satisfies_the_consumer_protocol():
    """`ApprovalTokenStore` was not modified for #132 and does not need to be: it already
    satisfies the protocol the write server now asks for."""
    assert isinstance(ApprovalTokenStore(), TokenConsumer)
    assert isinstance(RemoteTokenStore(Path("/tmp/x.sock")), TokenConsumer)
