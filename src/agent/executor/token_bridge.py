"""The approval-token bridge: one process's `ApprovalTokenStore`, reachable from another
(Issue #132, `docs/agent_design.md` Sections 0, 2 and 5).

    async with serve_token_store(store) as socket_path:
        # a subprocess launched with --token-bridge <socket_path> validates against `store`

This closes the gap PR #131 (#127) flagged rather than worked around. `write_server.py`'s
`main()` built its own fresh, empty `ApprovalTokenStore`, so a token minted in the
orchestrator could never be consumed by a `python -m src.agent.mcp.write_server` subprocess --
and the orchestrator's only working option was to call `build_server()`'s in-process server
directly, giving up the stdio process boundary Section 2 describes at the one place a real
order is actually placed.

**The mechanism: the store stays put and the *validation call* travels.** The orchestrator
keeps the one authoritative `ApprovalTokenStore` -- in-process, no durable storage, dying with
the process, exactly as Issue #124 built it. `serve_token_store` exposes only `consume` over a
Unix domain socket in a `0700` directory, for the lifetime of a single order.
`RemoteTokenStore` is what the subprocess holds: it satisfies the one method
`src/agent/mcp/tools.py::place_order` actually calls, forwards the four scope scalars, and
reconstructs `approval.py`'s own `ApprovedOrder`/`TokenError` from the reply. Nothing in
`approval.py` changes -- not `mint`, not `consume`, not the four rejection reasons.

**Why this shape and not the obvious alternatives.** Issue #132 named three; investigation
turned up two more. Each is recorded with the reason it lost, in the spirit Section 1 and
#126 already set for this repo's design notes.

- **A persisted, file- or SQLite-backed token store** -- much the simplest to wire, and
  rejected on two independent grounds. It directly reverses Issue #124's "in-process, no
  durable storage" decision and, through it, Section 0's constraint that per-run state does
  not outlive the process; #132 requires such a reversal to be deliberate rather than
  incidental, and nothing here needs it. More concretely, **a token that survives a restart
  is the wrong behaviour for the credential this is** -- #124's own docstring makes that
  argument ("a token that does not survive a restart is the correct behaviour for a
  credential that is supposed to expire in a few minutes anyway"). It would also move
  single-use enforcement into a file, where preventing a double-spend means getting
  cross-process locking right; the in-memory store gets that property for free.
- **Pre-seeding the subprocess's store with the one already-minted token** (passed by argv,
  environment, or a `0600` temp file). Cheaper than a socket, and rejected because it
  **splits the single-use invariant across two copies of the record**. `approval.py` is
  explicit that the store is the sole owner of the consumed transition -- "a caller cannot
  mark its own token used by holding a reference to it" -- and a copy in a subprocess is
  exactly such a reference: it would flip `consumed` on the copy while the orchestrator's
  authoritative record stayed unspent. It also puts a live credential into argv or the
  environment, both readable elsewhere on the machine, for no gain over a socket path, which
  is a rendezvous address and not a secret.
- **An MCP-native server-to-client callback.** `mcp.server.context.Context` does expose
  `can_send_request`/`send_raw_request`, so the write server could in principle ask its
  client to validate over the *existing* stdio transport, with no second channel at all --
  genuinely attractive, and the closest thing to a "free" bridge. Rejected on blast radius:
  it means inventing a bespoke JSON-RPC method riding MCP, teaching `ClientSession` to answer
  an inbound custom request, and making `tools.py::place_order` async and context-aware --
  which ripples into every existing synchronous test of that function. A Unix socket reaches
  the same place while leaving `tools.py` untouched entirely.
- **Adding a `mint` tool to the write server** so the subprocess owns the store. A
  non-starter, recorded because it is the first idea that occurs: it would put mint authority
  on an MCP tool surface, destroying Section 5's out-of-band property -- the whole point is
  that the token is "never present in any model's context until the human has approved".
- **Concluding the in-process call was right after all** and closing #132 with a docstring
  edit. This was a real candidate and it is worth saying why it lost, because on threat model
  alone it very nearly wins. The documented attack -- prompt injection reaching `place_order`
  -- is blocked by the tool list handed to `tool_runner`, not by the process boundary, and it
  stays blocked either way; and an attacker who achieves code execution inside the
  orchestrator holds mint authority regardless, so a subprocess buys nothing against them.
  What decided it is this repository's own **"two independent layers, verified rather than
  documented"** pattern -- `src/serving/single_worker.py` (#84) refuses a second worker twice
  over, and Section 5's approval gate is enforced both by token validation and by
  `orders.approved_by`'s `NOT NULL`. Without this bridge, the separation between the reading
  agent and the one side-effecting tool rests on a single layer: the list of tools passed to
  `tool_runner`. That layer is correct today and is tested, but it is a data structure in the
  same process, and a design intent held by one layer is the kind this project has twice
  decided not to rely on. Restoring the boundary also stops #126's `write_tools()` being a
  path only tests take: before this issue, the one production caller of the executor bypassed
  it.

**What this does *not* claim.** It does not make the executor safe against a compromised
orchestrator, and nothing could: the orchestrator legitimately mints. It does not add a
defence against prompt injection that was not already there. What it adds is OS-level memory
isolation between the harness and the only code in this system that writes to the inventory
database, and it makes the architecture Section 2 describes the one the production path
actually uses.

**Exposure of the socket itself, stated rather than left implicit.** The socket accepts
`consume` calls from any process running as the same user for as long as it is open. That is
not a widening of real exposure -- a same-user process can already read the orchestrator's
memory, where both the token store and the API key live -- and it is bounded on both ends
anyway: the directory is `0700`, and `serve_token_store` is a context manager scoped to a
single order rather than a daemon, so the socket exists for the milliseconds an order takes
and is unlinked afterwards.

**Failing closed, and the one honest wart.** A bridge that cannot be reached returns
`TokenError(UNKNOWN)` -- the order is refused and nothing is written. `UNKNOWN`'s message
("the approval token is not recognized") is not literally what happened, and the right reason
would be a fifth one meaning "the validator was unreachable". Issue #132 forbids changing
`approval.py`'s typed rejection reasons, and a closed four-value vocabulary is worth more than
a precise message here, so the mismatch is left in place and named rather than smuggled past:
the security-relevant behaviour -- refuse, write nothing -- is right, and only the wording is
imprecise. Flagged in the PR.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from src.agent.executor.approval import (
    UNKNOWN,
    ApprovalTokenStore,
    ApprovedOrder,
    TokenError,
)

# The socket is opened per order and closed with it, so a generous timeout still cannot hang
# a run for long. It is here so a bridge that dies mid-order fails closed rather than blocking
# the write server forever on a read that will never return.
DEFAULT_TIMEOUT_SECONDS = 10.0

SOCKET_NAME = "approval.sock"


@runtime_checkable
class TokenConsumer(Protocol):
    """The one method `src/agent/mcp/tools.py::place_order` actually calls on a token store.

    `runtime_checkable` so substitutability is assertable rather than assumed -- it checks
    for the method's presence, not its signature, which is enough to state the thing worth
    stating: `ApprovalTokenStore` satisfies this without having been modified for #132.

    `ApprovalTokenStore` (#124) satisfies it without modification, and so does
    `RemoteTokenStore` below -- which is what lets the write server hold either one without
    knowing which it has. Deliberately narrower than `ApprovalTokenStore`: **`mint` is not on
    it**, so nothing reachable from the write server's process can create an approval, only
    spend one (Section 5).
    """

    def consume(
        self, token: str, part_number: str, quantity: int, bearing_id: str | None
    ) -> ApprovedOrder | TokenError: ...


# --- The wire format --------------------------------------------------------------------
#
# One line of JSON each way. A line-delimited format rather than a length prefix because the
# payload is five scalars and a reply, and `readline` is the whole parser.


def _encode_request(
    token: str, part_number: str, quantity: int, bearing_id: str | None
) -> bytes:
    return (
        json.dumps(
            {
                "token": token,
                "part_number": part_number,
                "quantity": quantity,
                "bearing_id": bearing_id,
            }
        )
        + "\n"
    ).encode("utf-8")


def _encode_response(outcome: ApprovedOrder | TokenError) -> bytes:
    if isinstance(outcome, TokenError):
        payload: dict[str, Any] = {"error": {"reason": outcome.reason}}
    else:
        payload = {
            "approved": {
                "part_number": outcome.part_number,
                "quantity": outcome.quantity,
                "bearing_id": outcome.bearing_id,
                "approved_by": outcome.approved_by,
                "approved_at": outcome.approved_at.isoformat(),
            }
        }
    return (json.dumps(payload) + "\n").encode("utf-8")


def _decode_response(line: bytes) -> ApprovedOrder | TokenError:
    """Rebuild `approval.py`'s own types from a reply.

    Anything unparseable is a refusal, not an exception: this runs inside a tool body, where
    a raised error would become a stack-trace-shaped string instead of the plain-language
    result `src/agent/mcp/results.py` requires.
    """
    try:
        payload = json.loads(line)
        if "approved" in payload:
            approved = payload["approved"]
            return ApprovedOrder(
                part_number=approved["part_number"],
                quantity=approved["quantity"],
                bearing_id=approved["bearing_id"],
                approved_by=approved["approved_by"],
                approved_at=datetime.fromisoformat(approved["approved_at"]),
            )
        return TokenError(payload["error"]["reason"])
    except (TypeError, ValueError, KeyError):
        return TokenError(UNKNOWN)


# --- The subprocess's half ----------------------------------------------------------------


@dataclass(frozen=True)
class RemoteTokenStore:
    """A `TokenConsumer` that forwards `consume` to an `ApprovalTokenStore` in another
    process. This is what the write server holds when it is launched with `--token-bridge`.

    Blocking sockets on purpose: `tools.place_order` is a synchronous function called from a
    synchronous tool body, so an async client here would need an event loop that does not
    exist at that point. The orchestrator on the other end *is* async and services the socket
    from the same loop it awaits the order on -- it is idle there, so there is no deadlock.
    """

    socket_path: Path
    timeout: float = DEFAULT_TIMEOUT_SECONDS

    def consume(
        self, token: str, part_number: str, quantity: int, bearing_id: str | None
    ) -> ApprovedOrder | TokenError:
        """Ask the authoritative store to validate and spend one token.

        Never raises, and fails closed: an unreachable or unreadable bridge is
        `TokenError(UNKNOWN)`, so the order is refused and nothing is written. See the module
        docstring on why `UNKNOWN` rather than a fifth, more accurate reason.
        """
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.timeout)
                sock.connect(str(self.socket_path))
                sock.sendall(_encode_request(token, part_number, quantity, bearing_id))
                return _decode_response(_recv_line(sock))
        except OSError:
            return TokenError(UNKNOWN)


def _recv_line(sock: socket.socket) -> bytes:
    """Read until the first newline. Returns what it has if the peer closes early, which
    `_decode_response` then reports as a refusal."""
    chunks: list[bytes] = []
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
        if b"\n" in chunk:
            break
    return b"".join(chunks)


# --- The orchestrator's half ---------------------------------------------------------------


async def _handle(store: TokenConsumer, reader: asyncio.StreamReader,
                  writer: asyncio.StreamWriter) -> None:
    try:
        line = await reader.readline()
        if not line:
            return
        try:
            request = json.loads(line)
            outcome = store.consume(
                request["token"],
                request["part_number"],
                request["quantity"],
                request["bearing_id"],
            )
        except (TypeError, ValueError, KeyError):
            # A malformed request spends nothing and is refused, rather than raising inside
            # a connection handler where the exception would be invisible to the caller.
            outcome = TokenError(UNKNOWN)
        writer.write(_encode_response(outcome))
        await writer.drain()
    finally:
        writer.close()


@asynccontextmanager
async def serve_token_store(
    store: TokenConsumer, *, directory: Path | None = None
) -> AsyncIterator[Path]:
    """Expose `store.consume` on a Unix socket for the duration of the block, and yield the
    socket's path to hand to a subprocess as `--token-bridge`.

    Only `consume` is reachable. `mint` is not exposed by any request this speaks, which is
    what keeps Section 5's "minted out-of-band" property true across the boundary: the
    subprocess can spend an approval a human already gave and can create none.

    The socket lives in a `0700` directory that is removed on exit, so nothing is left behind
    between orders.
    """
    owned = directory is None
    base = Path(tempfile.mkdtemp(prefix="prognos-approval-")) if owned else Path(directory)
    socket_path = base / SOCKET_NAME

    server = await asyncio.start_unix_server(
        lambda reader, writer: _handle(store, reader, writer), path=str(socket_path)
    )
    try:
        os.chmod(socket_path, 0o600)
        async with server:
            yield socket_path
    finally:
        socket_path.unlink(missing_ok=True)
        if owned:
            shutil.rmtree(base, ignore_errors=True)


def token_store_for_bridge(socket_path: str | Path | None) -> TokenConsumer | None:
    """What `write_server.py`'s `main()` should hold, given its `--token-bridge` argument.

    `None` in, `None` out -- so a server started without a bridge keeps exactly the behaviour
    Issue #110 gave it (its own fresh, empty store, every token unknown), and the flag is
    additive rather than a change to the default path.
    """
    if socket_path is None:
        return None
    return RemoteTokenStore(Path(socket_path))


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "ApprovalTokenStore",
    "RemoteTokenStore",
    "TokenConsumer",
    "serve_token_store",
    "token_store_for_bridge",
]
