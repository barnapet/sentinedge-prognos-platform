"""Agent C, the Executor -- the client half (Issue #126, `docs/agent_design.md` Sections 1,
5 and 10; the write server it connects to is Issue #110/#125's
`src/agent/mcp/write_server.py`, the token mechanism it relies on is Issue #124's
`src/agent/executor/approval.py`, neither modified here).

    order = OrderRequest(part_number="ZA-2115", quantity=1, approval_token=token,
                          requested_by="tech-01", bearing_id="2nd_test-demo")
    result = execute_order_sync(order)

Section 5's row for this agent: it holds exactly one tool and can only reach it with a
human-issued approval token; it sees the order parameters and the token, nothing else; it is
the one agent of the three that changes something. This module builds the client and the
structural guarantee that it can reach nothing but `place_order` -- mirroring
`answerer.py`'s `readonly_server_params()`/`readonly_tools()` on the write server instead of
the read-only one. It does not build the orchestrator that decides when to invoke this module,
mints a token, or assembles an `OrderRequest` from an approved recommendation -- Issue #126
takes that as a later issue's job and this module takes the fixed-schema record as an
already-given input.

**The open design question, decided here rather than silently: Agent C makes no LLM call.**
`execute_order` is a plain deterministic function that calls `place_order` directly over an
MCP session; nothing in this module imports `anthropic` or builds a prompt.

The case for a model call is real and is stated plainly rather than dismissed: it would match
Agent A and B's shape, and `tool_runner` would produce a trace-shaped record (Section 9) for
free. Section 1's own "Model and request configuration" text ("[the executor] runs on the same
model purely so the golden set measures one model, not two") reads as if this were already
decided -- but Issue #126 explicitly names it unresolved, and on inspection that sentence is a
configuration note for *if* C calls a model, not a decision that it does; nothing in Section 5
requires it. Weighed against the case for a call, three points favour skipping it, all
mechanical rather than stylistic:

1. **There is no decision left for a model to make.** By the time this module's entry point
   is called, a human has already approved an exact `(part_number, quantity, bearing_id)`, the
   token store binds `execute_order`'s one tool call to exactly that tuple (Section 5's "scoped
   to one order"), and the tool schema now offers exactly one tool. A model given this input
   has exactly one useful action available to it -- call `place_order` with the arguments it
   was handed, unchanged -- which is precisely what the deterministic path below does with no
   chance of drift. The critic's own design (Section 6, "why not deterministic-only") draws
   this line the other way when a genuine judgement remains (citation entailment); the mirror
   image applies here, where none does.
2. **A model call can only make the outcome worse, never better.** Its two possible
   deviations from "call the one tool with the given arguments" are refusing to call it, or
   calling it with different arguments -- both strictly worse than the deterministic path, and
   the second is exactly the shape of failure Section 10's executor-hardening exists to rule
   out structurally. Inserting a model between a validated record and its one permitted effect
   widens the surface for that failure for a decision that was never the model's to make.
3. **Cost, latency and non-determinism are being spent on nothing.** The same three
   objections Section 6 raises against an LLM-only critic apply again, sharpened: unlike the
   critic's entailment question, there is not even a closed judgement call here for a model to
   answer, so the whole cost is overhead.

The consistency and free-trace arguments are the honest counterweight, and both are weaker on
inspection than they look. Consistency for its own sake is not this codebase's practice --
Section 1 already sets the critic's `effort` to `low` against the answerer's `high` because the
*job* differs, not because uniformity was a goal to defend. And the "trace entry for free" is
not, in fact, free today: Section 9's trace panel is unbuilt regardless of this decision, so
nothing is gained by this module now, only a shape that would need to be produced by hand
later either way -- which this module leaves to whichever issue builds tracing, the same
deferral Issue #126 already makes for the orchestrator.

**A `requested_by` field that neither the issue's fixed-schema example nor Section 10's
illustrative JSON lists, flagged rather than silently added.** Both show
`{"part_number", "quantity", "bearing_id", "approval_token"}`, and Section 10 additionally
states that `requested_by` is "supplied by the harness from the token record" alongside
`approved_by`/`approved_at`. But `place_order`'s actual, already-merged argument surface
(`write_server.py`, Issue #125) requires `requested_by` as an ordinary caller-supplied
argument -- Issue #125's own docstring says so explicitly ("`requested_by` is unaffected by
any of this and stays an ordinary caller-supplied argument") -- and `ApprovalToken`/
`ApprovedOrder` (`approval.py`, Issue #124, unmodified here) carry no `requested_by` field at
all for the token to "supply it from". Without it, this module could not construct a
call `place_order` will accept at all. `OrderRequest` below therefore carries `requested_by`
as a fifth field, sourced by the harness the same way the other four are (a later issue's
job, not this module's), rather than reproducing a four-field record that cannot actually
place an order against the tool as it exists today.

**Why `execute_order` takes a session rather than opening its own.** The structural
least-privilege claims (this client's tool surface, and that it cannot reach a read-only
tool) are tested the way Section 10 case 6 and `test_agent_mcp_servers.py` already test the
answerer and the servers: a real subprocess over real stdio, via `write_tools()` below. But a
subprocess launched by `python -m src.agent.mcp.write_server` builds its own fresh,
in-process `ApprovalTokenStore` in `main()` (Issue #125's own tests hit this same wall, and
say so in `test_agent_mcp_servers.py`'s module docstring) -- nothing a test process mints is
ever visible to it, and wiring a token across that boundary is exactly the orchestrator
question Issue #126 defers. So the deterministic call logic is factored out as `execute_order
(order, session)`, taking anything with an MCP-shaped `call_tool(name, arguments)` -- a real
`ClientSession` from `write_tools()`, or `write_server.build_server()`'s in-process
`MCPServer`, the same object `test_agent_mcp_servers.py` already calls directly to get a
*shared*, real `ApprovalTokenStore` for its own behavioural tests. Structural claims use the
former; a real approved order succeeding, and each of the four token-rejection reasons
(Issue #125), use the latter. `execute_order_via_write_server` is the convenience entry point
that does both steps for a real caller who does not need to inject a session.
"""
from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Protocol

from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.types import CallToolResult

from src.agent.mcp.results import payload_of
from src.agent.mcp.write_server import WRITE_TOOL_NAMES

REPO_ROOT = Path(__file__).resolve().parents[3]


# --- The fixed-schema input (Task 4) ---------------------------------------------------


@dataclass(frozen=True)
class OrderRequest:
    """The only input this module's entry points accept. Every field is a scalar the harness
    already validated after the human approval gate -- see the module docstring on why
    `requested_by` is a fifth field rather than the issue's illustrative four.

    No constructor path here takes a `Draft`, a question, or any other free-text value: every
    field is `str` or `int`, and there is no field for prose of any kind.
    """

    part_number: str
    quantity: int
    approval_token: str
    requested_by: str
    bearing_id: str | None = None

    def as_tool_arguments(self) -> dict[str, Any]:
        """This record, shaped exactly as `place_order`'s argument surface expects it."""
        return {
            "part_number": self.part_number,
            "quantity": self.quantity,
            "requested_by": self.requested_by,
            "approval_token": self.approval_token,
            "bearing_id": self.bearing_id,
        }


# --- The typed result (Task 5) ----------------------------------------------------------
#
# The same convention `approval.py`'s `ApprovedOrder`/`TokenError` and `results.py`'s
# `ok`/`failed` already use: a rejection is a value to branch on, not an exception to catch.


@dataclass(frozen=True)
class OrderPlaced:
    """`place_order` succeeded. Fields are read back from the tool's own response payload,
    not echoed from the request, so a mismatch between what was asked and what was recorded
    would show up as a failing assertion rather than being masked by returning the input."""

    order_id: int
    part_number: str
    quantity: int
    bearing_id: str | None
    requested_by: str
    approved_by: str
    approved_at: str


@dataclass(frozen=True)
class OrderRejected:
    """`place_order` refused the order. `reason` is the tool's own plain-language message,
    unchanged -- an unknown/expired/already-used/scope-mismatched token (Issue #125), an
    oversell, an unknown part, or a non-positive quantity all arrive here the same way."""

    reason: str


ExecutionResult = OrderPlaced | OrderRejected


# --- Tool wiring (Task 2, Task 3) -------------------------------------------------------


class ToolCallSession(Protocol):
    """What `execute_order` needs from its session: a real `mcp.ClientSession` (over a real
    subprocess, via `write_tools()`) and `write_server.build_server()`'s in-process
    `MCPServer` both satisfy this -- see the module docstring for why both are used."""

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult: ...


def write_server_params(db_path: Path | None = None) -> StdioServerParameters:
    """How to launch the write-capable MCP server (#110/#125) as this agent's only tool
    source. The write-server analogue of `answerer.py`'s `readonly_server_params()`.

    **This function is the whole of Section 5's least-privilege claim for Agent C, the other
    direction.** It names one server module, and that module registers one tool,
    `place_order`; the four read-only tools live on `src.agent.mcp.readonly_server`, in a
    different process, over a transport this agent never opens.
    """
    args = ["-m", "src.agent.mcp.write_server"]
    if db_path is not None:
        args += ["--db-path", str(db_path)]
    return StdioServerParameters(
        command=sys.executable,
        args=args,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )


def _assert_write_only_surface(names: Iterable[str]) -> None:
    """Raise if `names` (a connected server's actual tool list) carries anything outside
    `WRITE_TOOL_NAMES`. Factored out of `write_tools()` so it is testable without spawning a
    real server that misbehaves -- `write_server.py` is not modified here, and there is no
    way to make the real one offer a second tool to test against."""
    unexpected = sorted(set(names) - set(WRITE_TOOL_NAMES))
    if unexpected:
        raise RuntimeError(
            f"the executor's server offered tools outside its write-only set: {unexpected}"
        )


@asynccontextmanager
async def write_tools(db_path: Path | None = None) -> AsyncIterator[ClientSession]:
    """Open a session to the write server and yield it, ready for `execute_order`.

    Raises `RuntimeError` if the connected server offers anything outside
    `WRITE_TOOL_NAMES` -- same "fails loudly at wiring time" posture as `answerer.py`'s
    `readonly_tools()`. Unlike that function, this yields the bare session rather than a list
    of `tool_runner`-shaped tools: there is no model loop here for tools to be wrapped for
    (see the module docstring's LLM-call decision).
    """
    async with stdio_client(write_server_params(db_path)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = (await session.list_tools()).tools
            _assert_write_only_surface(t.name for t in listed)
            yield session


# --- The call itself (Task 5) ------------------------------------------------------------


async def execute_order(order: OrderRequest, session: ToolCallSession) -> ExecutionResult:
    """Call `place_order` with `order`'s fields, unchanged, and nothing else. This is the
    entirety of Agent C's decision-making, which is to say there is none: see the module
    docstring for why that is the point rather than a gap.
    """
    result = await session.call_tool("place_order", order.as_tool_arguments())
    payload = payload_of(result)

    if result.is_error:
        return OrderRejected(reason=payload.get("error", "the order was refused"))

    data = payload["data"]
    return OrderPlaced(
        order_id=data["order_id"],
        part_number=data["part_number"],
        quantity=data["quantity"],
        bearing_id=data["bearing_id"],
        requested_by=data["requested_by"],
        approved_by=data["approved_by"],
        approved_at=data["approved_at"],
    )


async def execute_order_via_write_server(
    order: OrderRequest, *, db_path: Path | None = None
) -> ExecutionResult:
    """The real entry point: open a fresh session to the actual write server and place one
    order. A fresh session per call, matching `write_server.py`'s single-tool-call-per-
    connection shape in `test_agent_mcp_servers.py`'s own subprocess tests.
    """
    async with write_tools(db_path) as session:
        return await execute_order(order, session)


def execute_order_sync(order: OrderRequest, **kwargs: Any) -> ExecutionResult:
    """Synchronous entry point. The MCP stdio transport is async-native, so the async path is
    the real one and this is the thin wrapper around it -- the same shape `answerer.py`'s
    `answer()` takes around `answer_async()`."""
    return asyncio.run(execute_order_via_write_server(order, **kwargs))
