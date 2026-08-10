"""The write-capable stdio MCP server (Issue #110, `docs/agent_design.md` Section 2; the
`place_order` argument surface fixed by Issue #125).

    python -m src.agent.mcp.write_server

One tool: `place_order`. Nothing else is registered here, and none of the read-only tools
are -- Section 5's agent C "holds exactly one tool and can only reach it with a human-issued
approval token", and the first half of that sentence is this module.

**What Issue #110 built and what it left for #125.** It built the tool: the transport, the
argument surface, the transactional write through `src/agent/inventory/orders.py`, the
result envelope -- but it did not build the approval gate itself, and its own docstring said
so: "the mechanism above closes this by construction -- the harness fills both from the
token record -- and that is a **requirement** on the tool schema" (Section 10). Until #125,
`approved_by`/`approved_at` were ordinary model-fillable arguments, and the guarantee that
survived was only the one the schema owns (`NOT NULL` on both columns).

**What #125 changes.** `approved_by`/`approved_at` are gone from this tool's argument
surface entirely, replaced by `approval_token: str`. Validating it and deriving those two
fields from the result lives in `src/agent/mcp/tools.py::place_order`, against a
`token_store` (`src/agent/executor/approval.py`, Issue #124, not modified here) -- this
module's job is only to hold that store and thread it through, the same shape it already
uses for `budget`. Minting tokens (i.e. who calls `token_store.mint(...)`, and when) is the
orchestrator's job, a later issue; this module never mints, only consumes.

Separating this into its own process is what makes Section 2's least-privilege claim
structural. A prompt injection inside a retrieved chunk or an inventory `description` field
cannot call a tool whose transport the reading process does not hold, however convincing
the injected text is.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mcp.server import MCPServer
from mcp.types import CallToolResult

from src.agent.executor.approval import ApprovalTokenStore
from src.agent.inventory.build_db import DB_PATH, build_db
from src.agent.mcp import tools
from src.agent.mcp.budget import ToolCallBudget
from src.agent.mcp.tools import INVENTORY_SOURCE_ID

SERVER_NAME = "prognos-write"

WRITE_TOOL_NAMES = ("place_order",)


def build_server(
    db_path: Path = DB_PATH,
    budget: ToolCallBudget | None = None,
    token_store: ApprovalTokenStore | None = None,
) -> tuple[MCPServer, ToolCallBudget, ApprovalTokenStore]:
    """Build one write-capable server, the budget its tool is charged against, and the
    approval-token store its tool validates against.

    The cap applies here too. It is a smaller lever on a one-tool server than on the
    answerer's four, but Issue #110 asks for it at the server level so that it holds
    "regardless of what calls these servers later" -- and a loop that repeatedly retries a
    rejected order is exactly the shape it is meant to stop.

    `token_store` follows the same injectable-with-a-default shape `budget` already has:
    pass the same `ApprovalTokenStore` instance a later issue's orchestrator mints tokens
    into, or omit it to get a fresh, empty store (every token unknown, useful standalone and
    in tests that mint their own tokens against the returned store).
    """
    budget = budget if budget is not None else ToolCallBudget()
    token_store = token_store if token_store is not None else ApprovalTokenStore()
    server = MCPServer(SERVER_NAME, instructions=__doc__)

    @server.tool()
    def place_order(
        part_number: str,
        quantity: int,
        requested_by: str,
        approval_token: str,
        bearing_id: str | None = None,
    ) -> CallToolResult:
        """Place a spare-part order: decrements stock and records the order, in one
        transaction that aborts rather than overselling. approval_token is minted
        out-of-band by a human approval (Section 5) and is scoped to this exact part,
        quantity, and bearing, single-use, and short-lived; approved_by and approved_at are
        derived from the validated token, not supplied here. requested_by is unaffected and
        stays a plain argument. Returns the new order id, or an explanation of why the
        order was rejected."""
        refusal = budget.guard("inventory", INVENTORY_SOURCE_ID)
        return refusal or tools.place_order(
            part_number,
            quantity,
            requested_by,
            approval_token,
            bearing_id,
            token_store=token_store,
            db_path=db_path,
        )

    return server, budget, token_store


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db-path", type=Path, default=DB_PATH, help="inventory database path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    build_db(args.db_path)
    server, _, _ = build_server(db_path=args.db_path)
    server.run("stdio")


if __name__ == "__main__":
    main()
