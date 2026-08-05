"""The write-capable stdio MCP server (Issue #110, `docs/agent_design.md` Section 2).

    python -m src.agent.mcp.write_server

One tool: `place_order`. Nothing else is registered here, and none of the read-only tools
are -- Section 5's agent C "holds exactly one tool and can only reach it with a human-issued
approval token", and the first half of that sentence is this module.

**What this issue builds and what it does not.** It builds the tool: the transport, the
argument surface, the transactional write through `src/agent/inventory/orders.py`, the
result envelope. It does **not** build the approval gate -- Section 5's `approval_token`
(minted out-of-band, scoped to one `(part_number, quantity, bearing_id)`, single-use,
short-lived) and the decision about which agent connects to this server are excluded from
Issue #110 and belong to the later issue that builds the agent loop. Until then, the
guarantee that survives is the one the schema owns: `orders.approved_by` and
`orders.approved_at` are `NOT NULL`, so a row without a recorded approval cannot exist,
whatever calls this.

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

from src.agent.inventory.build_db import DB_PATH, build_db
from src.agent.mcp import tools
from src.agent.mcp.budget import ToolCallBudget
from src.agent.mcp.tools import INVENTORY_SOURCE_ID

SERVER_NAME = "prognos-write"

WRITE_TOOL_NAMES = ("place_order",)


def build_server(
    db_path: Path = DB_PATH, budget: ToolCallBudget | None = None
) -> tuple[MCPServer, ToolCallBudget]:
    """Build one write-capable server and the budget its tool is charged against.

    The cap applies here too. It is a smaller lever on a one-tool server than on the
    answerer's four, but Issue #110 asks for it at the server level so that it holds
    "regardless of what calls these servers later" -- and a loop that repeatedly retries a
    rejected order is exactly the shape it is meant to stop.
    """
    budget = budget if budget is not None else ToolCallBudget()
    server = MCPServer(SERVER_NAME, instructions=__doc__)

    @server.tool()
    def place_order(
        part_number: str,
        quantity: int,
        requested_by: str,
        approved_by: str,
        approved_at: str,
        bearing_id: str | None = None,
    ) -> CallToolResult:
        """Place a spare-part order: decrements stock and records the order, in one
        transaction that aborts rather than overselling. approved_by and approved_at record
        who authorised this order and when; both are required and are stored on the order
        row. Returns the new order id, or an explanation of why the order was rejected."""
        refusal = budget.guard("inventory", INVENTORY_SOURCE_ID)
        return refusal or tools.place_order(
            part_number,
            quantity,
            requested_by,
            approved_by,
            approved_at,
            bearing_id,
            db_path=db_path,
        )

    return server, budget


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db-path", type=Path, default=DB_PATH, help="inventory database path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    build_db(args.db_path)
    server, _ = build_server(db_path=args.db_path)
    server.run("stdio")


if __name__ == "__main__":
    main()
