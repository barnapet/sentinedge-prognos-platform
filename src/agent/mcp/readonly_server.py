"""The read-only stdio MCP server (Issue #110, `docs/agent_design.md` Section 2).

    python -m src.agent.mcp.readonly_server

Four tools, all of them read-only: `get_bearing_status`, `predict_health_state`,
`check_inventory`, `search_documentation`. This is the server the answerer (Section 5's
agent A) connects to, and `place_order` is **not** on it -- not filtered out of it, not
disabled on it: it is registered on a different server, in a different process, reachable
only over a transport this one does not hold. Section 10 case 6 asserts that on the client
configuration rather than on whether the model happened not to try.

Everything a tool needs is bound at registration time, so the schema the model sees carries
only the model's own arguments -- no base URLs, no database paths, no injectable search
function. An argument that is not in the schema is an argument no prompt can supply.

`build_server()` is a factory rather than a module-level singleton for the same reason
`src/serving/api.py`'s `create_app()` is: the tests need instances pointed at their own
temporary database and their own stub retrieval, without reaching into module internals.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mcp.server import MCPServer
from mcp.types import CallToolResult

from src.agent.inventory.build_db import DB_PATH, build_db
from src.agent.mcp import tools
from src.agent.mcp.budget import ToolCallBudget
from src.agent.mcp.serving_client import (
    DEFAULT_BASE_URL,
    DRIFT_ENDPOINT,
    PREDICT_ENDPOINT,
)
from src.agent.mcp.tools import DOCS_SOURCE_ID, INVENTORY_SOURCE_ID, SearchFn
from src.agent.rag.retrieval import DEFAULT_LIMIT

SERVER_NAME = "prognos-readonly"

READONLY_TOOL_NAMES = (
    "get_bearing_status",
    "predict_health_state",
    "check_inventory",
    "search_documentation",
    "find_similar_historical_pattern",
)

# Section 2's fifth read-only tool is registered as of Issue #140: Section 12's
# `models/trajectory_archive.parquet` now exists and is committed, so the tool answers from
# real reference data rather than the invented kind Issue #110 declined to stub.


def build_server(
    base_url: str = DEFAULT_BASE_URL,
    db_path: Path = DB_PATH,
    search: SearchFn | None = None,
    budget: ToolCallBudget | None = None,
) -> tuple[MCPServer, ToolCallBudget]:
    """Build one read-only server and the budget its tools are charged against.

    The budget is returned alongside rather than hidden inside, so a harness can inspect
    or reset it in-process. It is deliberately not reachable through any tool -- see
    `budget.py` for why a model that can reset its own cap does not have one.
    """
    budget = budget if budget is not None else ToolCallBudget()
    server = MCPServer(SERVER_NAME, instructions=__doc__)

    @server.tool()
    def get_bearing_status(bearing_id: str | None = None) -> CallToolResult:
        """Current monitoring state for a bearing the serving process is tracking: how many
        windows it has seen, whether its baseline is still warming up, whether any feature
        is drifting, its latest rms_ratio, and the predicted-class counts so far. Omit
        bearing_id to list every tracked bearing. An unknown bearing_id returns a
        not-found, never an invented state."""
        refusal = budget.guard("live_endpoint", DRIFT_ENDPOINT)
        return refusal or tools.get_bearing_status(bearing_id, base_url=base_url)

    @server.tool()
    def predict_health_state(bearing_id: str, signal: list[float]) -> CallToolResult:
        """Score one raw vibration window for one bearing and return its predicted health
        state. Requires the complete raw signal for a single snapshot (about 20,480
        samples) -- if you do not have one, use get_bearing_status instead, which reads
        state the running system already produced. Never fabricate or reuse a signal."""
        refusal = budget.guard("live_endpoint", PREDICT_ENDPOINT)
        return refusal or tools.predict_health_state(bearing_id, signal, base_url=base_url)

    @server.tool()
    def check_inventory(
        part_number: str | None = None, bearing_type: str | None = None
    ) -> CallToolResult:
        """Look up spare parts: description, quantity on hand, unit price, lead time and
        stock location. Filter by exact part_number, by bearing_type, by both, or by
        neither for the whole catalogue. Read-only -- this reports stock, it never orders
        anything."""
        refusal = budget.guard("inventory", INVENTORY_SOURCE_ID)
        return refusal or tools.check_inventory(part_number, bearing_type, db_path=db_path)

    @server.tool()
    def search_documentation(
        query: str, limit: int = DEFAULT_LIMIT, source_type: str | None = None
    ) -> CallToolResult:
        """Search this project's own documentation and its cited public references for
        passages relevant to a question. Returns verbatim chunks, each with the stable
        chunk id to cite it by and the file or URL it came from. Optionally restrict to
        'decision_doc' (this repository's docs) or 'public_reference'."""
        refusal = budget.guard("live_endpoint", DOCS_SOURCE_ID)
        return refusal or tools.search_documentation(query, limit, source_type, search=search)

    @server.tool()
    def find_similar_historical_pattern(
        bearing_id: str, window: int = tools.QUERY_WINDOW
    ) -> CallToolResult:
        """Compare a bearing's recent trajectory against three archived bearing failures
        from the NASA IMS lab rig, and report which it most resembles -- or that it
        resembles none of them closely enough to say. Uses the shape of the last `window`
        readings, not their absolute level. There are only three references, all from one
        rig at one operating condition, so the result is a ranking among those three, not
        evidence about bearings in general; report it that way."""
        refusal = budget.guard("trajectory_match", tools.trajectory_source_id())
        return refusal or tools.find_similar_historical_pattern(
            bearing_id, window, base_url=base_url
        )

    return server, budget


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--url",
        default=DEFAULT_BASE_URL,
        help="base URL of the already-running serving API (never started by this process)",
    )
    parser.add_argument("--db-path", type=Path, default=DB_PATH, help="inventory database path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # Idempotent, and a no-op when the database already exists (Issue #101): a fresh clone
    # gets a seeded database rather than a tool that reports the inventory as unavailable.
    build_db(args.db_path)
    server, _ = build_server(base_url=args.url, db_path=args.db_path)
    server.run("stdio")


if __name__ == "__main__":
    main()
