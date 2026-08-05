"""Tool result shape (Issue #110, `docs/agent_design.md` Section 2).

Every tool result is a single JSON text block in one of exactly two shapes:

    {"source": {"source_type": ..., "source_id": ..., "retrieved_at": ...}, "data": {...}}
    {"source": {...}, "error": "<plain-language message>"}          # with is_error = True

Two properties of that are load-bearing, and both come straight from Section 2:

**The tool layer mints the `source` block; the model never does.** `retrieved_at` is
stamped here, `source_id` is a constant owned by the tool, and `source_type` is validated
against Section 2's closed vocabulary. Section 6's citation-existence check is a set
membership test against exactly these ids -- which only works if the ids come from the
harness rather than from the model's own text.

**Failures are returned, not raised.** A raised exception would break the
`tool_use`/`tool_result` pairing and give the model nothing to degrade from; Section 6's
tier-3 degraded answer is only reachable if the failure arrives as a readable tool result.
So every tool body is wrapped such that its failure mode is an `error` envelope with
`is_error = True` and a message written for a reader, not a stack trace.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from mcp.types import CallToolResult, TextContent

# `docs/agent_design.md` Section 2's vocabulary, verbatim and closed. `trajectory_match`
# is listed because Section 2 lists it; nothing mints it yet (see the registry note in
# `readonly_server.py` about `find_similar_historical_pattern`).
SOURCE_TYPES = frozenset(
    {"live_endpoint", "decision_doc", "public_reference", "inventory", "trajectory_match"}
)

# The plain-language failure messages, as constants rather than inline strings: Issue #110
# names the first one literally ("the prediction service is not reachable"), and Section
# 8's tier-3 tests assert on the text a degraded answer is produced from, so it needs to
# be one definition rather than several near-copies.
SERVICE_UNREACHABLE = "the prediction service is not reachable"
DOCS_INDEX_UNREACHABLE = "the documentation index is not reachable"
INVENTORY_UNAVAILABLE = "the inventory database is not available"
ORDER_FAILED = "the order could not be placed"


def source_block(source_type: str, source_id: str) -> dict[str, str]:
    """Mint one `source` block, stamping `retrieved_at` now.

    Raises `ValueError` on a `source_type` outside Section 2's vocabulary. That is a
    programming error in this package (the values are hard-coded per tool, never taken
    from a caller or a model), so it is the one failure here that is raised rather than
    returned -- an envelope carrying an unknown `source_type` would pass silently through
    the grounding check's namespace and is worse than a loud failure at registration time.
    """
    if source_type not in SOURCE_TYPES:
        raise ValueError(
            f"unknown source_type {source_type!r}; expected one of {sorted(SOURCE_TYPES)}"
        )
    return {
        "source_type": source_type,
        "source_id": source_id,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }


def _as_result(payload: dict[str, Any], *, is_error: bool) -> CallToolResult:
    """One text block of pretty-printed JSON. `CallToolResult` is returned from the tool
    body rather than raised so `is_error` is ours to set: an exception escaping a tool
    would be wrapped by the MCP server as `Error executing tool <name>: <repr>`, which is
    a stack-trace-shaped string, not the plain-language message Section 2 requires."""
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, indent=2, default=str))],
        is_error=is_error,
    )


def ok(source_type: str, source_id: str, data: Any) -> CallToolResult:
    """A successful tool result: the minted `source` block plus `data`."""
    return _as_result({"source": source_block(source_type, source_id), "data": data}, is_error=False)


def failed(source_type: str, source_id: str, message: str) -> CallToolResult:
    """A failed tool result: the minted `source` block plus a plain-language `error`.

    The `source` block is present on failures too, so the envelope has one shape and a
    reader can always tell *which* source was unreachable. There is no `data` key, so a
    consumer that reads `payload["data"]` fails loudly on an error result rather than
    quietly treating an error as an empty answer.
    """
    return _as_result(
        {"source": source_block(source_type, source_id), "error": message}, is_error=True
    )


def payload_of(result: CallToolResult) -> dict[str, Any]:
    """Parse a result's single JSON text block back into a dict.

    Used by the tests, and by anything that needs to read a result it just built. Kept
    here so the encode/decode pair lives in one module and cannot drift.
    """
    (block,) = result.content
    assert isinstance(block, TextContent), f"expected one text block, got {type(block).__name__}"
    return json.loads(block.text)
