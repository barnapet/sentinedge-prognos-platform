"""The 8-tool-call cap (Issue #110, `docs/agent_design.md` Section 2).

> **Hard cap: 8 tool calls per question.** Hitting it ends the loop and produces the
> degraded response rather than continuing. This is both a cost control and a security
> control -- Section 10 case 7 is a question crafted to cause unbounded tool looping.

Issue #110 requires the cap at the tool-runner/server level "so it's structurally true
regardless of what calls these servers later". So it lives here, on the server side of the
process boundary, and every registered tool goes through `guard()` before its body runs.
An agent loop that forgot to count, or was talked out of counting by an injected
instruction, still gets refused by the ninth call -- the refusal is not something the
model can decline to perform.

**Scope: one budget per server process.** Section 2 says "per question", and the unit this
side of the boundary can see is the stdio session, not the question. A harness that starts
one server process per question therefore gets exactly Section 2's semantics with no
further work, and that is the intended wiring -- but the wiring itself is a later issue
(#110 excludes the agent loop and client configuration), so this module states the
constraint rather than implementing it. `reset()` exists for the in-process case and is
deliberately **not** exposed as a tool: a model that could reset its own cap does not have
a cap.

Refusal is a tool result with `is_error = True`, not an exception -- same reason as every
other failure here (`results.py`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mcp.types import CallToolResult

from src.agent.mcp.results import failed

# Section 2's number. Not configurable per call site on purpose: a cap that each server
# picks for itself is a default, not a cap.
MAX_TOOL_CALLS = 8

BUDGET_EXHAUSTED = (
    f"the {MAX_TOOL_CALLS}-tool-call limit for this question has been reached; "
    "answer from what has already been gathered, or say what could not be established"
)


@dataclass
class ToolCallBudget:
    """A monotonic counter of tool calls served by one server process."""

    limit: int = MAX_TOOL_CALLS
    used: int = field(default=0, init=False)

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    def reset(self) -> None:
        """Start a new question's budget. Callable in-process only -- see module docstring
        for why this is not a tool."""
        self.used = 0

    def guard(self, source_type: str, source_id: str) -> CallToolResult | None:
        """Charge one call against the budget.

        Returns `None` when the call may proceed, or the refusal result when it may not.
        The refusal carries the tool's own `source` block so the envelope shape is
        identical to every other failure, and **the attempt still counts**: a caller that
        keeps hammering a refused server does not get a free retry each time, so the
        counter cannot be walked backwards by failing calls.
        """
        self.used += 1
        if self.used > self.limit:
            return failed(source_type, source_id, BUDGET_EXHAUSTED)
        return None
