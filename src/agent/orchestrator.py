"""The human approval gate, and the last stage of Section 1's topology (Issue #127,
`docs/agent_design.md` Sections 1, 5, 10, 11 and 13).

    result = run("does 2nd_test-demo need a replacement bearing ordered?")

Everything from a Critic-passed recommendation through to a placed -- or declined -- order:

    recommendation shown to the human
             |
      approved   declined --> stop, nothing executed
         |
         v
      [C: Executor]

It is glue and nothing else. The three pieces it joins are built, tested and decided
elsewhere and none of them is re-implemented or modified here: Issue #124's `approval.py`
(`mint`/`consume`), Issue #125's validated `place_order`, and Issue #126's Executor client.

**Why this is its own module rather than a third stage in `pipeline.py`.** `pipeline.py` is
deliberately thin and side-effect-free -- a question goes in, a verified answer comes out, and
its only outward effects are the model and tool calls the two agents make. This module blocks
on a human at a terminal, mints a credential, and writes a row to a database. Folding a
blocking `input()` into `pipeline.py` would make `answer_and_verify` unusable in exactly the
headless paths that call it today (`tests/test_agent_pipeline.py`, the live tests, and this
module's own first stage). The A -> B seam stays reusable; this module consumes it.

**The open question, decided here rather than inherited silently: option (a).** The human, on
approving, supplies the order parameters directly; the recommendation text is *shown as
context and never parsed*. Nothing in this module reads `part_number`, `quantity` or
`bearing_id` out of `GroundedResponse.recommendation`, and there is no code path that could --
the executor's record is built from `Approved`, whose every field came from a terminal prompt.

Issue #127 names the tension precisely: Section 10 requires Agent C's input to be "re-derived
by the harness from the approved order record... not copied out of A's prose", but
`Draft.recommendation` is a plain `str | None`, so nothing in this repo produces those three
fields in a form the harness could take without parsing untrusted model text. Three reasons
favour (a):

1. **It is the strongest available form of Section 10's own requirement, not a compromise on
   it.** Under (a) the order's parameters do not originate from model output at all -- not
   from prose, and not from a verified structured field either. The harness-authenticated
   level of Section 10's trust table ("minted out-of-band, never derived from model output")
   is where the whole `(part_number, quantity, bearing_id)` tuple ends up sitting, which is
   exactly what that row describes.
2. **(b) is not an alternative to this gate; it is a later refinement of what the gate shows.**
   Section 5 requires a human approval regardless of how well-structured the recommendation
   is, so even with a structured `Draft.recommendation` a human would still have to approve
   the values at a prompt. (b) would change where the prompt's *default values* come from, not
   whether the prompt exists. Nothing built here would be thrown away by (b) -- it would gain
   a pre-fill step.
3. **It requires no change to an already-merged agent.** (b) means reopening
   `src/agent/answerer.py`'s `Draft` (#112) and threading a new structure through Section 6's
   citation and numeric-fidelity checks, which is a larger change to a merged, tested agent
   and belongs in its own issue.

**(b) is explicitly not closed off.** It remains a sensible follow-up to #112: a structured,
Section-6-verified recommendation would let this prompt offer the model's proposal as a
default the human confirms or overrides, which is a genuine usability gain over typing five
fields. What it must not become is a path where the human's confirmation is a formality over
values the harness never independently held -- so if it is built, the prompt below stays and
gains defaults, rather than being replaced by a bare yes/no over model-supplied scalars.

**Where `requested_by` comes from, flagged rather than decided silently.** PR #130 (#126)
recorded a real three-way inconsistency and this module is where it has to be resolved
concretely: Section 10 says `requested_by` is "supplied by the harness from the token record",
but `ApprovalToken`/`ApprovedOrder` (#124) carry no such field, and `place_order` (#125)
requires it as an ordinary caller-supplied argument -- so `OrderRequest` (#126) has five
fields, not Section 10's illustrative four. This module prompts the human for it, alongside
the other four, because under option (a) the human is already the source of every order
parameter and this is simply one more. Two consequences a reader should not have to work out:

- **`requested_by` and `approved_by` have different trust properties, and the asymmetry is
  real.** `approved_by` is carried *inside* the minted token and is derived by `place_order`
  from the validated token record, so no caller -- and no model -- can set it independently of
  a real approval. `requested_by` is a plain tool argument. It is supplied here from the
  human's own typing and never from model output, but it is not harness-authenticated the way
  `approved_by` is, and nothing downstream could tell the difference if some future caller
  filled it from somewhere worse.
- **Neither identity is authenticated, and cannot be.** Section 13 excludes multi-user and
  authenticated access by name; there is one person at one terminal. Maker-checker's premise
  is that the maker and the checker are different people, and in a single-reviewer demo they
  will usually be the same person typing two strings. `orders.approved_by`/`approved_at` being
  `NOT NULL` (Section 7) is a real structural control over *whether an approval happened*; it
  is not evidence about *who* approved, and this module does not pretend otherwise.

**The executor runs in a real subprocess, over real stdio (Issue #132).** This was the one
limitation PR #131 shipped with and flagged rather than worked around: `write_server.py`'s
`main()` built its own fresh, empty `ApprovalTokenStore` with no seam to inject one, so a
token minted here could never be consumed there, and the only working option was an
in-process server sharing this module's memory. #132 closed it. `executor_session()` below
starts a Unix-socket bridge serving *this* process's store, launches the write server through
#126's `write_tools()`, and hands it the socket path -- so the store stays here, in memory,
dying with the process exactly as Issue #124 built it, and only `consume` crosses the
boundary. `mint` is not reachable from the subprocess at all. The mechanism, and the four
alternatives it was chosen over, are documented in `src/agent/executor/token_bridge.py`.

**What that boundary is and is not worth, stated plainly rather than left to a reader.**
Section 2's process boundary exists so that *the reading agent* cannot reach `place_order` --
and that protection was never the thing at risk here, because Agent A's model is handed only
`readonly_tools()`' four tools and an injected instruction cannot emit a `tool_use` block for
a tool that is not in its request. That was true before #132 and is true after. What #132 buys
is a **second, independent layer** under the same claim -- the separation no longer rests
solely on which list a tool ended up in, which is this repository's established posture for
anything load-bearing (`src/serving/single_worker.py`'s two independent refusals, #84; Section
5's token validation alongside `orders.approved_by`'s `NOT NULL`). It also buys OS-level
memory isolation between this process, which holds the API key and the token store, and the
only code in the system that writes to the inventory database. It does **not** protect against
a compromised orchestrator, and nothing could: this process legitimately mints.

`build_executor()` remains for the in-process case, which is still what tests want when they
need to mint into and inspect the same store; `run_from_response_async` accepts an injected
session for that reason.

**Fail-closed, everywhere a decision is read.** Section 10 case 5's whole point is that no
text the model reads can produce a valid approval; this module extends the same posture to the
text the *human* types. `is_affirmative` requires an exact match against a short closed
vocabulary after stripping and lowercasing -- so "yes, order 5 of ZA-2115" is a **decline**,
not an approval with parameters attached, and an unreadable stdin (`EOFError`) is a decline
too. There is no default-yes, no timeout, and no re-prompt loop: a malformed parameter
declines the whole gate rather than asking again.

**No trace writing (Section 9).** `OrchestrationResult` below carries everything a trace record
would need -- the tiered response, whether the gate was reached, what the human decided, and
the executor's result -- so a later issue's writer can consume the returned object. Nothing
here opens a file.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from src.agent.critic.grounding import GROUNDED, PARTIAL, GroundedResponse
from src.agent.executor.approval import ApprovalTokenStore
from src.agent.executor.client import (
    ExecutionResult,
    OrderPlaced,
    OrderRequest,
    ToolCallSession,
    execute_order,
    write_tools,
)
from src.agent.executor.token_bridge import serve_token_store
from src.agent.inventory.build_db import DB_PATH, build_db
from src.agent.mcp.write_server import build_server as build_write_server
from src.agent.pipeline import answer_and_verify_async

# The closed vocabulary of an unambiguous affirmative. Matched exactly, after `.strip()` and
# `.lower()` -- see the module docstring on why containment would be the wrong test.
AFFIRMATIVE = frozenset({"y", "yes", "approve", "approved"})

DECLINED_BY_HUMAN = "the human declined"
DECLINED_UNREADABLE = "no readable answer was given"
DECLINED_BAD_PARAMETER = "the order parameters were not supplied in a usable form"


# --- What the human decided ------------------------------------------------------------
#
# A typed outcome rather than a raised exception or a bare bool, matching `approval.py`'s
# `ApprovedOrder`/`TokenError` and `client.py`'s `OrderPlaced`/`OrderRejected`: a decline is a
# value the caller must branch on, not something a forgotten `try` can skip past.


@dataclass(frozen=True)
class Approved:
    """Every order parameter, as the human supplied it at the gate. Nothing on this record
    was read out of the recommendation text -- see the module docstring's option (a)."""

    part_number: str
    quantity: int
    requested_by: str
    approved_by: str
    bearing_id: str | None = None


@dataclass(frozen=True)
class Declined:
    """The gate was reached and the answer was not an unambiguous affirmative. `reason` says
    which way it failed closed; every one of them stops the run identically."""

    reason: str = DECLINED_BY_HUMAN


ApprovalOutcome = Approved | Declined

ApprovalPrompt = Callable[[GroundedResponse], ApprovalOutcome]


@dataclass(frozen=True)
class OrchestrationResult:
    """One question, all the way through. Also the extension point Section 9's trace writer
    would consume -- it carries the tier, the gate decision and the executor's result, and
    this module writes none of it anywhere."""

    response: GroundedResponse
    gate_reached: bool
    approval: ApprovalOutcome | None = None
    execution: ExecutionResult | None = None

    @property
    def approved(self) -> bool:
        return isinstance(self.approval, Approved)


# --- The gate condition (Task 3) ---------------------------------------------------------


def needs_human_approval(response: GroundedResponse) -> bool:
    """Whether this response reaches the approval gate at all.

    All three conditions are checked here even though `grounding.assemble` already couples
    them (it nulls `recommendation` and clears `requires_approval` on a tier-3 response), so
    that the gate's precondition is legible in one place and does not silently depend on that
    coupling holding. A tier-3 response never reaches a human approval prompt: Section 6
    withholds the recommendation from the released text precisely because releasing a
    suggested action under "I don't have a sourced answer" would be answering un-grounded
    where it matters most, and offering that same action for approval would be worse.
    """
    return (
        response.grounding_tier in (GROUNDED, PARTIAL)
        and response.requires_approval
        and response.recommendation is not None
    )


# --- The terminal prompt (Task 2, Task 3) -------------------------------------------------


def is_affirmative(answer: str) -> bool:
    """Section 3's "anything other than an unambiguous affirmative is a decline", as one
    testable predicate. Exact match against `AFFIRMATIVE`, never containment."""
    return answer.strip().lower() in AFFIRMATIVE


def _read_line(read: Callable[[str], str], prompt: str) -> str | None:
    """One line, or `None` if stdin cannot be read. A closed stdin is a decline, not a
    crash and certainly not an approval."""
    try:
        return read(prompt)
    except EOFError:
        return None


def prompt_for_approval(
    response: GroundedResponse,
    *,
    read: Callable[[str], str] = input,
    write: Callable[[str], Any] = print,
) -> ApprovalOutcome:
    """Show the recommendation and its grounding tier, then block for approve/decline and,
    on approval, for the order parameters.

    `read`/`write` are injected so the flow is testable by feeding scripted lines rather than
    by monkeypatching builtins -- the same "inject the seam, don't patch the world" shape
    `readonly_server.py`'s `search` and `approval.py`'s `clock` already use.

    A malformed parameter declines the whole gate rather than re-prompting. A retry loop in a
    blocking terminal gate is a way to end up approving something after five attempts to say
    what was meant; failing closed and re-running the question is safer and simpler.
    """
    write("")
    write(f"Grounding tier: {response.grounding_tier}")
    write(f"Recommendation: {response.recommendation}")
    write(
        "This recommendation is shown as context only. It is never parsed, and no value "
        "below is taken from it -- you supply every order parameter yourself."
    )

    answer = _read_line(read, "Approve this order? [y/N]: ")
    if answer is None:
        return Declined(DECLINED_UNREADABLE)
    if not is_affirmative(answer):
        return Declined(DECLINED_BY_HUMAN)

    part_number = _read_line(read, "Part number: ")
    raw_quantity = _read_line(read, "Quantity: ")
    bearing_id = _read_line(read, "Bearing id (blank for none): ")
    requested_by = _read_line(read, "Requested by: ")
    approved_by = _read_line(read, "Approved by (your identity): ")

    fields = (part_number, raw_quantity, bearing_id, requested_by, approved_by)
    if any(field is None for field in fields):
        return Declined(DECLINED_UNREADABLE)

    return build_approval(
        part_number=part_number,
        raw_quantity=raw_quantity,
        bearing_id=bearing_id,
        requested_by=requested_by,
        approved_by=approved_by,
    )


def build_approval(
    *,
    part_number: str,
    raw_quantity: str,
    bearing_id: str,
    requested_by: str,
    approved_by: str,
) -> ApprovalOutcome:
    """Validate five typed-in strings into an `Approved`, or decline.

    Separate from the prompt so the validation is testable without a scripted terminal, and
    so there is exactly one place that decides what counts as a usable parameter. Quantity
    validity is checked here as a usability matter only -- `place_order` and
    `orders.place_order` reject a non-positive quantity independently (`approval.py`'s
    docstring says so explicitly), and this does not replace either.
    """
    part_number = part_number.strip()
    requested_by = requested_by.strip()
    approved_by = approved_by.strip()
    bearing = bearing_id.strip() or None

    if not part_number or not requested_by or not approved_by:
        return Declined(DECLINED_BAD_PARAMETER)

    try:
        quantity = int(raw_quantity.strip())
    except ValueError:
        return Declined(DECLINED_BAD_PARAMETER)
    if quantity <= 0:
        return Declined(DECLINED_BAD_PARAMETER)

    return Approved(
        part_number=part_number,
        quantity=quantity,
        requested_by=requested_by,
        approved_by=approved_by,
        bearing_id=bearing,
    )


# --- The executor's session ---------------------------------------------------------------


@asynccontextmanager
async def executor_session(
    store: ApprovalTokenStore, *, db_path: Path | None = None
) -> AsyncIterator[ToolCallSession]:
    """**The production path (Issue #132): a real write-server subprocess over real stdio.**

    `store` stays here, in this process, and only `consume` crosses the boundary -- over a
    Unix socket that exists for the duration of this block and is handed to the subprocess as
    a rendezvous path, never as a credential. `src/agent/executor/token_bridge.py` holds the
    mechanism and the alternatives it was chosen over.

    The ordering matters and is not incidental: the bridge is serving before the subprocess
    is launched, so the server can validate on its very first call rather than racing a
    socket that does not exist yet.
    """
    async with serve_token_store(store) as socket_path:
        async with write_tools(db_path=db_path, token_bridge=socket_path) as session:
            yield session


def build_executor(
    *, db_path: Path | None = None, token_store: ApprovalTokenStore | None = None
) -> tuple[ToolCallSession, ApprovalTokenStore]:
    """The write server and the `ApprovalTokenStore` it validates against, in **one process**.

    No longer the default path -- `executor_session` above is (Issue #132). This is kept
    because it remains the simplest way to hand a test a server and a store it can both mint
    into and inspect, which is what `tests/test_agent_orchestrator.py`,
    `tests/test_agent_executor_client.py` and `tests/test_agent_mcp_servers.py` all rely on,
    and because `run_from_response_async` still accepts an injected session for exactly that
    reason.

    `build_db` is called for the same reason both servers' `main()` calls it (Issue #101): a
    fresh clone gets a seeded database rather than a tool that reports the inventory as
    unavailable. It is idempotent and a no-op when the database already exists.
    """
    path = db_path if db_path is not None else DB_PATH
    build_db(path)
    server, _budget, store = build_write_server(db_path=path, token_store=token_store)
    return server, store


# --- The run (Task 4, Task 5) ------------------------------------------------------------


async def run_from_response_async(
    response: GroundedResponse,
    *,
    prompt: ApprovalPrompt = prompt_for_approval,
    session: ToolCallSession | None = None,
    token_store: ApprovalTokenStore | None = None,
    db_path: Path | None = None,
) -> OrchestrationResult:
    """Take one already-verified response through the gate and, only on approval, the
    executor.

    Split from `run_async` for the reason `pipeline.py` splits `verify_turn_async` from
    `answer_and_verify_async`: it lets the entire gate-and-execute half be exercised on a
    constructed response with no model call and no API key, which is what
    `tests/test_agent_orchestrator.py` does.

    Three properties this function is arranged to make true by construction rather than by
    care, each asserted in that test module:

    - A response that does not reach the gate never invokes `prompt` -- so nothing is shown,
      nothing is read, and there is no path to a mint.
    - A decline returns before `token_store.mint` is reached. Declining is declining, never
      "approve with different parameters", regardless of what the declining text contained.
    - The `OrderRequest` handed to the executor is built from `Approved`'s fields and the
      minted token, and from nothing else. `response` is not in scope for any of them.
    """
    if not needs_human_approval(response):
        return OrchestrationResult(response=response, gate_reached=False)

    outcome = prompt(response)
    if not isinstance(outcome, Approved):
        return OrchestrationResult(response=response, gate_reached=True, approval=outcome)

    if session is not None and token_store is None:
        raise ValueError("a caller-supplied session must come with the token store it shares")

    store = token_store if token_store is not None else ApprovalTokenStore()

    # Minted only now: after the human approved, scoped to exactly what they approved
    # (Section 5's "scoped to one order"), and never earlier in the flow.
    token = store.mint(
        outcome.part_number, outcome.quantity, outcome.bearing_id, outcome.approved_by
    )
    order = OrderRequest(
        part_number=outcome.part_number,
        quantity=outcome.quantity,
        approval_token=token.token,
        requested_by=outcome.requested_by,
        bearing_id=outcome.bearing_id,
    )

    if session is not None:
        execution = await execute_order(order, session)
    else:
        # Issue #132's default: a real subprocess, with `store` staying in this process and
        # only `consume` crossing the boundary.
        async with executor_session(store, db_path=db_path) as subprocess_session:
            execution = await execute_order(order, subprocess_session)

    return OrchestrationResult(
        response=response, gate_reached=True, approval=outcome, execution=execution
    )


async def run_async(
    question: str,
    *,
    prompt: ApprovalPrompt = prompt_for_approval,
    session: ToolCallSession | None = None,
    token_store: ApprovalTokenStore | None = None,
    db_path: Path | None = None,
    **pipeline_kwargs: Any,
) -> OrchestrationResult:
    """One question, all the way from Agent A to a placed or declined order.

    `db_path` is handed to both halves deliberately: the answerer's `check_inventory` reads
    the same database the executor writes to, so a caller pointing one at a temporary
    database must point both.
    """
    response = await answer_and_verify_async(question, db_path=db_path, **pipeline_kwargs)
    return await run_from_response_async(
        response,
        prompt=prompt,
        session=session,
        token_store=token_store,
        db_path=db_path,
    )


def run(question: str, **kwargs: Any) -> OrchestrationResult:
    """Synchronous entry point, and the one this module is meant to be used through: the
    whole flow is human-present and blocking by design (Section 11), so the async path exists
    only because the MCP stdio transport underneath it is async-native."""
    return asyncio.run(run_async(question, **kwargs))


def describe(result: OrchestrationResult, write: Callable[[str], Any] = print) -> None:
    """Surface the executor's result -- the order id, or its rejection reason unchanged --
    back to the terminal (Task 5). Kept separate from `run_async` so the flow itself returns
    a value rather than printing one, and a caller that wants neither can skip it."""
    if not result.gate_reached:
        write("No approval was requested for this response.")
        return
    if not result.approved:
        assert isinstance(result.approval, Declined)
        write(f"Declined: {result.approval.reason}. Nothing was ordered.")
        return
    if isinstance(result.execution, OrderPlaced):
        write(f"Order placed: id {result.execution.order_id}.")
    else:
        assert result.execution is not None
        write(f"Order rejected: {result.execution.reason}")
