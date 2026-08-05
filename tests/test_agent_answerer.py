"""Tier-1 tests for Agent A (Issue #112, `docs/agent_design.md` Sections 1, 5, 6 and 10).
No API key, no model call, no network beyond a local stdio pipe.

What is asserted here is the *wiring*, which is the part that can be checked without a model:
the request configuration matches Section 1, the client's tool surface matches Section 5, the
draft schema matches Section 6, the system prompt is genuinely static, and no untrusted text
reaches a message unwrapped. Whether the model then writes a good answer is Section 8's
golden set, a separate issue.

The one live call lives in `tests/test_agent_answerer_live.py` and is API-key-gated.
"""
from __future__ import annotations

import ast
import asyncio
import json
import re
import sqlite3
from pathlib import Path

import pytest
from mcp import ClientSession, stdio_client
from pydantic import ValidationError

from src.agent import answerer
from src.agent.answerer import (
    EFFORT,
    MAX_ITERATIONS,
    MAX_TOKENS,
    MODEL,
    SYSTEM_PROMPT,
    THINKING,
    Claim,
    Draft,
    build_messages,
    draft_schema,
    parse_draft,
    readonly_server_params,
    readonly_tools,
)
from src.agent.inventory.build_db import build_db
from src.agent.mcp.budget import MAX_TOOL_CALLS
from src.agent.mcp.readonly_server import READONLY_TOOL_NAMES
from src.agent.untrusted import TAG, UNTRUSTED_DATA_RULE, UntrustedEnvelope
from tests.fixtures.adversarial_payloads import CASE_9_ENVELOPE_BREAKOUT

REPO_ROOT = Path(__file__).resolve().parents[1]
DESIGN_DOC = REPO_ROOT / "docs" / "agent_design.md"
ANSWERER_SOURCE = REPO_ROOT / "src" / "agent" / "answerer.py"

# Nothing is listening here, so the serving-backed tools fail closed. That is fine: these
# tests measure the wiring, not the payloads.
CLOSED_PORT_URL = "http://127.0.0.1:9"


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "inventory.db"
    build_db(path)
    return path


class _TextBlock:
    """The shape `parse_draft` reads out of a final message."""

    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


# --- Section 1: model and request configuration ---------------------------------------


def test_request_configuration_is_section_1s():
    assert MODEL == "claude-opus-5"
    assert MAX_TOKENS == 16000
    assert EFFORT == "high"


def test_thinking_is_adaptive_and_is_not_disabled():
    """Section 1 gives a mechanical reason, not a preference: with thinking off this model
    can write a tool call into visible text instead of emitting a `tool_use` block, which in
    an agentic loop is a call that silently never runs."""
    assert THINKING == {"type": "adaptive"}
    assert THINKING.get("type") != "disabled"


def test_the_model_and_token_budget_match_the_design_documents_own_text():
    """Anti-drift, the same way `MODEL_NOTES` is pinned in #84: read Section 1 rather than
    trusting that a constant still says what the decision says."""
    section_1 = DESIGN_DOC.read_text(encoding="utf-8").split("## 2. The MCP tool layer")[0]

    assert f"`{MODEL}`" in section_1
    assert re.search(rf"\*\*`max_tokens`:\s*{MAX_TOKENS}\*\*,\s*non-streaming", section_1)
    assert "**Effort:** `high` for the answerer" in section_1


def test_the_loop_bound_sits_above_the_servers_tool_call_cap():
    """`max_iterations` is a bound on the *loop*, not a second cap: Section 2's cap of 8 is
    enforced by the server, where it holds regardless of what connects. This stops the runner
    spinning against a server that is refusing every call."""
    assert MAX_ITERATIONS > MAX_TOOL_CALLS


# --- Section 1: the cached, static prefix ----------------------------------------------


def _system_prompt_assignment() -> ast.AST:
    tree = ast.parse(ANSWERER_SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = getattr(node, "targets", [])
        if any(isinstance(t, ast.Name) and t.id == "SYSTEM_PROMPT" for t in targets):
            return node.value
    raise AssertionError("SYSTEM_PROMPT is no longer a module-level assignment")


def test_the_system_prompt_interpolates_nothing_per_request():
    """**The check Section 1 asks for, done statically.** "Nothing dynamic (timestamp,
    request ID, bearing ID) is interpolated into the system prompt — that would invalidate
    the prefix on every call."

    The failure mode is silent: a prefix that changes every call simply never caches, and
    nothing errors. So this reads the source and asserts that the only interpolation in the
    prompt's f-string is a bare module-level constant — a call, an attribute access, or an
    argument would all be ways for per-request data to arrive.
    """
    value = _system_prompt_assignment()
    interpolations = [n for n in ast.walk(value) if isinstance(n, ast.FormattedValue)]

    for node in interpolations:
        assert isinstance(node.value, ast.Name), (
            "the system prompt interpolates an expression, not a constant: "
            f"{ast.dump(node.value)[:120]}"
        )
        assert node.value.id in {"UNTRUSTED_DATA_RULE"}, (
            f"the system prompt interpolates {node.value.id}, which is not a known constant"
        )


def test_the_system_prompt_is_byte_identical_across_calls_and_questions():
    """The property the static check exists to protect, asserted on the rendered blocks:
    two different questions produce the same system prefix, byte for byte."""
    first = answerer._system_blocks()
    second = answerer._system_blocks()

    assert first == second
    assert first[0]["text"] == SYSTEM_PROMPT


def test_the_stable_prefix_carries_the_cache_breakpoint():
    """Tools render before `system`, so one breakpoint on the last system block caches the
    tool definitions with it — which is exactly the stable prefix Section 1 describes."""
    blocks = answerer._system_blocks()

    assert blocks[-1]["cache_control"] == {"type": "ephemeral"}


def test_the_system_prompt_carries_section_10s_standing_rule_verbatim():
    """Section 10: "The trusted system prompt carries one standing rule, phrased in a single
    sentence so it can be quoted and tested"."""
    assert UNTRUSTED_DATA_RULE in SYSTEM_PROMPT


# --- Section 10: no untrusted text reaches a message unwrapped -------------------------


def test_the_question_enters_the_conversation_enveloped():
    envelope = UntrustedEnvelope()

    (message,) = build_messages("what is the status of 2nd_test-demo?", envelope)

    assert message["role"] == "user"
    assert message["content"].startswith(f'<{TAG} source_id="technician-question"')
    assert message["content"].endswith(envelope.closing_tag)


def test_a_question_carrying_the_case_9_payload_cannot_break_out_of_its_envelope():
    """Section 10 puts the technician's question in `untrusted-data` precisely because it
    arrives over the same interface whether a technician or an attacker typed it. The
    breakout payload is as contained there as it is in a retrieved chunk."""
    envelope = UntrustedEnvelope()

    (message,) = build_messages(CASE_9_ENVELOPE_BREAKOUT, envelope)

    assert message["content"].count(f"</{TAG}") == 1
    assert message["content"].endswith(envelope.closing_tag)


def test_each_request_wraps_its_question_with_a_different_nonce():
    first = build_messages("q", UntrustedEnvelope())[0]["content"]
    second = build_messages("q", UntrustedEnvelope())[0]["content"]

    assert first != second


# --- Section 5: the tool surface, against a real server process -------------------------


def test_the_answerer_launches_the_read_only_server_and_only_that():
    """Static half of the least-privilege claim: `place_order` lives on a different module,
    and this agent never names it."""
    params = readonly_server_params()

    assert params.args[:2] == ["-m", "src.agent.mcp.readonly_server"]
    assert not any("write_server" in arg for arg in params.args)


def test_the_answerers_tools_are_exactly_the_four_read_only_ones(db_path):
    """The real thing, over a real stdio transport to a real subprocess — #111's pattern,
    applied to the client this agent actually builds."""

    async def run() -> list[str]:
        async with readonly_tools(
            UntrustedEnvelope(), serving_url=CLOSED_PORT_URL, db_path=db_path
        ) as (_session, tools):
            return [tool.name for tool in tools]

    names = asyncio.run(run())

    assert names == list(READONLY_TOOL_NAMES)
    assert "place_order" not in names


def test_the_answerers_own_session_cannot_reach_place_order(db_path):
    """Section 10 case 6, on this agent's actual connection: `place_order` is not filtered
    out of the answerer's tool list — it does not exist on the transport the answerer holds,
    so a perfectly well-formed order comes back "Unknown tool" and writes nothing."""
    order = {
        "part_number": "BRG-6205-2RS",
        "quantity": 1,
        "requested_by": "tech-01",
        "approved_by": "supervisor-02",
        "approved_at": "2026-08-05T09:00:00+00:00",
    }

    async def run() -> dict:
        params = readonly_server_params(serving_url=CLOSED_PORT_URL, db_path=db_path)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("place_order", order)
                return {
                    "is_error": bool(result.is_error),
                    "text": result.content[0].text if result.content else "",
                }

    seen = asyncio.run(run())

    assert seen["is_error"] is True
    assert "Unknown tool: place_order" in seen["text"]

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0
    finally:
        conn.close()


def test_every_tool_result_comes_back_enveloped(db_path):
    """The chokepoint, exercised rather than described: the tool functions handed to
    `tool_runner` are the enveloping ones, so a result cannot reach a message unwrapped by
    someone forgetting to call a helper."""

    async def run() -> str:
        envelope = UntrustedEnvelope()
        async with readonly_tools(
            envelope, serving_url=CLOSED_PORT_URL, db_path=db_path
        ) as (_session, tools):
            tool = {t.name: t for t in tools}["check_inventory"]
            result = await tool.call({"part_number": "BRG-6205-2RS"})
            assert result.endswith(envelope.closing_tag)
            return result

    rendered = asyncio.run(run())

    assert rendered.startswith(f'<{TAG} source_id="tool:check_inventory"')
    # The tool-minted source block is inside the envelope, where Section 10 puts every tool
    # result -- and where Section 6's citation check still finds the id it compares.
    assert '"source_id": "data/agent/inventory.db"' in rendered


def test_a_failed_tool_result_is_enveloped_too(db_path):
    """A failure message is still runtime text. It also has to stay a *failure*: the runner
    renders `ToolError` with `is_error` set, which preserves the pairing Section 2 requires
    and gives the model something to degrade from."""
    from anthropic.lib.tools._beta_functions import ToolError

    async def run() -> str:
        envelope = UntrustedEnvelope()
        async with readonly_tools(
            envelope, serving_url=CLOSED_PORT_URL, db_path=db_path
        ) as (_session, tools):
            tool = {t.name: t for t in tools}["get_bearing_status"]
            with pytest.raises(ToolError) as excinfo:
                await tool.call({})
            return excinfo.value.content

    content = asyncio.run(run())

    assert content.startswith(f'<{TAG} source_id="tool:get_bearing_status"')
    assert "the prediction service is not reachable" in content


# --- Section 6: the draft's shape -------------------------------------------------------


def test_the_schema_is_section_6s_three_keys():
    schema = draft_schema()

    assert set(schema["properties"]) == {"claims", "recommendation", "unanswered"}
    assert schema["required"] == ["claims", "recommendation", "unanswered"]
    claim = schema["$defs"]["Claim"]
    assert set(claim["properties"]) == {"text", "source_ids"}
    assert claim["required"] == ["text", "source_ids"]


def test_the_schema_forbids_extra_properties_everywhere():
    """Structured outputs requires `additionalProperties: false` on every object, and it is
    also what stops the model inventing a field a checker would not read."""
    schema = draft_schema()

    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Claim"]["additionalProperties"] is False


def test_a_recommendation_is_nullable_rather_than_optional():
    """Section 6 shows all three keys present. "The key is missing" and "there is no
    recommendation" are different things to a checker, so the key is required and its value
    is nullable."""
    schema = draft_schema()

    assert schema["properties"]["recommendation"]["anyOf"] == [
        {"type": "string"},
        {"type": "null"},
    ]


def test_a_hand_built_draft_validates():
    draft = Draft(
        claims=[
            Claim(
                text="Bearing 2nd_test-demo has a stable baseline after 120 windows.",
                source_ids=["GET /monitoring/drift"],
            ),
            Claim(
                text="Critical recall on the held-out 1st_test fold is 0.059.",
                source_ids=["docs/model_training_decision.md::7"],
            ),
        ],
        recommendation=None,
        unanswered=["whether the vibration spike at 03:00 was mechanical"],
    )

    assert len(draft.claims) == 2
    assert draft.recommendation is None


def test_a_claim_without_citations_is_still_structurally_valid():
    """Citation *coverage* is the critic's job (Section 6, step 3), not the schema's. The
    schema requires the key so an uncited claim is visible as an empty list rather than as a
    missing field a checker has to guess about."""
    draft = Draft(claims=[Claim(text="uncited", source_ids=[])], recommendation=None, unanswered=[])

    assert draft.claims[0].source_ids == []


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"claims": [{"text": "x"}], "recommendation": None, "unanswered": []},
                     id="claim-missing-source_ids"),
        pytest.param({"claims": [], "unanswered": []}, id="recommendation-key-absent"),
        pytest.param(
            {"claims": [], "recommendation": None, "unanswered": [], "grounding_tier": "grounded"},
            id="extra-key",
        ),
    ],
)
def test_a_malformed_draft_is_rejected(payload):
    with pytest.raises(ValidationError):
        Draft.model_validate(payload)


def test_parse_draft_reads_the_final_messages_json():
    payload = {
        "claims": [{"text": "a", "source_ids": ["tool:check_inventory"]}],
        "recommendation": "order one ZA-2115",
        "unanswered": [],
    }

    draft = parse_draft([_TextBlock(json.dumps(payload))])

    assert draft.recommendation == "order one ZA-2115"
    assert draft.claims[0].source_ids == ["tool:check_inventory"]


def test_parse_draft_fails_loudly_on_an_empty_response():
    """`output_config.format` guarantees schema-conformant JSON, so this is a parse rather
    than a rescue: if the guarantee ever does not hold, a checker should not receive a
    half-shaped draft."""
    with pytest.raises(ValueError, match="no text block"):
        parse_draft([])
