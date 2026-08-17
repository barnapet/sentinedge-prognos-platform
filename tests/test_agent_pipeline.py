"""Tier-1 tests for the A → B pipeline (Issue #116, `docs/agent_design.md` Sections 5 and 6).
No API key, no model call, no network beyond a local stdio pipe.

**The turn these tests replay is a real one.** `tests/fixtures/answerer_turn.json` was written
by `tests/fixtures/record_answerer_turn.py` against real infrastructure — a real serving
process with a real tracked bearing, the real `prognos_docs` Qdrant collection with real
embeddings and real cosine scores, and the real inventory database — through the real
read-only MCP server subprocess and the real `ToolResultRecorder`. Its `tool_payloads` are
what those services actually returned; nothing in them is hand-typed. Which half of the
fixture is model-authored is recorded in the file itself, and
`test_the_fixture_says_plainly_how_it_was_recorded` asserts the tests agree with what it
says.

Four burdens, carried by four groups of tests:

1. The fixture really is a recording, and really has the shape the critic consumes.
2. The answerer's recording plumbing works against a real MCP server subprocess.
3. The critic half runs end to end on the replayed real turn.
4. `answer_and_verify_async` genuinely joins the two, with a replayed `tool_runner`.

The one live call lives in `tests/test_agent_pipeline_live.py` and is API-key-gated.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from src.agent.answerer import Draft, ToolResultRecorder, readonly_tools
from src.agent.critic.deterministic import TurnEvidence, verify
from src.agent.critic.escalation import escalations_needed
from src.agent.critic.grounding import GROUNDED, PARTIAL, TIERS, UNGROUNDED
from src.agent.critic.retrieval_confidence import TAU_TOP
from src.agent.inventory.build_db import build_db
from src.agent.mcp.readonly_server import READONLY_TOOL_NAMES
from src.agent.mcp.results import SOURCE_TYPES
from src.agent.mcp.tools import INVENTORY_SOURCE_ID
from src.agent.pipeline import (
    answer_and_verify_async,
    evidence_for,
    turn_from_payloads,
    verify_turn_async,
)
from src.agent.untrusted import UntrustedEnvelope

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "answerer_turn.json"

# Nothing is listening here, so the serving-backed tools fail closed — which is fine, and in
# one test below is the point.
CLOSED_PORT_URL = "http://127.0.0.1:9"


@pytest.fixture(scope="module")
def recorded() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture()
def recorded_turn(recorded):
    return turn_from_payloads(Draft.model_validate(recorded["draft"]), recorded["tool_payloads"])


@pytest.fixture()
def db_path(tmp_path):
    path = tmp_path / "inventory.db"
    build_db(path)
    return path


# --- 1. The fixture is a recording, not an approximation ---------------------------------


def test_the_fixture_says_plainly_how_it_was_recorded(recorded):
    """The one thing a reader must not have to guess. `tool_payloads` is always a real
    recording; `draft` is model-authored only when the recorder was run with credentials, and
    the file says which."""
    assert recorded["draft_source"] in (
        "live_model_call",
        "synthesized_from_recorded_payloads",
    )
    assert recorded["recorded_at"]
    assert recorded["question"]


def test_the_recorded_payloads_carry_real_minted_source_blocks(recorded):
    """Every payload is `src/agent/mcp/results.py`'s shape, with a `source_type` from Section
    2's closed vocabulary — because these came out of that module, not out of a fixture
    author's head."""
    payloads = recorded["tool_payloads"]

    assert payloads, "the recording captured no tool results at all"
    for payload in payloads:
        assert payload["source"]["source_type"] in SOURCE_TYPES
        assert payload["source"]["source_id"]
        assert payload["source"]["retrieved_at"]
        assert ("data" in payload) ^ ("error" in payload)


def test_the_recorded_chunks_are_real_documents_with_real_similarity_scores(recorded):
    """A recorded chunk id names a file that exists, and its text appears verbatim in that
    file — Section 4's no-authored-chunks rule, checked against the recording. A hand-built
    fixture cannot pass this test by accident."""
    evidence = TurnEvidence.from_tool_payloads(recorded["tool_payloads"])
    scored = [item for item in evidence.items if item.score is not None]

    assert scored, "the recording captured no retrieved chunks"
    for item in scored:
        assert 0.0 < item.score <= 1.0
        path, _, index = item.source_id.rpartition("::")
        assert index.isdigit(), item.source_id
        source_file = REPO_ROOT / path
        assert source_file.is_file(), f"{item.source_id} names a file that does not exist"
        assert item.text in source_file.read_text(encoding="utf-8"), (
            f"{item.source_id}'s recorded text is not verbatim in {path}"
        )


def test_the_recorded_turns_retrieval_clears_section_6s_starting_threshold(recorded):
    """Not a calibration — an observation from the one real turn on record. The starting
    values were chosen with no data at all, so it is worth knowing that a real question's
    retrieval is not marginal against them."""
    evidence = TurnEvidence.from_tool_payloads(recorded["tool_payloads"])

    assert evidence.retrieval_scores
    assert evidence.retrieval_scores[0] > TAU_TOP


def test_every_citation_in_the_recorded_draft_resolves_to_a_recorded_id(recorded_turn):
    """The plumbing's whole point, stated as one assertion: the ids the draft cites are the
    ids the turn's tool results actually carried."""
    evidence = evidence_for(recorded_turn)

    cited = {sid for claim in recorded_turn.draft.claims for sid in claim.source_ids}
    assert cited, "the recorded draft cites nothing"
    assert cited <= evidence.source_ids


def test_issue_119s_scoping_on_the_recorded_turns_own_measurements(recorded_turn):
    """Issue #119, measured on #117's recorded turn rather than on a built example.

    #117 reported trigger (b) firing on 100% of claims, with these four (claim, source)
    overlaps against an unchanged floor of 0.6. Every number below is #117's, re-measured;
    what changed is which of them the trigger is allowed to consult.
    """
    from src.agent.critic.escalation import PROSE_SOURCE_TYPES, lexical_overlap

    evidence = evidence_for(recorded_turn)
    by_id = {item.source_id: item for item in evidence.items}
    measured = {
        (index, sid): (by_id[sid].source_type, round(lexical_overlap(claim.text, by_id[sid].text), 3))
        for index, claim in enumerate(recorded_turn.draft.claims)
        for sid in claim.source_ids
    }

    assert measured == {
        (0, "GET /monitoring/drift"): ("live_endpoint", 0.333),
        (1, "docs/class_imbalance_decision.md::8"): ("decision_doc", 0.25),
        (1, "docs/class_imbalance_decision.md::9"): ("decision_doc", 0.417),
        (1, "docs/model_training_decision.md::9"): ("decision_doc", 0.25),
    }, "#117's measured table, unchanged — this issue does not move any overlap"

    # Claim 0 cites only live-tool JSON, and `"file_count": 197` supports it exactly.
    assert PROSE_SOURCE_TYPES.isdisjoint(
        {by_id[sid].source_type for sid in recorded_turn.draft.claims[0].source_ids}
    )
    # Claim 1 cites prose, and its best chunk is still below the unchanged floor.
    assert recorded_turn.draft.recommendation is None, "so only trigger (b) is in play here"

    report = verify(recorded_turn.draft, evidence)
    requests = escalations_needed(recorded_turn.draft, report, evidence)

    assert report.clean is True
    assert [(r.claim_index, r.trigger, r.source_id) for r in requests] == [
        (1, "lexical_overlap", "docs/class_imbalance_decision.md::9")
    ], "the JSON-cited claim no longer escalates; the prose-cited one still does"


# --- 2. The answerer's recorder, against a real MCP server subprocess ---------------------


def test_the_recorder_captures_real_payloads_through_the_real_tool_path(db_path):
    """The answerer-side half of #116, exercised rather than described: a real subprocess, a
    real stdio transport, the real enveloping wrapper, and the real recorder. `check_inventory`
    is used because it is the one tool that is fully offline — no serving process, no Qdrant —
    so this is deterministic in CI and on a laptop alike."""

    async def run() -> list[dict]:
        recorder = ToolResultRecorder()
        async with readonly_tools(
            UntrustedEnvelope(), CLOSED_PORT_URL, db_path, recorder
        ) as (_session, tools):
            by_name = {tool.name: tool for tool in tools}
            await by_name["check_inventory"].call({"part_number": "ZA-2115"})
        return recorder.payloads

    payloads = asyncio.run(run())

    (payload,) = payloads
    assert payload["source"]["source_id"] == INVENTORY_SOURCE_ID
    assert payload["data"]["parts"][0]["part_number"] == "ZA-2115"

    evidence = TurnEvidence.from_tool_payloads(payloads)
    assert INVENTORY_SOURCE_ID in evidence.source_ids


def test_a_failing_tool_is_recorded_rather_than_dropped(db_path):
    """A failure carries a minted `source` block and a plain-language message, and "the
    prediction service is not reachable" is a real observation a claim may cite. Dropping it
    would make a claim about it look uncited to the critic."""

    async def run() -> list[dict]:
        recorder = ToolResultRecorder()
        async with readonly_tools(
            UntrustedEnvelope(), CLOSED_PORT_URL, db_path, recorder
        ) as (_session, tools):
            by_name = {tool.name: tool for tool in tools}
            with pytest.raises(Exception):
                await by_name["get_bearing_status"].call({"bearing_id": "2nd_test-demo"})
        return recorder.payloads

    (payload,) = asyncio.run(run())

    assert "error" in payload
    assert payload["source"]["source_id"]


def test_the_recorder_never_breaks_the_tool_path_on_unparseable_output():
    """It runs inside a tool call, so a recorder that could raise would be a monitoring
    feature that takes the agent down."""
    recorder = ToolResultRecorder()

    recorder.record("not json at all")
    recorder.record("[1, 2, 3]")
    recorder.record(json.dumps({"no": "source block"}))

    assert recorder.payloads == []


def test_no_recorder_means_no_recording_and_no_change_to_the_tool_path(db_path):
    """#113's behaviour is the default: the recorder is optional and absent by default, which
    is why none of #113's own tests needed changing."""

    async def run() -> str:
        async with readonly_tools(UntrustedEnvelope(), CLOSED_PORT_URL, db_path) as (
            _session,
            tools,
        ):
            by_name = {tool.name: tool for tool in tools}
            return await by_name["check_inventory"].call({"part_number": "ZA-2115"})

    wrapped = asyncio.run(run())

    assert wrapped.startswith("<untrusted-data "), "the result is still enveloped"


# --- 3. The critic half, on the replayed real turn ----------------------------------------


def test_the_real_recorded_turn_verifies_clean_and_releases_grounded(recorded_turn):
    """The deterministic tiers over the real turn: every citation resolves, every number in
    every claim appears verbatim in a source it cites, retrieval clears the threshold."""
    response = asyncio.run(verify_turn_async(recorded_turn, escalate=False))

    assert response.grounding_tier == GROUNDED
    assert response.report.clean is True
    assert len(response.claims) == len(recorded_turn.draft.claims)
    assert response.retrieval.passed is True

    for claim in response.claims:
        assert claim.source_ids
        assert f"[{', '.join(claim.source_ids)}]" in response.text


def test_the_draft_s_unanswered_parts_survive_into_the_released_text(recorded_turn):
    response = asyncio.run(verify_turn_async(recorded_turn, escalate=False))

    for unanswered in recorded_turn.draft.unanswered:
        assert unanswered in response.text


class _TextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _Message:
    def __init__(self, text: str) -> None:
        self.content = [_TextBlock(text)]


class _StubCriticMessages:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Message(json.dumps({"verdict": self.verdict}))


class _StubCritic:
    """A critic client with no tool surface at all — see #115's tests for why that is the
    only shape Agent B ever gets."""

    def __init__(self, verdict: str = "yes") -> None:
        self.messages = _StubCriticMessages(verdict)


def test_the_llm_critic_is_reached_only_through_section_6s_escalation_rule(recorded_turn):
    """The escalated path is wired: with `escalate=False` no call is made at all, and with it
    on, the number of calls is exactly the number of claim/chunk pairs the rule selected."""
    from src.agent.critic.deterministic import verify
    from src.agent.critic.escalation import escalations_needed

    evidence = evidence_for(recorded_turn)
    expected = escalations_needed(
        recorded_turn.draft, verify(recorded_turn.draft, evidence), evidence
    )

    silent = _StubCritic("yes")
    asyncio.run(verify_turn_async(recorded_turn, critic_client=silent, escalate=False))
    assert silent.messages.calls == []

    called = _StubCritic("yes")
    asyncio.run(verify_turn_async(recorded_turn, critic_client=called, escalate=True))
    assert len(called.messages.calls) == len(expected)


def test_a_no_verdict_from_the_llm_critic_degrades_the_real_turn(recorded_turn):
    """`no` demotes rather than rewrites, and a turn whose every claim is demoted is tier 3 —
    Section 6's table, reached through the real pipeline over a real turn."""
    response = asyncio.run(
        verify_turn_async(recorded_turn, critic_client=_StubCritic("no"), escalate=True)
    )

    assert response.grounding_tier in (PARTIAL, UNGROUNDED)
    for dropped in response.dropped:
        assert dropped.text in {claim.text for claim in recorded_turn.draft.claims}


def test_a_yes_verdict_leaves_the_real_turn_grounded(recorded_turn):
    response = asyncio.run(
        verify_turn_async(recorded_turn, critic_client=_StubCritic("yes"), escalate=True)
    )

    assert response.grounding_tier == GROUNDED


# --- Issue #171: the concept-domain check, wired into this call site ------------------------


def test_the_concept_domain_check_leaves_the_real_recorded_turn_untouched(recorded_turn):
    """The first thing a new gate has to prove: it does not demote a true claim. The recorded
    turn's claim 0 cites `GET /monitoring/drift` and nothing else — precisely the case that had
    no relevance check at all — and `scored`/`windows` are what that source reports on, so it
    survives and the turn is still `grounded` with no model call made."""
    from src.agent.critic.relevance import check_claim_domain, off_domain_demotions

    evidence = evidence_for(recorded_turn)
    report = verify(recorded_turn.draft, evidence)
    checked = check_claim_domain(recorded_turn.draft.claims[0], evidence)

    assert checked.checked is True, "the live-only claim really is evaluated by this check"
    assert checked.off_domain is False
    assert off_domain_demotions(report, evidence) == {}

    response = asyncio.run(verify_turn_async(recorded_turn, escalate=False))
    assert response.grounding_tier == GROUNDED
    assert len(response.claims) == len(recorded_turn.draft.claims)


def test_an_off_domain_live_claim_is_demoted_with_no_api_key_and_no_escalation(recorded_turn):
    """The check reaches the response through `escalate=False`, which is what makes it
    deterministic-tier: same real payloads, one claim replaced with an off-domain one, no
    critic client anywhere."""
    from src.agent.answerer import Claim
    from src.agent.critic.grounding import OFF_DOMAIN_REASON

    off_domain = Claim(
        text="The oil temperature alarm setpoint for this machine is configured at the rig.",
        source_ids=["GET /monitoring/drift"],
    )
    turn = turn_from_payloads(
        Draft(
            claims=[off_domain, *recorded_turn.draft.claims[1:]],
            recommendation=None,
            unanswered=[],
        ),
        [dict(payload) for payload in recorded_turn.tool_payloads],
    )

    response = asyncio.run(verify_turn_async(turn, escalate=False))

    assert response.grounding_tier == PARTIAL
    assert off_domain.text not in {claim.text for claim in response.claims}
    (dropped,) = response.dropped
    assert dropped.text == off_domain.text
    assert dropped.reasons == (f"{OFF_DOMAIN_REASON} (checked against GET /monitoring/drift)",)


def test_both_demotion_sources_merge_and_neither_reason_is_lost(recorded_turn):
    """The merge at this call site, on a claim both sources demote. `grounding.py` reports one
    reason per drop, so the two are joined rather than one overwriting the other — and the
    escalation set is unchanged by the deterministic demotion above it, so the number of model
    calls is still Section 6's rule and nothing else."""
    from src.agent.answerer import Claim
    from src.agent.critic.grounding import DEMOTED_REASON, OFF_DOMAIN_REASON

    # Cites the live source (off-domain for it) *and* a prose chunk the recorded turn really
    # retrieved, so trigger (b) can fire on the prose while this check fires on the live id.
    both = Claim(
        text="The oil temperature alarm setpoint for this machine is configured at the rig.",
        source_ids=["GET /monitoring/drift", "docs/class_imbalance_decision.md::9"],
    )
    live_only = Claim(
        text="The lockout procedure must be completed before the housing cover is opened.",
        source_ids=["GET /monitoring/drift"],
    )
    # The recommendation is what makes the merge reachable at all, and it is #119's rule rather
    # than a fixture convenience: trigger (b) is prose-scoped, so a live-only claim is escalated
    # only when the draft carries a recommendation (trigger (a)), which considers every cited
    # source. Without one, `live_only` would be demoted by this check and never seen by the LLM
    # tier — the case the test above already covers.
    turn = turn_from_payloads(
        Draft(
            claims=[both, live_only],
            recommendation="Order 1 x ZA-2115 for this bearing.",
            unanswered=[],
        ),
        [dict(payload) for payload in recorded_turn.tool_payloads],
    )
    critic = _StubCritic("no")

    response = asyncio.run(verify_turn_async(turn, critic_client=critic, escalate=True))

    # `both` cites prose, so this check does not evaluate it; the LLM tier does, and demotes it.
    # `live_only` cites the live source alone: demoted here, and — because the escalation set is
    # computed from the deterministic report alone — still put to the critic, which also demotes
    # it. That claim is the one carrying two reasons.
    reasons = {dropped.text: dropped.reasons for dropped in response.dropped}
    assert reasons[both.text] == (
        f"{DEMOTED_REASON} (checked against docs/class_imbalance_decision.md::9: no)",
    )
    (merged,) = reasons[live_only.text]
    assert merged.startswith(f"{OFF_DOMAIN_REASON} (checked against GET /monitoring/drift)")
    assert merged.endswith(f"{DEMOTED_REASON} (checked against GET /monitoring/drift: no)")
    assert response.grounding_tier == UNGROUNDED, "no claim survived either check"


# --- 4. The join: `answer_and_verify` with a replayed tool_runner --------------------------


class _ReplayedRunner:
    """A `tool_runner` whose model half is replayed and whose tools are real.

    It calls the tools the recorded turn called — really, over the real stdio transport — and
    then returns the recorded final message. Everything downstream of the model is production
    code: the enveloping wrapper, the recorder, `parse_draft`, and the whole critic half.
    """

    def __init__(self, tools, calls, final_text: str) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self._calls = calls
        self._final_text = final_text

    async def until_done(self) -> _Message:
        for name, arguments in self._calls:
            try:
                await self._tools[name].call(arguments)
            except Exception:  # noqa: BLE001 - a failed tool is a real outcome, and recorded
                pass
        return _Message(self._final_text)


class _ReplayedClient:
    def __init__(self, calls, final_text: str) -> None:
        self._calls = calls
        self._final_text = final_text
        self.runner_kwargs: dict = {}
        self.beta = self
        self.messages = self

    def tool_runner(self, **kwargs):
        self.runner_kwargs = kwargs
        return _ReplayedRunner(kwargs["tools"], self._calls, self._final_text)


def test_answer_and_verify_joins_a_real_tool_result_to_a_verified_response(db_path):
    """The end-to-end join, with no network and no key: a real MCP server subprocess, a real
    `check_inventory` result, the real recorder, and the real critic. The draft is replayed;
    the id it cites is one the tool really minted in this run, not one written into a
    fixture."""
    draft = Draft(
        claims=[
            {
                "text": "Part ZA-2115 has 4 units on hand.",
                "source_ids": [INVENTORY_SOURCE_ID],
            }
        ],
        recommendation=None,
        unanswered=[],
    )
    client = _ReplayedClient(
        calls=[("check_inventory", {"part_number": "ZA-2115"})],
        final_text=draft.model_dump_json(),
    )

    response = asyncio.run(
        answer_and_verify_async(
            "how many ZA-2115 do we have?",
            client=client,
            serving_url=CLOSED_PORT_URL,
            db_path=db_path,
            escalate=False,
        )
    )

    assert response.grounding_tier in TIERS
    assert response.grounding_tier == GROUNDED
    assert response.report.clean is True
    assert [claim.source_ids for claim in response.claims] == [(INVENTORY_SOURCE_ID,)]
    assert "4 units on hand" in response.text

    # The quantity really came from the database this run wrote and read.
    with sqlite3.connect(db_path) as conn:
        on_hand = conn.execute(
            "SELECT quantity_on_hand FROM parts WHERE part_number = 'ZA-2115'"
        ).fetchone()[0]
    assert str(on_hand) in response.text


def test_a_claim_citing_an_id_this_turn_never_produced_cannot_survive_the_pipeline(db_path):
    """Section 10's shape at the pipeline level: the draft is otherwise perfect, the id is
    plausible, and it was not minted this turn. The response is tier 3 and carries no claim."""
    draft = Draft(
        claims=[
            {
                "text": "Part ZA-2115 has 4 units on hand.",
                "source_ids": ["data/agent/inventory.db::fabricated"],
            }
        ],
        recommendation=None,
        unanswered=[],
    )
    client = _ReplayedClient(
        calls=[("check_inventory", {"part_number": "ZA-2115"})],
        final_text=draft.model_dump_json(),
    )

    response = asyncio.run(
        answer_and_verify_async(
            "how many ZA-2115 do we have?",
            client=client,
            serving_url=CLOSED_PORT_URL,
            db_path=db_path,
            escalate=False,
        )
    )

    assert response.grounding_tier == UNGROUNDED
    assert response.claims == ()
    assert "4 units on hand" not in response.text


def test_the_pipeline_passes_section_1s_configuration_through_untouched(db_path):
    """The pipeline is plumbing, not a second place where the answerer's request is decided:
    what reaches `tool_runner` is what `answerer.py`'s constants say."""
    from src.agent.answerer import EFFORT, MAX_TOKENS, MODEL, THINKING

    draft = Draft(claims=[], recommendation=None, unanswered=["everything"])
    client = _ReplayedClient(calls=[], final_text=draft.model_dump_json())

    asyncio.run(
        answer_and_verify_async(
            "anything",
            client=client,
            serving_url=CLOSED_PORT_URL,
            db_path=db_path,
            escalate=False,
        )
    )

    assert client.runner_kwargs["model"] == MODEL
    assert client.runner_kwargs["max_tokens"] == MAX_TOKENS
    assert client.runner_kwargs["thinking"] == THINKING
    assert client.runner_kwargs["output_config"]["effort"] == EFFORT
    # The read-only server's tool set, not a second copy of it: Section 5's table gives
    # Agent A every read-only tool, so a tool added there (as #140 added
    # `find_similar_historical_pattern`) must reach the answerer without this test needing
    # an edit to say so -- while a tool appearing here that is *not* on that server, or
    # `place_order` appearing at all, still fails.
    assert [tool.name for tool in client.runner_kwargs["tools"]] == list(READONLY_TOOL_NAMES)
    assert "place_order" not in READONLY_TOOL_NAMES
