"""The one recorded call through the whole A → B pipeline (Issue #116).

**This is not Section 8's golden set**, which is a separate, later issue, and it is not
evidence that the agent answers well. It exists to show, once, against real infrastructure,
that a question goes in and a verified tiered answer comes out: a real serving process with
real per-bearing state, a real read-only MCP server subprocess, the real inventory database,
a real model call for the answerer, and a real escalated critic call if Section 6's rule
fires. Everything else about the pipeline is asserted without a model in
`tests/test_agent_pipeline.py`, where it cannot flake.

**Both model calls are replayed from a committed cassette by default (Issue #122)** — the
answerer's and, when Section 6's escalation rule fires, the critic's, from the one cassette,
since they are one conversation with the API from the transport's point of view. The test now
runs on every PR in ordinary CI with no API key, no network access and no cost, where before
it skipped. Everything else is still real, including the tool calls `tool_runner` makes in
response to the recorded `tool_use` blocks and the deterministic critic, which never called a
model in the first place.

**What a replay stops proving**: that the model would choose these tools and write this draft
today. It still proves that a real turn's evidence set contains every id the released claims
cite — the assertion below is against ids the *harness* recorded from this run's own tool
calls, not against anything in the cassette.

**When to re-record.** Deliberately, not routinely — after a change to either agent's prompt,
schema, tool definitions, or model/request configuration:

    pytest tests/test_agent_pipeline_live.py --record      # real, billed API calls

A replay fails by itself if a prompt or the request configuration moved on since the recording
(`tests/fixtures/cassette.py` hashes both into the fixture), so this is enforced rather than
remembered. `--cassette-mode live` hits the real API without touching the committed fixture.

**Running this in `record` or `live` mode is also how the tier-1 fixture gets its model half.**
`tests/fixtures/answerer_turn.json` carries recorded-for-real `tool_payloads` and a draft
synthesized from them, because no credentials were available when it was written.
`python -m tests.fixtures.record_answerer_turn --with-model` re-records both halves from one
real turn and flips the fixture's `draft_source` marker; nothing else changes.

Qdrant is not required, and **the committed cassettes are deliberately recorded without it**,
matching CI, where the documentation index sits behind an opt-in compose profile that does not
run. Without it `search_documentation` returns a real failed payload and the answer degrades —
a legitimate outcome for this test to observe, and why the assertions below are about the
*tier being valid and honest*, never about it being `grounded`. Recording with Qdrant up would
be the one genuinely unsafe direction: the recorded draft would cite chunk ids that CI's
evidence set cannot contain. The reverse is harmless, so replaying on a machine that does have
Qdrant is fine. Each cassette's `notes` records the probed state.
"""
from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from demo import playback
from src.agent.answerer import answer_turn_async
from src.agent.critic.grounding import GROUNDED, TIERS, UNGROUNDED
from src.agent.inventory.build_db import build_db
from src.agent.pipeline import answer_and_verify_async, evidence_for, verify_turn_async

REPO_ROOT = Path(__file__).resolve().parents[1]

REPLAY_WINDOWS = 60
BEARING_ID = "2nd_test-demo"
QUESTION = (
    f"What is the current status of bearing {BEARING_ID}, and what does this project's own "
    "documentation say about how far its predictions can be trusted?"
)

# One cassette per test: each is an independent conversation, and sharing one would make the
# second test's result depend on the first having run, in order, in the same process.
TWO_STEP_CASSETTE = "pipeline_live__verified_tiered_answer"
ONE_CALL_CASSETTE = "pipeline_live__answer_and_verify"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def populated_serving_api(tmp_path):
    """A real serving process with one bearing replayed past its cold-start boundary — the
    same fixture shape #113's live test uses, for the same reason: the MCP tools speak HTTP
    over a socket, so an in-process transport would not exercise what they actually do."""
    port = _free_port()
    script = (
        "import uvicorn\n"
        "from pathlib import Path\n"
        "from src.serving.api import create_app\n"
        f"app = create_app(lock_path=Path({str(tmp_path / 'pipeline.lock')!r}))\n"
        f"uvicorn.run(app, host='127.0.0.1', port={port}, workers=1, log_level='warning')\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", script], cwd=REPO_ROOT)
    base_url = f"http://127.0.0.1:{port}"
    try:
        playback.wait_for_server(base_url, timeout_s=60)
        playback.main(
            ["--url", base_url, "--interval", "0", "--limit", str(REPLAY_WINDOWS),
             "--bearing-id", BEARING_ID]
        )
        yield base_url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_one_real_question_produces_a_verified_tiered_answer(
    populated_serving_api, tmp_path, capsys, model_cassette
):
    """The two-step form, so the test can hold both halves of the turn and check that the
    released claims trace back to tool results from **this** call.
    `answer_and_verify_async` is exactly these two lines composed.

    The two steps share one `asyncio.run` (they were two before #122) because they now share
    one client, and an httpx connection pool belongs to the loop it was opened on — recording
    across two loops would tear the pool down between the answerer's calls and the critic's.
    Nothing about what is asserted changes; the two calls are still separate and still made in
    that order."""
    db_path = tmp_path / "inventory.db"
    build_db(db_path)

    async def _run(client):
        turn = await answer_turn_async(
            QUESTION, client=client, serving_url=populated_serving_api, db_path=db_path
        )
        return turn, await verify_turn_async(turn, critic_client=client)

    with model_cassette(TWO_STEP_CASSETTE) as client:
        turn, response = asyncio.run(_run(client))

    assert response.grounding_tier in TIERS
    assert response.text.strip()

    evidence = evidence_for(turn)
    assert turn.tool_payloads, "the answerer answered without calling a tool at all"

    # Every released claim's citations came out of this turn's own tool results. This is the
    # property the whole pipeline exists to make true, and it is checked against the ids the
    # harness recorded, not against anything the model said about them.
    for claim in response.claims:
        assert claim.source_ids
        assert set(claim.source_ids) <= evidence.source_ids

    if response.grounding_tier == UNGROUNDED:
        assert response.claims == ()

    with capsys.disabled():
        print("\n--- pipeline run ---")
        print(f"tier: {response.grounding_tier}")
        print(f"tool results recorded: {len(turn.tool_payloads)}")
        for payload in turn.tool_payloads:
            kind = "error" if "error" in payload else "data"
            print(f"  {payload['source']['source_id']} ({kind})")
        print(f"citable ids: {len(evidence.source_ids)}")
        print(f"retrieval: {response.retrieval}")
        print(f"claims released: {len(response.claims)}  dropped: {len(response.dropped)}")
        for dropped in response.dropped:
            print(f"  dropped: {dropped.text}\n    because {'; '.join(dropped.reasons)}")
        print("--- released text ---")
        print(response.text)


def test_the_one_call_entry_point_returns_the_same_shape(
    populated_serving_api, tmp_path, capsys, model_cassette
):
    """`answer_and_verify` is the documented entry point, so it gets its own call rather than
    being assumed equivalent to the composed form above.

    One client for both `client` and `critic_client`. `pipeline.answer_and_verify_async` keeps
    them separate arguments because A and B are separate agents with separate request
    configurations, and its own docstring says passing one object for both changes nothing
    about the boundary — which is the tool surface, not the transport."""
    db_path = tmp_path / "inventory.db"
    build_db(db_path)

    with model_cassette(ONE_CALL_CASSETTE) as client:
        response = asyncio.run(
            answer_and_verify_async(
                QUESTION,
                client=client,
                critic_client=client,
                serving_url=populated_serving_api,
                db_path=db_path,
            )
        )

    assert response.grounding_tier in TIERS
    assert response.text.strip()
    if response.grounding_tier == GROUNDED:
        assert response.claims

    with capsys.disabled():
        print("\n--- answer_and_verify ---")
        print(f"tier: {response.grounding_tier}")
        print(response.text)
