"""The one live call through the whole A → B pipeline (Issue #116).

**This is not Section 8's golden set**, which is a separate, later issue, and it is not
evidence that the agent answers well. It exists to show, once, against real infrastructure,
that a question goes in and a verified tiered answer comes out: a real serving process with
real per-bearing state, a real read-only MCP server subprocess, the real inventory database,
a real model call for the answerer, and a real escalated critic call if Section 6's rule
fires. Everything else about the pipeline is asserted without a model in
`tests/test_agent_pipeline.py`, where it cannot flake.

**Running this is also how the tier-1 fixture gets its model half.**
`tests/fixtures/answerer_turn.json` currently carries recorded-for-real `tool_payloads` and a
draft synthesized from them, because no credentials were available when it was written.
`python -m tests.fixtures.record_answerer_turn --with-model` re-records both halves from one
real turn and flips the fixture's `draft_source` marker; nothing else changes.

**Skips cleanly without an API key**, following #113/#115's pattern: the reason string says
what is missing and how to supply it, so a skip in a CI log is self-explaining. A skip is not
a pass, and a run that skipped this should say so.

Qdrant is not required. Without it `search_documentation` returns a real failed payload and
the answer degrades — which is a legitimate outcome for this test to observe, and is why the
assertions below are about the *tier being valid and honest*, never about it being
`grounded`.
"""
from __future__ import annotations

import asyncio
import os
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


def _has_anthropic_credentials() -> bool:
    """Whether the SDK has *something* to authenticate with. Same check as #113/#115: an
    unset `ANTHROPIC_API_KEY` does not by itself mean there are no credentials."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    config_dir = Path(
        os.environ.get("ANTHROPIC_CONFIG_DIR", Path.home() / ".config" / "anthropic")
    )
    return (config_dir / "credentials").is_dir() and any(
        (config_dir / "credentials").glob("*.json")
    )


requires_api_key = pytest.mark.skipif(
    not _has_anthropic_credentials(),
    reason=(
        "requires Anthropic credentials (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an "
        "`ant auth login` profile): this is the single live end-to-end call for Issue #116. "
        "Every other assertion about the pipeline runs without credentials in "
        "tests/test_agent_pipeline.py"
    ),
)


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


@requires_api_key
def test_one_real_question_produces_a_verified_tiered_answer(
    populated_serving_api, tmp_path, capsys
):
    """The two-step form, so the test can hold both halves of the turn and check that the
    released claims trace back to tool results from **this** call.
    `answer_and_verify_async` is exactly these two lines composed."""
    db_path = tmp_path / "inventory.db"
    build_db(db_path)

    turn = asyncio.run(
        answer_turn_async(QUESTION, serving_url=populated_serving_api, db_path=db_path)
    )
    response = asyncio.run(verify_turn_async(turn))

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
        print("\n--- live pipeline run ---")
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


@requires_api_key
def test_the_one_call_entry_point_returns_the_same_shape(
    populated_serving_api, tmp_path, capsys
):
    """`answer_and_verify` is the documented entry point, so it gets its own live call rather
    than being assumed equivalent to the composed form above."""
    db_path = tmp_path / "inventory.db"
    build_db(db_path)

    response = asyncio.run(
        answer_and_verify_async(
            QUESTION, serving_url=populated_serving_api, db_path=db_path
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
