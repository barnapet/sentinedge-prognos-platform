"""Record one real Answerer turn to `tests/fixtures/answerer_turn.json` (Issue #116).

    python -m tests.fixtures.record_answerer_turn --url http://localhost:8000

The pipeline's tier-1 tests are supposed to run against a shape that really came out of the
Answerer, not a hand-typed approximation of one, and this is what produces that shape. It has
two modes, and **which one ran is recorded in the fixture itself** so a reader never has to
guess how real the file in front of them is:

- **`--with-model` (needs Anthropic credentials)** — a real `answer_turn()` call: the real
  model chooses the tools, the real read-only MCP server subprocess serves them, and both
  halves of the fixture (`draft` and `tool_payloads`) are recorded from that one turn.
- **default (no credentials needed)** — the same real MCP server subprocess and the same real
  `ToolResultRecorder`, with this script calling the tools directly instead of a model
  choosing them. `tool_payloads` is then just as real as in the first mode; `draft` is
  synthesized *from those payloads* (its claims cite ids read out of the recording, never
  invented) and marked `"draft_source": "synthesized_from_recorded_payloads"`.

The second mode exists because the model half cannot be recorded without a key, and the tool
half — which is what `TurnEvidence.from_tool_payloads` actually consumes, and therefore what
this issue's plumbing is about — can be. Re-running with `--with-model` overwrites the file
with a fully-real turn and flips the marker; nothing else has to change.

Real infrastructure is required either way: a serving process with a tracked bearing (the
`--url`), the Qdrant collection from `docker compose --profile agent up`, and the inventory
database. What is recorded is what those really returned.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agent.answerer import (
    Claim,
    Draft,
    ToolResultRecorder,
    readonly_tools,
)
from src.agent.inventory.build_db import DB_PATH
from src.agent.mcp.serving_client import DEFAULT_BASE_URL
from src.agent.untrusted import UntrustedEnvelope

FIXTURE_PATH = Path(__file__).resolve().parent / "answerer_turn.json"

BEARING_ID = "2nd_test-demo"
QUESTION = (
    f"What is the current status of bearing {BEARING_ID}, and what does this project's "
    "own documentation say about how far its predictions can be trusted?"
)

# The calls a real turn on this question makes: the bearing's live state, the documentation
# behind the caveat, and the part that would be ordered if it came to that.
DIRECT_CALLS = (
    ("get_bearing_status", {"bearing_id": BEARING_ID}),
    ("search_documentation", {"query": "Critical recall on 1st_test held-out fold", "limit": 3}),
    ("check_inventory", {"part_number": "ZA-2115"}),
)


async def _record_tools_directly(serving_url: str, db_path: Path) -> list[dict[str, Any]]:
    """Drive the real tools over the real stdio transport, recording what comes back.

    Everything here is the production path except which tools get called: the same
    `readonly_tools` context manager the answerer uses, the same enveloping wrapper, the same
    `ToolResultRecorder`.
    """
    recorder = ToolResultRecorder()
    envelope = UntrustedEnvelope()
    async with readonly_tools(envelope, serving_url, db_path, recorder) as (_session, tools):
        by_name = {tool.name: tool for tool in tools}
        for name, arguments in DIRECT_CALLS:
            try:
                await by_name[name].call(arguments)
            except Exception as exc:  # noqa: BLE001 - a failed tool is still a real payload
                print(f"  {name}: returned a failure ({type(exc).__name__})")
            else:
                print(f"  {name}: ok")
    return recorder.payloads


def _synthesized_draft(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """A draft whose citations are read out of the recording rather than invented.

    Deliberately modest: it states what the payloads actually contain, cites the ids those
    payloads actually carried, and leaves one part unanswered. It is a stand-in for the model
    half and is labelled as one in the fixture.
    """
    from src.agent.critic.deterministic import TurnEvidence

    evidence = TurnEvidence.from_tool_payloads(payloads)
    claims: list[Claim] = []

    for item in evidence.items:
        if item.source_id.startswith("GET ") and "file_count" in item.text:
            count = json.loads(item.text)["status"]["file_count"]
            claims.append(
                Claim(
                    text=f"Bearing {BEARING_ID} has been scored on {count} windows so far.",
                    source_ids=[item.source_id],
                )
            )

    # One claim citing every retrieved chunk that is genuinely about the per-fold result,
    # rather than one near-identical claim per chunk — which is the shape a real draft takes
    # and, more to the point, the shape that exercises a multi-citation claim end to end.
    fold_chunks = [
        item.source_id
        for item in evidence.items
        if item.score is not None and "1st_test" in item.text
    ]
    if fold_chunks:
        claims.append(
            Claim(
                text=(
                    "This project's own evaluation reports the baseline's Critical recall "
                    "separately per fold rather than as a cross-fold mean."
                ),
                source_ids=fold_chunks,
            )
        )

    return Draft(
        claims=claims,
        recommendation=None,
        unanswered=["how the bearing's drift will develop over the next week"],
    ).model_dump()


async def _record(with_model: bool, serving_url: str, db_path: Path) -> dict[str, Any]:
    if with_model:
        from src.agent.answerer import answer_turn_async

        print("recording a real turn, model included")
        turn = await answer_turn_async(
            QUESTION, serving_url=serving_url, db_path=db_path
        )
        return {
            "draft_source": "live_model_call",
            "draft": turn.draft.model_dump(),
            "tool_payloads": list(turn.tool_payloads),
        }

    print("recording the tool half only (no Anthropic credentials required)")
    payloads = await _record_tools_directly(serving_url, db_path)
    return {
        "draft_source": "synthesized_from_recorded_payloads",
        "draft": _synthesized_draft(payloads),
        "tool_payloads": payloads,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="a running serving API")
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument(
        "--with-model",
        action="store_true",
        help="make the real model call too (requires Anthropic credentials)",
    )
    args = parser.parse_args(argv)

    recorded = asyncio.run(_record(args.with_model, args.url, args.db_path))
    recorded = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "question": QUESTION,
        **recorded,
    }
    FIXTURE_PATH.write_text(json.dumps(recorded, indent=2) + "\n", encoding="utf-8")

    payloads = recorded["tool_payloads"]
    print(f"wrote {FIXTURE_PATH.relative_to(Path(__file__).resolve().parents[2])}")
    print(f"  tool payloads: {len(payloads)}")
    for payload in payloads:
        source = payload["source"]
        kind = "error" if "error" in payload else "data"
        print(f"    {source['source_type']:<14} {source['source_id']}  ({kind})")


if __name__ == "__main__":
    main()
