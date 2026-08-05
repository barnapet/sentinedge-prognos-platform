"""Tier-1 tests for `docs/agent_design.md` Section 10 **case 10 -- corpus hygiene** (Issue #104).

Case 3 requires its injection payload to be "loaded into a test collection and never into the
real corpus." Until Issue #104 that was an intention rather than a gate, and it was false:
Section 10's case 3 row specified its payload by quoting it verbatim, and Section 4 puts every
`docs/*.md` -- including `agent_design.md` itself -- into the launch corpus, so `index.py`
upserted exactly one chunk carrying the literal string into the real `prognos_docs` collection.

This module is the gate. No API key, no network, no Qdrant: the loaders read tracked files and
these are string checks over what they emit.
"""
from __future__ import annotations

from pathlib import Path

from src.agent.rag.index import default_loaders
from src.agent.rag.loaders.decision_doc import (
    REPO_ROOT,
    DecisionDocLoader,
    decision_doc_paths,
)
from src.agent.rag.loaders.public_reference import REFERENCES_PATH
from tests.fixtures.adversarial_payloads import CASE_3_RETRIEVED_CONTENT_INJECTION


def _indexed_source_files() -> list[Path]:
    """Every file whose text can reach the index: `decision_doc`'s markdown set plus the
    committed references JSON `public_reference` reads."""
    return [*decision_doc_paths(), REFERENCES_PATH]


def test_case_3_payload_is_absent_from_every_decision_doc_chunk():
    """The exact check Issue #102 ran to find the violation, now asserted rather than
    performed by hand. Before #104 this returned `['docs/agent_design.md::91']`."""
    offenders = [
        chunk.chunk_id
        for chunk in DecisionDocLoader().iter_chunks()
        if CASE_3_RETRIEVED_CONTENT_INJECTION in chunk.text
    ]
    assert offenders == [], (
        f"case 3's literal payload reached the real corpus in {offenders}. Section 10 "
        f"specifies payloads structurally; the literal belongs only in "
        f"tests/fixtures/adversarial_payloads.py."
    )


def test_case_3_payload_is_absent_from_every_chunk_of_every_launch_corpus_loader():
    """`decision_doc` is where the violation was, but the invariant is about the whole
    collection -- `index.py` builds it from every loader `default_loaders()` returns, so the
    sweep follows that list rather than hard-coding one loader."""
    offenders = [
        chunk.chunk_id
        for loader in default_loaders()
        for chunk in loader.iter_chunks()
        if CASE_3_RETRIEVED_CONTENT_INJECTION in chunk.text
    ]
    assert offenders == []


def test_case_3_payload_is_absent_from_the_indexed_source_files_themselves():
    """Belt and braces, and not redundant: `bound_section` can split a section anywhere, so a
    payload straddling a chunk boundary would be absent from every individual chunk's text
    while still being present in the document -- and would still be retrievable in halves. A
    per-chunk substring test cannot see that; the raw file can.
    """
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in _indexed_source_files()
        if CASE_3_RETRIEVED_CONTENT_INJECTION in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_no_indexed_source_file_lives_under_tests():
    """The fixture's containment, asserted rather than assumed: if a loader ever walked
    `tests/`, the payload module would index itself and every assertion above would start
    failing for the right reason. Pinning it here makes that a caught change, not a surprise.
    """
    tests_dir = REPO_ROOT / "tests"
    for path in _indexed_source_files():
        assert not path.resolve().is_relative_to(tests_dir.resolve()), (
            f"{path} is under tests/ and reachable by a RAG loader"
        )
