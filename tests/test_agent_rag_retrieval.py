"""Tier-1 tests for query-side retrieval: lexical scoring and hybrid reranking (Issue #175).

No model, no network, no Qdrant, no API key -- `docs/agent_design.md` Section 8 tier 1's hard
requirement. `search()` is exercised against a stub client and a stub embedder, so the one
thing that genuinely needs a live index (that the corpus retrieves what the golden set says it
does) stays where it belongs: `tests/fixtures/measure_hybrid_rerank.py`, run against a real
collection.

Two properties here are boundaries rather than behaviours, and both are asserted rather than
documented: `src/agent/rag/lexical.py` never reaches into `src/agent/critic/escalation.py`
despite implementing the same technique, and `RetrievedChunk.score` still carries the vector
similarity after reranking -- the thresholds in `retrieval_confidence.py` were calibrated on
that scale, and a combined score published in the same field would reinterpret them silently.
"""
from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from src.agent.rag import retrieval
from src.agent.rag.lexical import candidate_text, content_tokens, lexical_overlap
from src.agent.rag.retrieval import (
    DEFAULT_LIMIT,
    INTERNAL_CANDIDATE_LIMIT,
    LEXICAL_WEIGHT,
    RetrievedChunk,
    _normalized,
    rerank,
    search,
)


def _chunk(chunk_id: str, score: float, text: str = "", heading: str = "") -> RetrievedChunk:
    source_id, _, chunk_index = chunk_id.partition("::")
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_type="decision_doc",
        source_id=source_id,
        source_ref=source_id,
        heading_path=heading,
        chunk_index=int(chunk_index),
        text=text,
        score=score,
    )


# --- The lexical measure ---------------------------------------------------------------


def test_content_tokens_lowercases_and_drops_function_words():
    assert content_tokens("What is the Drift check?") == ("drift", "check")


def test_content_tokens_keep_project_vocabulary_whole():
    """`rms_ratio`, `0.059` and `98.5%` are one term each -- they are the highest-signal words
    a question about this corpus contains, and splitting them on `_`/`.`/`%` would throw
    exactly those away while leaving the ordinary English intact."""
    tokens = content_tokens("Is rms_ratio at 0.059 or 98.5% here?")

    assert "rms_ratio" in tokens
    assert "0.059" in tokens
    assert "98.5%" in tokens


def test_a_trailing_question_mark_is_not_part_of_the_word():
    assert content_tokens("kurtosis?") == ("kurtosis",)
    assert content_tokens("in section 4.") == ("section", "4")


def test_overlap_is_containment_of_the_query_not_of_the_document():
    """The asymmetry is the whole reason this is usable at 1,200-character chunks: a
    symmetric measure would score every honest pairing near zero, because a question is one
    sentence and a candidate is a page."""
    query = "drift baseline"
    document = "drift baseline " + " ".join(f"unrelated{index}" for index in range(200))

    assert lexical_overlap(query, document) == 1.0
    assert lexical_overlap(document, query) < 0.05


def test_partial_containment_is_the_fraction_of_distinct_query_terms():
    assert lexical_overlap("kurtosis skewness rms", "kurtosis and rms only") == pytest.approx(
        2 / 3
    )


def test_a_query_of_only_function_words_scores_zero_not_one():
    """"Nothing matched nothing" scoring 1.0 would hand every candidate the same maximum and
    silently turn the ranking term off on exactly the emptiest queries."""
    assert lexical_overlap("what is it", "anything at all") == 0.0


def test_candidate_text_includes_the_heading_path():
    """The heading path is part of what was embedded (`Chunk.embedding_text()`), and for a
    continuation chunk it is often the only place the subject is named at all."""
    assert lexical_overlap("cold start", candidate_text("Cold start", "the body")) == 1.0
    assert lexical_overlap("cold start", candidate_text("", "the body")) == 0.0


# --- Normalization ---------------------------------------------------------------------


def test_normalized_maps_the_pool_onto_the_unit_interval():
    assert _normalized([0.70, 0.75, 0.80]) == [0.0, 0.5, 1.0]


def test_a_flat_pool_normalizes_to_zero_rather_than_one():
    """All-ones would let a signal with no spread among the candidates outvote one that has
    spread; all-zeros lets the other term -- and then the stable sort -- decide."""
    assert _normalized([0.8, 0.8, 0.8]) == [0.0, 0.0, 0.0]
    assert _normalized([0.8]) == [0.0]
    assert _normalized([]) == []


# --- Reranking -------------------------------------------------------------------------


def test_rerank_returns_exactly_the_requested_limit():
    candidates = [_chunk(f"doc.md::{index}", 0.8 - 0.01 * index) for index in range(20)]

    assert len(rerank("a query", candidates, limit=DEFAULT_LIMIT)) == DEFAULT_LIMIT


def test_a_pool_smaller_than_the_limit_returns_what_there_is():
    candidates = [_chunk("doc.md::0", 0.8), _chunk("doc.md::1", 0.7)]

    assert len(rerank("a query", candidates, limit=DEFAULT_LIMIT)) == 2


def test_zero_weight_reproduces_the_vector_ranking_exactly():
    """The escape hatch has to be real: `LEXICAL_WEIGHT = 0.0` must return what `search()`
    returned before Issue #175, chunk for chunk and in order."""
    candidates = [
        _chunk("doc.md::0", 0.80, text="nothing in common"),
        _chunk("doc.md::1", 0.79, text="drift baseline z-score"),
        _chunk("doc.md::2", 0.78, text="also nothing"),
    ]

    ranked = rerank("drift baseline z-score", candidates, limit=3, lexical_weight=0.0)

    assert [chunk.chunk_id for chunk in ranked] == ["doc.md::0", "doc.md::1", "doc.md::2"]


def test_the_lexical_term_can_promote_a_deep_candidate():
    """The mechanism the issue asks for, in its smallest form: a candidate the vector search
    ranked last, carrying every word of the query, reaches the top when the lexical term is
    weighted heavily."""
    candidates = [
        *(_chunk(f"doc.md::{index}", 0.80 - 0.001 * index, text="unrelated") for index in range(19)),
        _chunk("doc.md::19", 0.70, text="drift baseline z-score persistence rule"),
    ]

    ranked = rerank("drift baseline z-score", candidates, limit=1, lexical_weight=0.9)

    assert ranked[0].chunk_id == "doc.md::19"


def test_ties_on_the_combined_score_keep_the_vector_order():
    """Containment over a handful of query words takes few distinct values, so ties are the
    common case rather than an edge case. Resolving them by anything other than "what the
    vector search already thought" would make the ranking depend on iteration order."""
    candidates = [
        _chunk("doc.md::0", 0.80, text="drift"),
        _chunk("doc.md::1", 0.80, text="drift"),
        _chunk("doc.md::2", 0.80, text="drift"),
    ]

    ranked = rerank("drift", candidates, limit=3, lexical_weight=0.5)

    assert [chunk.chunk_id for chunk in ranked] == ["doc.md::0", "doc.md::1", "doc.md::2"]


def test_reranking_never_rewrites_the_score_field():
    """`score` stays the cosine similarity Qdrant returned. `TAU_TOP`/`TAU_SUPPORT` (Issue
    #163's sweep) and Section 8's must-refuse mirror metric both read it as one."""
    candidates = [
        _chunk("doc.md::0", 0.80, text="unrelated"),
        _chunk("doc.md::1", 0.70, text="drift baseline"),
    ]

    ranked = rerank("drift baseline", candidates, limit=2, lexical_weight=0.9)

    assert ranked[0].chunk_id == "doc.md::1"
    assert ranked[0].score == 0.70
    assert ranked[1].score == 0.80


def test_rerank_on_an_empty_pool_returns_nothing():
    assert rerank("a query", [], limit=DEFAULT_LIMIT) == []


# --- `search()`'s wiring ---------------------------------------------------------------


class _StubPoint:
    def __init__(self, index: int, score: float) -> None:
        self.payload = {
            "source_type": "decision_doc",
            "source_id": "doc.md",
            "source_ref": "docs/doc.md",
            "heading_path": "doc.md > Section",
            "chunk_index": index,
            "text": f"chunk {index}",
        }
        self.score = score


class _StubResponse:
    def __init__(self, count: int) -> None:
        self.points = [_StubPoint(index, 0.90 - 0.01 * index) for index in range(count)]


class _StubClient:
    """Records the one `query_points` call `search()` makes, and answers it."""

    calls: list[dict[str, Any]] = []

    def __init__(self, url: str) -> None:
        self.url = url

    def query_points(self, **kwargs: Any) -> _StubResponse:
        _StubClient.calls.append(kwargs)
        return _StubResponse(kwargs["limit"])


class _StubVector:
    def tolist(self) -> list[float]:
        return [0.0] * 384


class _StubEmbedder:
    def embed(self, texts: Any) -> Any:
        return iter([_StubVector()])


@pytest.fixture()
def stub_qdrant(monkeypatch: pytest.MonkeyPatch) -> type[_StubClient]:
    _StubClient.calls = []
    monkeypatch.setattr(retrieval, "QdrantClient", _StubClient)
    monkeypatch.setattr(retrieval, "_get_embedder", lambda: _StubEmbedder())
    return _StubClient


def test_the_internal_query_is_wider_than_what_is_returned(stub_qdrant):
    results = search("a query")

    assert stub_qdrant.calls[0]["limit"] == INTERNAL_CANDIDATE_LIMIT
    assert len(results) == DEFAULT_LIMIT


def test_a_caller_asking_for_more_than_the_pool_still_gets_what_it_asked_for(stub_qdrant):
    """`limit` never means "at most `INTERNAL_CANDIDATE_LIMIT`" -- the pool widens to the
    request instead of the request being quietly capped."""
    results = search("a query", limit=INTERNAL_CANDIDATE_LIMIT + 10)

    assert stub_qdrant.calls[0]["limit"] == INTERNAL_CANDIDATE_LIMIT + 10
    assert len(results) == INTERNAL_CANDIDATE_LIMIT + 10


def test_the_source_type_filter_is_applied_to_the_wider_query(stub_qdrant):
    """Widening the candidate pool must not widen what the filter admits: the filter goes on
    the internal query, so no chunk the caller excluded can be reranked into the answer."""
    search("a query", source_type="public_reference")

    query_filter = stub_qdrant.calls[0]["query_filter"]
    condition = query_filter.must[0]

    assert stub_qdrant.calls[0]["limit"] == INTERNAL_CANDIDATE_LIMIT
    assert condition.key == "source_type"
    assert condition.match.value == "public_reference"


def test_no_filter_is_sent_when_no_source_type_is_asked_for(stub_qdrant):
    search("a query")

    assert stub_qdrant.calls[0]["query_filter"] is None


def test_search_returns_the_unchanged_retrieved_chunk_shape(stub_qdrant):
    """Same dataclass, same fields, same `chunk_id` reconstruction -- Issue #175 changes which
    chunks come back and in what order, and nothing about the contract they come back in."""
    results = search("a query", limit=1)

    assert results[0] == RetrievedChunk(
        chunk_id="doc.md::0",
        source_type="decision_doc",
        source_id="doc.md",
        source_ref="docs/doc.md",
        heading_path="doc.md > Section",
        chunk_index=0,
        text="chunk 0",
        score=pytest.approx(0.90),
    )


def test_the_shipped_weight_is_inside_the_measured_no_loss_plateau():
    """Pins the value against the sweep that chose it. `tests/fixtures/measure_hybrid_rerank.py`
    measured recall@5 8/8 and precision@5 0.425 -- identical to vector-only -- for every weight
    up to and including 0.08, with precision falling from 0.09 upward. A future change that
    raises this constant should have to re-run that sweep and update this bound with it."""
    assert 0.0 <= LEXICAL_WEIGHT <= 0.08


# --- The zero-import boundary ----------------------------------------------------------


def test_the_rag_package_never_imports_the_critic():
    """Issue #175's constraint, asserted the way `tests/test_agent_critic.py` asserts the
    critic's no-tools property: import the modules in a clean interpreter and look at
    `sys.modules`. `lexical.py` reimplements `escalation.py`'s containment measure on purpose,
    and a later "let's not duplicate this" edit is exactly what this test exists to catch."""
    program = (
        "import sys; import src.agent.rag.lexical, src.agent.rag.retrieval; "
        "assert not [name for name in sys.modules if name.startswith('src.agent.critic')], "
        "sorted(name for name in sys.modules if name.startswith('src.agent.critic'))"
    )
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
