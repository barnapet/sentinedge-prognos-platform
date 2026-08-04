"""Tier-1 tests (Issue #98, `docs/agent_design.md` Section 8) for `src/agent/rag/chunking.py`
and `src/agent/rag/schema.py`: the chunker's boundary rules and heading-path construction.
No API key, no network -- pure functions over in-memory strings.
"""
from __future__ import annotations

import pytest

from src.agent.rag.chunking import (
    CHUNK_CHAR_BOUND,
    CHUNK_OVERLAP_CHARS,
    bound_section,
    build_heading_path,
    chunk_document,
    split_into_sections,
)
from src.agent.rag.schema import Chunk, ChunkMetadata, KNOWN_SOURCE_TYPES


# --- split_into_sections: rule 1, heading-first split -----------------------------------


def test_split_into_sections_builds_nested_heading_paths():
    text = (
        "# Title\n"
        "intro text\n"
        "## 1. Section one\n"
        "section one body\n"
        "### 1a. Subsection\n"
        "subsection body\n"
        "## 2. Section two\n"
        "section two body\n"
    )
    sections = split_into_sections(text)
    paths = [s.heading_path_parts for s in sections]
    # The level-1 title opens a new section but is never itself a path component --
    # Section 4's own example path starts at a level-2 heading.
    assert paths == [
        (),
        ("1. Section one",),
        ("1. Section one", "1a. Subsection"),
        ("2. Section two",),
    ]
    assert "subsection body" in sections[2].text
    assert "section one body" in sections[1].text
    # A heading's direct body excludes its descendant's text.
    assert "subsection body" not in sections[1].text


def test_split_into_sections_ignores_headings_inside_fenced_code_blocks():
    text = (
        "## Real heading\n"
        "before fence\n"
        "```python\n"
        "# not a heading, a python comment\n"
        "## also not a heading\n"
        "```\n"
        "after fence\n"
    )
    sections = split_into_sections(text)
    assert [s.heading_path_parts for s in sections] == [("Real heading",)]
    assert "# not a heading, a python comment" in sections[0].text
    assert "## also not a heading" in sections[0].text


def test_split_into_sections_empty_section_between_headings_is_dropped():
    """A heading immediately followed by a deeper heading, with no text between them,
    contributes no chunk of its own -- only sections with real body text are emitted."""
    text = "## 1. Parent\n### 1a. Child\nchild body\n"
    sections = split_into_sections(text)
    assert [s.heading_path_parts for s in sections] == [("1. Parent", "1a. Child")]


def test_split_into_sections_preamble_before_any_heading():
    text = "no heading yet\njust prose\n## 1. First heading\nbody\n"
    sections = split_into_sections(text)
    assert sections[0].heading_path_parts == ()
    assert "no heading yet" in sections[0].text


# --- build_heading_path ------------------------------------------------------------------


def test_build_heading_path_matches_section_4_format():
    path = build_heading_path(
        "model_training_decision.md",
        ("3. The `1st_test` fold: two stacked failures, not one", "3b. Failure two"),
    )
    assert path == (
        "model_training_decision.md > "
        "3. The `1st_test` fold: two stacked failures, not one > 3b. Failure two"
    )


def test_build_heading_path_with_no_ancestors_is_just_the_doc_label():
    assert build_heading_path("README.md", ()) == "README.md"


# --- bound_section: rules 2 and 3, char bound + overlap + never-split-a-table -----------


def test_bound_section_never_splits_a_short_section():
    """Rule 3's second half: a section shorter than the bound comes back as one chunk."""
    text = "a short paragraph, well under the character bound.\n"
    chunks = bound_section(text)
    assert chunks == [text]


def test_bound_section_splits_long_text_and_overlaps_adjacent_chunks():
    # Distinct, numbered lines (not repeated identical text) so overlap can be verified
    # precisely -- a specific line either does or doesn't reappear at the next chunk's head.
    lines = [f"line{i:03d} " + ("y" * 80) + "\n" for i in range(40)]
    text = "".join(lines)  # ~3560 chars, well past CHUNK_CHAR_BOUND
    chunks = bound_section(text)

    assert len(chunks) > 1
    for c in chunks:
        # "~1,200 characters": allow slack for the one line that pushes a chunk over the
        # bound (the check fires *before* adding a segment, per rule 3's table exception
        # reasoning), but a chunk should never balloon far past it for plain text.
        assert len(c) <= CHUNK_CHAR_BOUND + 200

    # Adjacent chunks share overlapping lines: at least one whole line at the tail of
    # chunk N reappears, verbatim, at the head of chunk N+1.
    for first, second in zip(chunks, chunks[1:]):
        first_lines = [l for l in first.splitlines() if l]
        second_lines = [l for l in second.splitlines() if l]
        overlap_lines = set(first_lines[-3:]) & set(second_lines[:3])
        assert overlap_lines, f"no overlap: {first_lines[-3:]!r} vs {second_lines[:3]!r}"


def test_bound_section_never_splits_a_table_even_past_the_bound():
    table = (
        "| Held out | train-fold min | held-out band | unreachable |\n"
        "|---|---|---|---|\n"
        "| `1st_test` | 2.870 | [1.948, 2.869] | **17 / 17** |\n"
        "| `2nd_test` | 1.948 | [2.870, 6.323] | 0 / 23 |\n"
        "| `3rd_test` | 1.948 | [3.090, 7.153] | 0 / 67 |\n"
    )
    # Padding long enough either side that a naive char-bound split would land inside the
    # table if the table weren't treated as atomic.
    padding = ("filler text to push past the bound. " * 40) + "\n"
    text = padding + table + padding

    chunks = bound_section(text, char_bound=200, overlap_chars=20)

    matches = [c for c in chunks if table in c]
    assert len(matches) == 1, "the table must appear whole in exactly one chunk"
    # And no chunk contains only a fragment of the table (a header without its rows, etc.).
    for c in chunks:
        if table not in c:
            assert "| Held out |" not in c


def test_bound_section_table_alone_can_exceed_the_bound():
    table_rows = ["| a | b |", "|---|---|"] + [f"| {i} | {'z' * 50} |" for i in range(30)]
    table = "\n".join(table_rows) + "\n"
    assert len(table) > CHUNK_CHAR_BOUND

    chunks = bound_section(table)
    assert len(chunks) == 1
    assert chunks[0] == table


# --- chunk_document: end-to-end composition ----------------------------------------------


def test_chunk_document_preserves_order_and_is_verbatim():
    text = (
        "# Title\n"
        "preamble\n"
        "## 1. First\n"
        "first body\n"
        "## 2. Second\n"
        "second body\n"
    )
    raw_chunks = chunk_document(text)
    assert [rc.heading_path_parts for rc in raw_chunks] == [(), ("1. First",), ("2. Second",)]
    for rc in raw_chunks:
        assert rc.text in text


# --- schema: ChunkMetadata / Chunk --------------------------------------------------------


def test_chunk_metadata_rejects_unknown_source_type():
    with pytest.raises(ValueError):
        ChunkMetadata(
            source_type="not_a_real_type",
            source_id="x",
            source_ref="x",
            heading_path="x",
            chunk_index=0,
            indexed_at="2026-01-01T00:00:00+00:00",
        )


def test_chunk_metadata_accepts_known_source_types():
    for source_type in KNOWN_SOURCE_TYPES:
        ChunkMetadata(
            source_type=source_type,
            source_id="x",
            source_ref="x",
            heading_path="x",
            chunk_index=0,
            indexed_at="2026-01-01T00:00:00+00:00",
        )


def test_chunk_id_and_embedding_text():
    metadata = ChunkMetadata(
        source_type="decision_doc",
        source_id="docs/example.md",
        source_ref="docs/example.md",
        heading_path="example.md > 1. Foo",
        chunk_index=3,
        indexed_at="2026-01-01T00:00:00+00:00",
    )
    chunk = Chunk(text="the body", metadata=metadata)

    assert chunk.chunk_id == "docs/example.md::3"
    assert chunk.embedding_text() == "example.md > 1. Foo\n\nthe body"
    # The heading-path prefix lives only in `embedding_text()`, never in `.text` -- so the
    # verbatim-in-source-file check (tests/test_agent_rag_loaders.py) can compare `.text`
    # directly against a file's contents with a plain substring test.
    assert chunk.text == "the body"
