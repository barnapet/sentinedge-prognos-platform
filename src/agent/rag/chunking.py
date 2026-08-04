"""Heading-aware chunking (Issue #98, `docs/agent_design.md` Section 4's three rules, in
priority order):

1. Split on markdown headings first -- `split_into_sections`.
2. Bound each chunk at ~1,200 characters with ~200 of overlap -- `bound_section`.
3. Never split a markdown table, and never split a section shorter than the bound --
   enforced inside `bound_section` by treating a table block as one atomic, unsplittable
   segment.

Source-agnostic, per Section 4's loader design: this module knows nothing about
`source_type`, `source_id`, or any specific loader. It turns one markdown document's text
into an ordered list of `RawChunk`s (heading-path parts + verbatim text); loaders attach the
source-specific metadata around that output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, NamedTuple

CHUNK_CHAR_BOUND = 1200
CHUNK_OVERLAP_CHARS = 200

# A heading line: 1-6 `#` at column 0, then whitespace, then the heading text. Indented
# code (4-space style) can never match this -- its `#` is not at column 0 -- so only fenced
# code blocks (checked separately, below) need explicit handling.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_FENCE_RE = re.compile(r"^\s*```")
# A markdown table's separator row: only `|`, `-`, `:`, and whitespace, with at least one
# run of two or more `-` -- e.g. `|---|---|---|` or `| :--- | ---: |`. This, not "a line
# containing `|`", is what actually distinguishes a table's second row from prose that
# happens to mention a pipe character.
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


@dataclass(frozen=True)
class HeadingSection:
    """One heading's direct body text (Section 4: "a `###` subsection is usually exactly
    one complete argument") -- everything after its heading line up to the next heading
    line of any level, not including any descendant section's own text.
    """

    heading_path_parts: tuple[str, ...]  # ancestor heading texts; the level-1 title is excluded
    text: str  # verbatim, original whitespace/newlines preserved


class RawChunk(NamedTuple):
    heading_path_parts: tuple[str, ...]
    text: str  # verbatim substring (or overlap-joined substrings) of the section's text


def build_heading_path(doc_label: str, heading_path_parts: Iterable[str]) -> str:
    """Section 4's `model_training_decision.md > 3. ... > 3b. ...` format."""
    return " > ".join([doc_label, *heading_path_parts])


def split_into_sections(text: str) -> list[HeadingSection]:
    """Split on every markdown heading line (any level), ignoring headings inside fenced
    code blocks.

    Level-1 (`#`) headings are document titles, not path components -- Section 4's own
    example path starts at a level-2 heading -- so they open a new section but are not
    pushed onto the heading-path stack; a document normally has exactly one, and the loader
    supplies the document's own label as the first path component instead (`build_heading_path`).
    """
    lines = text.splitlines(keepends=True)
    sections: list[HeadingSection] = []
    stack: list[tuple[int, str]] = []  # (level, heading text), ancestors only
    current_lines: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "".join(current_lines)
        if body.strip():
            sections.append(HeadingSection(tuple(h for _, h in stack), body))

    for line in lines:
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            current_lines.append(line)
            continue
        heading_match = None if in_fence else _HEADING_RE.match(line)
        if heading_match:
            flush()
            current_lines = []
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            if level > 1:
                stack.append((level, heading_text))
        else:
            current_lines.append(line)
    flush()
    return sections


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped != "" and "|" in stripped


def _is_table_segment(segment: str) -> bool:
    seg_lines = segment.splitlines(keepends=True)
    return len(seg_lines) >= 2 and _TABLE_SEPARATOR_RE.match(seg_lines[1].strip()) is not None


def _find_atomic_segments(lines: list[str]) -> list[str]:
    """Group `lines` into atomic, unsplittable segments: a contiguous markdown table
    (header row + separator row + data rows) is one segment; every other line is its own
    segment. A table is recognised by a `|`-bearing line immediately followed by a
    separator row -- the actual GFM signal for "this is a table", not just any line
    containing `|`.
    """
    segments: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        starts_table = (
            i + 1 < n
            and _is_table_row(lines[i])
            and "|" in lines[i + 1]
            and _TABLE_SEPARATOR_RE.match(lines[i + 1].strip()) is not None
        )
        if starts_table:
            j = i + 2
            while j < n and _is_table_row(lines[j]):
                j += 1
            segments.append("".join(lines[i:j]))
            i = j
        else:
            segments.append(lines[i])
            i += 1
    return segments


def bound_section(
    text: str,
    char_bound: int = CHUNK_CHAR_BOUND,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Rules 2 and 3: bound at ~`char_bound` characters with ~`overlap_chars` of overlap
    between adjacent chunks of the same section, never splitting a table and never
    splitting a section shorter than the bound.

    A section shorter than `char_bound` never trips the overflow check below, so it comes
    back as a single chunk -- rule 3's second half falls out of the packing logic rather
    than needing a separate short-circuit. A table wider than `char_bound` is packed in
    whole (the overflow check only fires *before* adding a segment, never mid-segment), so
    it ends up alone in its own oversized chunk -- rule 3's first half.
    """
    lines = text.splitlines(keepends=True)
    segments = _find_atomic_segments(lines)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for segment in segments:
        if current and current_len + len(segment) > char_bound:
            chunks.append("".join(current))
            # Overlap carries forward the trailing *non-table* segments of the
            # just-finalized chunk, up to ~overlap_chars. A table is atomic in this
            # direction too: duplicating a whole table into the next chunk just to satisfy
            # an overlap target would defeat the purpose of bounding at all, so overlap
            # stops (empty) the moment a table segment is reached walking backwards.
            overlap: list[str] = []
            overlap_len = 0
            for seg in reversed(current):
                if _is_table_segment(seg) or overlap_len >= overlap_chars:
                    break
                overlap.insert(0, seg)
                overlap_len += len(seg)
            current, current_len = overlap, overlap_len
        current.append(segment)
        current_len += len(segment)

    if current:
        chunks.append("".join(current))

    return [c for c in chunks if c.strip()]


def chunk_document(text: str) -> list[RawChunk]:
    """Sections -> bounded chunks, in document order.

    Loaders own the document-level label (filename) and the per-chunk metadata
    (`source_type`, `source_id`, ...); this function only ever sees raw text and the
    heading structure inside it.
    """
    raw_chunks: list[RawChunk] = []
    for section in split_into_sections(text):
        for piece in bound_section(section.text):
            raw_chunks.append(RawChunk(section.heading_path_parts, piece))
    return raw_chunks
