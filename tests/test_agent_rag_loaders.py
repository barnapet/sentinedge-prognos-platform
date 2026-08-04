"""Tier-1 tests (Issue #98, `docs/agent_design.md` Section 8) for the `decision_doc` and
`public_reference` loaders: the launch-corpus file list, and Section 4's hard rule --
"no chunk's text is ever authored for the corpus" -- enforced here as a real assertion, not
just followed by hand. No API key, no network: everything read is a file already tracked in
this repository.
"""
from __future__ import annotations

import json

from src.agent.rag.loaders.decision_doc import (
    EXCLUDED_FILENAMES,
    REPO_ROOT,
    DecisionDocLoader,
    decision_doc_paths,
)
from src.agent.rag.loaders.public_reference import (
    REFERENCES_PATH,
    PublicReferenceLoader,
    load_references,
)
from src.agent.rag.schema import KNOWN_SOURCE_TYPES


# --- decision_doc: launch-corpus file list -----------------------------------------------


def test_decision_doc_paths_include_readme_and_exclude_contributing():
    paths = decision_doc_paths()
    names = [p.name for p in paths]

    assert "README.md" in names
    assert "CONTRIBUTING.md" not in names
    assert EXCLUDED_FILENAMES == frozenset({"CONTRIBUTING.md"})

    # Enumerated at call time, not hard-coded -- matches Section 4's launch-corpus rule
    # exactly: every docs/*.md file except the excluded one, plus README.md.
    real_docs_md = sorted((REPO_ROOT / "docs").glob("*.md"))
    expected_count = len([p for p in real_docs_md if p.name not in EXCLUDED_FILENAMES]) + 1
    assert len(paths) == expected_count


def test_decision_doc_paths_are_all_real_files():
    for path in decision_doc_paths():
        assert path.is_file(), f"{path} does not exist"


# --- decision_doc: the hard verbatim rule -------------------------------------------------


def test_decision_doc_chunks_are_verbatim_in_their_source_file():
    """Section 4: 'every `decision_doc` chunk's text appears verbatim in the file its
    `source_ref` names.' Re-reads each source file independently (not via the loader's own
    internals) so this actually catches a fabricated or paraphrased chunk rather than just
    re-checking the loader agrees with itself.
    """
    loader = DecisionDocLoader()
    file_contents: dict[str, str] = {}
    checked = 0

    for chunk in loader.iter_chunks():
        source_ref = chunk.metadata.source_ref
        if source_ref not in file_contents:
            file_contents[source_ref] = (REPO_ROOT / source_ref).read_text(encoding="utf-8")
        assert chunk.text in file_contents[source_ref], (
            f"chunk {chunk.chunk_id} not found verbatim in {source_ref}"
        )
        checked += 1

    assert checked > 0


def test_decision_doc_chunk_metadata_uses_known_source_type():
    loader = DecisionDocLoader()
    for chunk in loader.iter_chunks():
        assert chunk.metadata.source_type == "decision_doc"
        assert chunk.metadata.source_type in KNOWN_SOURCE_TYPES


def test_decision_doc_chunk_indices_are_contiguous_per_document():
    loader = DecisionDocLoader()
    seen: dict[str, list[int]] = {}
    for chunk in loader.iter_chunks():
        seen.setdefault(chunk.metadata.source_id, []).append(chunk.metadata.chunk_index)
    for source_id, indices in seen.items():
        assert indices == list(range(len(indices))), f"non-contiguous indices for {source_id}"


def test_decision_doc_heading_path_is_prefixed_with_the_filename():
    loader = DecisionDocLoader()
    for chunk in loader.iter_chunks():
        expected_prefix = chunk.metadata.source_id.rsplit("/", 1)[-1]
        assert chunk.metadata.heading_path.startswith(expected_prefix)


def test_a_known_real_heading_path_and_its_table_survive_intact():
    """Integration-flavored sanity check against one real, specific document: the `3b`
    heading path from docs/agent_design.md Section 4's own worked example, and its
    accompanying table, both come through the real pipeline intact.
    """
    loader = DecisionDocLoader()
    chunks = [
        c
        for c in loader.iter_chunks()
        if c.metadata.source_id == "docs/model_training_decision.md"
    ]
    matching = [c for c in chunks if "3b. Failure two" in c.metadata.heading_path]
    assert matching, "expected at least one chunk under the 3b heading"

    table_chunks = [c for c in matching if "unreachable rows" in c.text]
    assert table_chunks, "expected the 3b table to survive in one of its chunks"
    # The table's rows must all land in the same chunk -- rule 3, never split a table.
    assert "**17 / 17**" in table_chunks[0].text
    assert "0 / 67" in table_chunks[0].text


# --- public_reference: launch corpus + the same verbatim rule -----------------------------


def test_public_references_file_has_exactly_the_four_launch_entries():
    entries = load_references()
    ids = {entry["id"] for entry in entries}
    assert ids == {
        "qiu_lee_lin_yu_2006",
        "ims_bearing_data_readme",
        "iso_15243_2017",
        "iso_20816_1_2016",
    }
    for entry in entries:
        assert entry["url"].startswith("https://")
        assert entry["treatment"] in {"citation_only", "citation_plus_scope", "full_text"}


def test_public_reference_paywalled_iso_entries_carry_no_body_text():
    """Section 4: paywalled sources are indexed as citation only, never body text or 
    scope text Guarded here so a future edit can't silently smuggle body/scope text
    into these two entries without a test noticing.
    """
    entries = {entry["id"]: entry for entry in load_references()}
    for source_id in ("iso_15243_2017", "iso_20816_1_2016"):
        assert entries[source_id]["treatment"] == "citation_only"
        # The bibliography's own Bibliography/Annex material (deep body content) must not
        # be present -- only Clause 1 ("Scope") text is indexed.
        assert "Annex" not in entries[source_id]["text"]


def test_public_reference_chunks_are_verbatim_in_the_committed_references_file():
    """The same hard rule as decision_doc chunks, applied to the other loader: a chunk's
    text must be a substring of its own entry's committed `text` field -- re-read
    independently from the JSON file, not via the loader's internals.
    """
    raw_entries = {
        entry["id"]: entry["text"]
        for entry in json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))["references"]
    }
    loader = PublicReferenceLoader()
    checked = 0
    for chunk in loader.iter_chunks():
        source_id = chunk.metadata.source_id
        assert chunk.text in raw_entries[source_id], (
            f"chunk {chunk.chunk_id} not found verbatim in references entry {source_id!r}"
        )
        checked += 1
    assert checked > 0


def test_public_reference_chunk_metadata_uses_known_source_type():
    loader = PublicReferenceLoader()
    for chunk in loader.iter_chunks():
        assert chunk.metadata.source_type == "public_reference"
        assert chunk.metadata.source_type in KNOWN_SOURCE_TYPES


def test_public_reference_source_ref_is_the_entrys_url_not_a_repo_path():
    entries = {entry["id"]: entry for entry in load_references()}
    loader = PublicReferenceLoader()
    for chunk in loader.iter_chunks():
        assert chunk.metadata.source_ref == entries[chunk.metadata.source_id]["url"]
