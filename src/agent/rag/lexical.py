"""Query-side lexical scoring for the `prognos_docs` search (Issue #175).

One measure, computed in pure Python from the text the vector search already returned:
**containment** -- the fraction of a query's distinct content words that appear in a
candidate chunk. Short text against long text, asymmetric on purpose, exactly as
`docs/agent_design.md` Section 6's escalation trigger measures a claim against the chunk it
cites: a question is short like a claim, and a candidate's `heading_path + text` is long
like a chunk.

**Why this is a separate module rather than more of `retrieval.py`.** Nothing here knows
that Qdrant, an embedding model, or a `RetrievedChunk` exists -- it is string and set
arithmetic over two strings, and it is unit-testable with neither an index nor a network.
`retrieval.py` is the module that owns the collection contract (`chunk_id` reconstruction,
the payload fields, the `source_type` filter); mixing a tokenizer and a stopword list into
it would put two unrelated reasons to change in one file. The split is also what lets the
zero-import boundary below be checked by importing *this* module alone.

**The duplication with `src/agent/critic/escalation.py` is deliberate, and it is the point.**
That module's `content_tokens`/`lexical_overlap` implement the same technique, and importing
them here would create the first `rag` -> `critic` import in the codebase. Section 5's
independence argument runs the other way (the critic must not reach into the answerer's
tools), but the boundary is worth keeping symmetric for a plainer reason: the two measures
answer different questions and are free to diverge. The critic's is a **gate** on a claim it
has already accepted a citation for, tuned against `LEXICAL_OVERLAP_FLOOR`; this one is a
**ranking term** among candidates that all came back from the same query, and it will be
re-tuned against retrieval metrics that have nothing to do with that floor. Sharing an
implementation would couple two calibrations that are measured on different evidence.
`tests/test_agent_rag_lexical.py` pins the boundary by importing this module in a clean
interpreter and asserting `src.agent.critic` never enters `sys.modules`.

Zero new dependencies: `re` and set operations, nothing else.
"""
from __future__ import annotations

import re

# Function words carry no evidence about subject matter and appear in essentially every
# 1,200-character chunk, so leaving them in would push every candidate's containment toward
# 1.0 and flatten the ranking term to noise. Deliberately the same short, closed list the
# critic uses rather than a larger one: a stopword list is a tuning knob, and this change
# already introduces two (the candidate count and the weight) that are measured. The
# interrogatives a question opens with -- what, how, why, which, when, where -- are already
# in it, which matters more here than it does on the claim side.
_STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for from by with
    without into over under is are was were be been being it its as not no nor do does did
    has have had can could may might will would should must there their they them we our you
    your he she his her which who whom what when where why how all any both each more most
    other some such only own same so too very s t
    """.split()
)

# A token may contain `.`, `-`, `_` and `%` internally -- `0.059`, `1st_test`, `inner-race`,
# `98.5%` are each one term -- but may not end on one, so a sentence-final full stop or a
# question mark does not become part of the word before it. Retrieval queries are full of
# exactly these terms (`rms_ratio`, `baseline_status`, `/predict`), and splitting them would
# throw away the highest-signal words a question contains.
_TOKEN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._%-]*[A-Za-z0-9%])?")


def content_tokens(text: str) -> tuple[str, ...]:
    """Lower-cased content words, function words removed, in order."""
    return tuple(
        token
        for token in (match.group().lower() for match in _TOKEN.finditer(text))
        if token not in _STOPWORDS
    )


def lexical_overlap(query: str, document: str) -> float:
    """The fraction of the query's distinct content words that appear in the document.

    Containment rather than Jaccard, and the asymmetry is what makes it usable here: chunks
    are bounded at 1,200 characters (`docs/agent_design.md` Section 4) and a question is one
    sentence, so a symmetric measure would score every honest pairing near zero and the
    weight below could never be set anywhere useful.

    A query with no content words at all returns 0.0 rather than 1.0 -- there is nothing to
    contain, and "nothing matched nothing" scoring as a perfect match would hand every
    candidate an identical maximum and silently turn the ranking term off.
    """
    query_terms = set(content_tokens(query))
    if not query_terms:
        return 0.0
    document_terms = set(content_tokens(document))
    return len(query_terms & document_terms) / len(query_terms)


def candidate_text(heading_path: str, text: str) -> str:
    """The long side of the measure, for one candidate chunk.

    Reassembled here rather than read off the payload, because the payload stores the two
    separately: `index.py` embeds `Chunk.embedding_text()` (Section 4's "every chunk's text
    is prefixed with its heading path") but writes `heading_path` and `text` as distinct
    fields. Joining them the same way is what makes this measure read the same string the
    vector score was computed over, instead of the chunk body alone.

    It matters most for the continuation chunks Section 4's ~200-character overlap mints:
    their body often never restates its subject, and the heading path is the only place the
    document's own vocabulary ("Drift detection", "Cold start") appears.
    """
    return f"{heading_path}\n\n{text}" if heading_path else text
