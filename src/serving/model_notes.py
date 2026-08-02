"""The static `model_notes` disclosure (Issue #84), sourced from `docs/serving_design.md`.

`docs/serving_design.md` Section 4 decided this text and *why* it is unconditional -- no
detector exists that could tell a request apart as more or less likely to hit the known
`1st_test`-shaped failure mode, so the same disclosure goes on every response, not just
ones the server "suspects." This module exists only to hold that one constant so it has a
single, importable source of truth: `src/serving/api.py` attaches it to every `/predict`
response, and `tests/test_api.py` parses Section 4's own code block out of the markdown
file and asserts this constant matches it byte-for-byte -- so the two cannot silently drift
apart the way a hand-copied string could.
"""
from __future__ import annotations

MODEL_NOTES = (
    "Trained on all 3 dataset experiments (1st_test/2nd_test/3rd_test) pooled. LOEO "
    "evaluation found this model class does not reliably detect the Critical health state "
    "on impulsive, inner-race degradation signatures resembling the 1st_test bearing "
    "(Critical recall 0.059 when that experiment was held out) — see "
    "docs/model_training_decision.md. Reliable on the two outer-race, amplitude-driven "
    "failure modes evaluated (Critical recall 0.913 / 1.000)."
)
