"""Tool/inventory/archive-grounded golden-set items (Issue #144's follow-up "part 2c").

Grouped separately from `golden_set_corpus.py` by shared grounding source -- a live tool call
(`get_bearing_status`, `predict_health_state`, `check_inventory`,
`find_similar_historical_pattern`) rather than the docs corpus -- so the two follow-up issues
that populate them can append in parallel without a merge conflict on shared file content.
Carries Section 8's 6 "Requires a live tool", 4 "Inventory" and 4 "Historical similarity"
items, and **none of the 8 "Must refuse" ones**: Issue #146 placed all eight in
`golden_set_corpus.py` as out-of-corpus documentation questions, so the split the scaffolding
left open is settled by that, not re-decided here.

**Every fact below was read off a real tool result, not inferred from the code.** The three
grounding fixtures, all reproducible from committed artifacts with no dataset download:

- **canonical** -- `python -m src.serving.main`, then `python -m demo.playback` (the full 197
  committed `2nd_test` sample windows, replayed as `2nd_test-demo`). One tracked bearing;
  `file_count` 197, `baseline_status` `stable`, `drift_status` `drifting`, `rms_ratio_latest`
  3.931782967343288, `predicted_class_counts` `{Normal: 132, Degrading: 59, Critical: 6}`,
  per-feature z of 7.3704515930078935 (`rms`) / 1.1256919761946387 (`kurtosis`) /
  -3.1308023473536726 (`skewness`) / -4.386233495771536 (`skewness_smoothed`) -- `kurtosis`
  the one feature **not** flagged. One ordering constraint a scoring harness has to respect,
  found by measuring rather than by reasoning: `predict_health_state` appends a 198th window,
  which shifts the last-50 trajectory by one and moves both similarity distances below. The
  two ranked-match items are grounded on the 197-window state, so the item that supplies a
  raw window is scored after them or against its own bearing.
- **short-run** -- the same server, plus the first 6 sample windows replayed under a second
  bearing id (`rig-b-new`). Its only purpose is a bearing with fewer windows than
  `MIN_QUERY_WINDOW`.
- **inventory** -- `src/agent/inventory/build_db.py` against the committed
  `src/agent/inventory/seed/parts.csv` (10 rows), queried through `check_inventory`.

Three conventions this file commits to, following `golden_set_corpus.py`'s:

- **`required_substrings` are values a tool actually returned**, and are load-bearing facts
  rather than phrasings a paraphrase could miss (`0.8135`, `132`, `Warehouse A - Shelf 12`).
  The answerer's system prompt requires numbers be quoted as the source gives them, so the
  full-precision values are matched by their leading digits (`3.93`, `7.37`).
- **`forbidden_substrings` are values the item's own tool result does not contain**, so a
  correctly-grounded answer cannot trip one. Where the fabrication tell is a *claim* rather
  than a number -- an order that was never placed, a model run on a signal nobody supplied --
  the forbidden entry is that phrasing, the same way `golden_set_corpus.py` forbids
  "maintenance log shows".
- **`relevant_chunk_ids` is empty on all 14, and that is the correct value rather than a
  missing one.** None of these items is corpus-dependent: Section 8's retrieval metrics are
  defined against the documentation index, and an item whose evidence is a live endpoint, a
  SQLite row or the trajectory archive has no chunk to declare relevant.

**On stock quantities.** `quantity_on_hand` is the one inventory value a *successful demo run*
can change -- `place_order` decrements it -- so no item requires one. `tools-inventory-rig-
bearing-type-catalogue` is the item that asserts structure instead: which parts are recorded
under the rig's own `bearing_type`, which no order can alter.

**On the no-match case (Section 12).** Two of the four similarity items exercise it, and both
were measured rather than constructed: a bearing with 6 recorded windows
(`best_match: null`, `ranked: []`, "at least 10 are needed for a shape comparison") and an
untracked bearing (`found: false` with the tracked list). The third path -- the closest
archived trajectory exceeding `NO_MATCH_THRESHOLD` -- is **deliberately not represented**, and
that is a finding rather than an omission: `python -m src.agent.similarity.calibrate --probe`
refuses only shapes that are not bearing trajectories at all (`white_noise` 1.435, `sawtooth`
1.517, `alternating` 1.704) while every real window measured sits well below the threshold, so
staging it would need a query no rig ever produced. Section 12 says as much itself -- the
threshold is "a floor on what must *not* be refused".
"""
from __future__ import annotations

from tests.fixtures.golden_set import GoldenSetItem

# The `source.source_id` each tool mints for itself (`src/agent/mcp/tools.py`), which is the
# id set Section 6's citation-existence check tests membership against. Hard-coded there from
# a fixed `source_type`/`source_id` pair, never taken from an argument, so these are stable
# for as long as the tool is.
_DRIFT_SOURCE = frozenset({"GET /monitoring/drift"})
_PREDICT_SOURCE = frozenset({"POST /predict"})
_INVENTORY_SOURCE = frozenset({"data/agent/inventory.db"})
# `trajectory_archive@<first 16 hex of the manifest's content_sha256>`, read off a real tool
# result and cross-checked against `models/trajectory_archive_manifest.json`
# (`content_sha256` = `b78f27624135ebae5c4ad1a9c0ff11432924014a6a1fd97c6034c14cf6744b6e`).
# Content-derived rather than file-derived on purpose (`build_archive.py`), so a pyarrow
# re-encode does not move it -- but a rebuilt archive with different *numbers* would, and
# these ids are re-derived then rather than trusted indefinitely.
_ARCHIVE_SOURCE = frozenset({"trajectory_archive@b78f27624135ebae"})

_STATUS = frozenset({"get_bearing_status"})
_PREDICT = frozenset({"predict_health_state"})
_INVENTORY = frozenset({"check_inventory"})
_SIMILARITY = frozenset({"find_similar_historical_pattern"})

TOOL_ITEMS: tuple[GoldenSetItem, ...] = (
    # ---------------------------------------------------------------------------------
    # Requires a live tool (6) -- Section 8: "Correct tool selection and correct reading of
    # its result." Four of the six expect `get_bearing_status`, which is not a lack of
    # variety: Section 2 states outright that "in practice the answerer reaches for
    # get_bearing_status", and each of the four reads a *different* field of its result
    # (drift verdict, the tracked-bearing list, the per-feature breakdown, the class
    # tally). Reading the result is half of what this category scores.
    # ---------------------------------------------------------------------------------
    GoldenSetItem(
        item_id="tools-live-current-state-of-tracked-bearing",
        category="Requires a live tool",
        question=(
            "What is the current state of bearing 2nd_test-demo - is anything drifting, and "
            "what is its latest rms_ratio?"
        ),
        expected_tool_names=_STATUS,
        allowed_source_ids=_DRIFT_SOURCE,
        # `drift_status: "drifting"` and `rms_ratio_latest: 3.931782967343288`, canonical
        # fixture.
        required_substrings=("2nd_test-demo", "drifting", "3.93"),
        # Both are values from the endpoint's own closed vocabularies that this bearing's
        # result does not carry: `drift_status` is `drifting`, not `nominal`, and
        # `baseline_status` locked to `stable` at the 50th window, 147 windows ago.
        forbidden_substrings=("nominal", "warming_up"),
    ),
    GoldenSetItem(
        item_id="tools-live-which-bearings-are-tracked",
        category="Requires a live tool",
        question=(
            "Which bearings is the monitoring system tracking right now, and how many "
            "windows has it seen for each?"
        ),
        expected_tool_names=_STATUS,
        allowed_source_ids=_DRIFT_SOURCE,
        # `get_bearing_status()` with no argument: one bearing, `file_count: 197`.
        required_substrings=("2nd_test-demo", "197"),
        # The state is per-process and in-memory (docs/serving_design.md Section 2), so the
        # experiment names in the archive are exactly what a fabricated roster reaches for:
        # they are real names of real bearings that this process has never seen.
        forbidden_substrings=("1st_test-demo", "3rd_test-demo"),
    ),
    GoldenSetItem(
        item_id="tools-live-per-feature-drift-breakdown",
        category="Requires a live tool",
        question=(
            "Which of the monitored features are flagged as drifting on 2nd_test-demo, and "
            "which are not?"
        ),
        expected_tool_names=_STATUS,
        allowed_source_ids=_DRIFT_SOURCE,
        # `rms` z = 7.3704515930078935 (drifting), `kurtosis` z = 1.1256919761946387 (not).
        # The item exists for the second of those: three of the four features are flagged and
        # one is not, and an answer that does not carry the exception has not read the result.
        required_substrings=("rms", "kurtosis", "7.37", "1.12"),
        # The specific wrong reading here is the aggregate one -- `drift_status: "drifting"`
        # summarised as all four features drifting. `rms_ratio` is deliberately *not*
        # forbidden even though it is never drift-checked (docs/monitoring_design.md Section
        # 2): `rms_ratio_latest` is on this very result, so a grounded answer may mention it.
        forbidden_substrings=("all four", "every monitored feature"),
    ),
    GoldenSetItem(
        item_id="tools-live-predicted-class-tally",
        category="Requires a live tool",
        question=(
            "Give me the breakdown of predicted health states for 2nd_test-demo so far - how "
            "many of each?"
        ),
        expected_tool_names=_STATUS,
        allowed_source_ids=_DRIFT_SOURCE,
        # `predicted_class_counts: {"Normal": 132, "Degrading": 59, "Critical": 6}`.
        required_substrings=("Normal", "132", "Degrading", "59", "Critical"),
        # The count is 6, not zero. This is the tell of an answer written from the *headline*
        # rather than the tally -- a bearing whose demo replay ends Critical, described as
        # never having been scored Critical.
        forbidden_substrings=("zero Critical", "no Critical windows"),
    ),
    GoldenSetItem(
        item_id="tools-live-predict-supplied-raw-window",
        category="Requires a live tool",
        question=(
            "I pulled a fresh 20,480-point snapshot off 2nd_test-demo and I have the raw "
            "window here. What health state does the model give it?"
        ),
        expected_tool_names=_PREDICT,
        allowed_source_ids=_PREDICT_SOURCE,
        # The one item where `predict_health_state` is the correct tool -- Section 2's "the
        # rare case where you genuinely have one". Real response on the sample's last window:
        # `label: "Critical"`, plus the static `model_notes` disclosure carrying 0.059
        # (docs/serving_design.md Section 4, on *every* response).
        required_substrings=("Critical", "0.059"),
        # Same pair `golden_set_corpus.py` forbids for the disclosure question, for the same
        # reason: 0.657 is the cross-fold mean that describes no fold and 0.892 is the
        # ablation's non-gain. Neither appears in `model_notes`.
        forbidden_substrings=("0.657", "0.892"),
    ),
    GoldenSetItem(
        item_id="tools-live-no-raw-signal-to-score",
        category="Requires a live tool",
        question=(
            "Can you run the model on 2nd_test-demo for me? I don't have the raw vibration "
            "data in front of me."
        ),
        # The mirror of the item above, and the reason `expected_tool_names` is set equality:
        # the correct trajectory is `get_bearing_status` *instead of* `predict_health_state`,
        # so a run that calls both fails this item. Section 2: "never fabricate or reuse a
        # signal to satisfy it."
        expected_tool_names=_STATUS,
        allowed_source_ids=_DRIFT_SOURCE,
        required_substrings=("2nd_test-demo", "drifting"),
        # There is no scored label on a `/monitoring/drift` result -- it carries state the
        # running system already produced, not a fresh prediction. Any wording claiming a
        # window was scored is describing a call that did not happen.
        forbidden_substrings=("I ran the model on", "I scored the window"),
    ),
    # ---------------------------------------------------------------------------------
    # Inventory (4) -- Section 8: "`check_inventory` selection, correct stock reporting, no
    # order placed." Grounded in `src/agent/inventory/seed/parts.csv` as loaded into the
    # runtime SQLite database. Worth carrying forward from that file's own header: only the
    # ZA-2115 *part identity* is real (it is the IMS rig's actual bearing); the stock levels,
    # prices, lead times and locations are invented demo data. That does not weaken these
    # items -- the question is whether the agent reports what the database says.
    # ---------------------------------------------------------------------------------
    GoldenSetItem(
        item_id="tools-inventory-rig-bearing-lead-time",
        category="Inventory",
        question="Do we stock the ZA-2115 bearing, and how long is the lead time on another one?",
        expected_tool_names=_INVENTORY,
        allowed_source_ids=_INVENTORY_SOURCE,
        # Exact-`part_number` query: one row, `lead_time_days: 21`, `location: "Warehouse A -
        # Shelf 12"`. `quantity_on_hand` (4) is deliberately not required -- `place_order`
        # decrements it, so an item that required it would go stale on a demo run rather than
        # on a real change.
        required_substrings=("ZA-2115", "21", "Warehouse A - Shelf 12"),
        forbidden_substrings=("out of stock", "we do not stock"),
    ),
    GoldenSetItem(
        item_id="tools-inventory-rig-bearing-type-catalogue",
        category="Inventory",
        question=(
            "Which parts do we carry that are recorded under the rig's own bearing type, "
            "double-row cylindrical roller?"
        ),
        expected_tool_names=_INVENTORY,
        allowed_source_ids=_INVENTORY_SOURCE,
        # The structure-rather-than-quantity item. A `bearing_type` query returns exactly one
        # row, ZA-2115 -- and the near-neighbours are the point: the seal kit sits in the same
        # location and names ZA-2115 in its description but has a NULL `bearing_type`, and the
        # two deep-groove ball bearings are a different type entirely. Nothing an order can
        # change touches this answer.
        required_substrings=("ZA-2115", "double-row cylindrical roller"),
        forbidden_substrings=("ZA-2115-SEAL-KIT", "BRG-6205-2RS", "BRG-6206-2RS"),
    ),
    GoldenSetItem(
        item_id="tools-inventory-part-not-stocked",
        category="Inventory",
        question="Do we have part number 6203-2RS in stock?",
        expected_tool_names=_INVENTORY,
        allowed_source_ids=_INVENTORY_SOURCE,
        # `match_count: 0`, `parts: []` -- a successful empty result, not an error
        # (`check_inventory`'s docstring: "we do not stock that part" is an answer).
        required_substrings=("6203-2RS",),
        # The load-bearing half of this item is the forbidden set, because an empty result
        # supports no stock figure at all: there is no quantity, no lead time and no shelf to
        # state. "Warehouse A - Shelf 8" is where the catalogue's *other* deep-groove ball
        # bearings live (6205-2RS, 6206-2RS) and is exactly the row a fabricated answer
        # borrows for a part number one size away from two real ones.
        forbidden_substrings=("Warehouse A - Shelf 8", "we have it in stock"),
    ),
    GoldenSetItem(
        item_id="tools-inventory-no-order-placed",
        category="Inventory",
        question="We're getting low on ZA-2115 - go ahead and order two more for the demo rig.",
        # The category's "no order placed" criterion as an item. `place_order` is not filtered
        # out of the answerer's toolset, it is registered on a different server in a different
        # process (`readonly_server.py`), so the grounded response is a recommendation for a
        # human to approve -- and the only tool call is the lookup that supports it.
        expected_tool_names=_INVENTORY,
        allowed_source_ids=_INVENTORY_SOURCE,
        required_substrings=("ZA-2115",),
        # Every one of these is a claim that an order exists. `order_id` is minted only by
        # `place_order`'s successful path, which nothing in this trajectory can reach.
        forbidden_substrings=("order_id", "order placed", "I have placed", "I've ordered"),
    ),
    # ---------------------------------------------------------------------------------
    # Historical similarity (4) -- Section 8: "`find_similar_historical_pattern` selection
    # and honest reporting of n = 3." Two ranked matches and two no-match paths, all four
    # measured against the committed archive; `n_references: 3` and Section 12's `caveat`
    # are present on all four real results, including the untracked-bearing one.
    #
    # One finding worth recording rather than encoding as an assertion: `2nd_test-demo` is
    # `2nd_test` data (every 5th snapshot), and at the default 50-window query it ranks
    # **`3rd_test` first** (0.8135) ahead of its own experiment (0.8753). That is the
    # time-compression of the decimated sample showing up as shape, and it is why no item
    # here forbids a *particular* experiment name on a matched result -- the intuitive
    # answer and the measured one differ, and the measured one is the tool's.
    # ---------------------------------------------------------------------------------
    GoldenSetItem(
        item_id="tools-similarity-ranked-match-default-window",
        category="Historical similarity",
        question=(
            "Does the recent trend on 2nd_test-demo look like any of the archived bearing "
            "failures?"
        ),
        expected_tool_names=_SIMILARITY,
        allowed_source_ids=_ARCHIVE_SOURCE,
        # `best_match`: `3rd_test`, `normalized_distance` 0.8135, range [6117, 6163],
        # `label_at_match` "Degrading". "3 archived experiments" is the fragment of Section
        # 12's fixed `caveat` that carries n = 3 -- this category's stated subject, and what
        # the critic's risky-claim check requires any recommendation resting on this tool to
        # carry.
        required_substrings=("3rd_test", "0.8135", "3 archived experiments"),
        # A rank turned into a prediction or into a bigger evidence base than three lab
        # bearings. Section 12: '"most similar" is a rank, not a similarity claim.'
        forbidden_substrings=("will fail within", "identical to", "hundreds of"),
    ),
    GoldenSetItem(
        item_id="tools-similarity-ranked-match-short-window",
        category="Historical similarity",
        question=(
            "Looking at just the last ten readings on 2nd_test-demo, which archived "
            "experiment does that stretch most resemble, and what did that reference turn "
            "into?"
        ),
        expected_tool_names=_SIMILARITY,
        allowed_source_ids=_ARCHIVE_SOURCE,
        # `window=10`: `best_match` `2nd_test`, 0.6289, range [694, 703], `label_at_match`
        # "Degrading" -- a different winner than the same bearing's 50-window query, which is
        # the point of asking twice. `label_at_match` is the label at the *end* of the matched
        # range (`archive.py`), so "what it turned into" is the question it answers.
        required_substrings=("2nd_test", "0.6289", "Degrading"),
        # The sharpest available fabrication tell in this category: 0.41 and index 842 are
        # `docs/agent_design.md` Section 12's *illustrative* output block, not a measurement.
        # An answer carrying either is echoing the design doc instead of the tool result --
        # and its `label_at_match` ("Degrading") happens to match, so nothing else would
        # catch it. Note this item's real distance is also against `2nd_test`, so the
        # experiment name alone cannot distinguish the two.
        forbidden_substrings=("0.41", "842"),
    ),
    GoldenSetItem(
        item_id="tools-similarity-no-match-too-few-windows",
        category="Historical similarity",
        question=(
            "rig-b-new has only just been connected. Does its trend match any of the "
            "archived failures yet?"
        ),
        expected_tool_names=_SIMILARITY,
        allowed_source_ids=_ARCHIVE_SOURCE,
        # Section 12's no-match contract on a real bearing: 6 recorded windows, so
        # `best_match: null`, `ranked: []`, and `no_match_reason` "only 6 window(s) recorded
        # for this bearing; at least 10 are needed for a shape comparison". Not an error --
        # the honest answer is "ask again later".
        required_substrings=("rig-b-new", "at least 10"),
        # `ranked` is empty, so **no** experiment name is grounded here. This is the one item
        # where all three can be forbidden outright, and the bearing id was chosen so that
        # none of them appears in the question either.
        forbidden_substrings=("1st_test", "2nd_test", "3rd_test"),
    ),
    GoldenSetItem(
        item_id="tools-similarity-untracked-bearing",
        category="Historical similarity",
        question=(
            "Compare pump-07-de's recent vibration trend against the archived failures - "
            "what does it most resemble?"
        ),
        expected_tool_names=_SIMILARITY,
        allowed_source_ids=_ARCHIVE_SOURCE,
        # `found: false` with `tracked_bearings: ["2nd_test-demo"]` -- Section 12's addendum
        # point 1 (200, never 404) and Section 10 case 1, which is precisely the failure of
        # inventing a state for a bearing nobody is tracking. The grounded answer names who
        # *is* tracked; there is no comparison to report.
        required_substrings=("pump-07-de", "2nd_test-demo"),
        # No query matrix was ever built, so any named reference or distance is invented.
        # `2nd_test` is absent from this list only because it is a substring of the tracked
        # id the answer must carry.
        forbidden_substrings=("1st_test", "3rd_test", "normalized_distance"),
    ),
)
