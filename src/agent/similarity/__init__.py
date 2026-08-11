"""Temporal-similarity search over the committed trajectory archive (Issue #140,
`docs/agent_design.md` Section 12).

Three modules, split along the same lines the rest of this repo already uses:

- `dtw` -- the metric. Pure numpy, no I/O, no project data: z-normalization and banded
  subsequence DTW as ordinary functions over arrays. Section 8's tier-1 tests check it
  against hand-computed cases, which is the concrete reason Section 12 gave for writing
  ~40 lines rather than taking `dtaidistance` or `tslearn` as a dependency.
- `archive` -- the reference data: loading `models/trajectory_archive.parquet`, and
  ranking one query against the three archived experiments.
- `build_archive` -- the offline CLI that produces that artifact and its manifest.

`calibrate` sits beside them as the leave-one-out threshold measurement, kept as a runnable
script rather than a notebook cell for the same reason
`src/training/compute_drift_baseline.py` is one: the number it produces is committed, so
how it was produced has to be re-runnable.

Nothing here imports `src/serving/`. The live half of a query -- a bearing's recent
trajectory -- arrives over HTTP through `src/agent/mcp/serving_client.py`, per Section 2's
constraint.
"""
