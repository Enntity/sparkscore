<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Working in this repository

SparkScore is a Jev-style typed decision server (state plus typed questions in,
one probability distribution per question out) on one DGX Spark with the Atlas
engine. What counts is decision quality and latency.

- Keep it simple: one entry point (`start.sh`), one install directory, one
  results tree. Prefer removing code to adding it.
- Engine changes belong in Atlas, never here. Propose them upstream
  (Atlas-Inf/atlas); until they merge, pin the fork branch that carries them.
  Move the pin only with `install/repin.sh`, which records commit and tree.
- A quality, calibration or latency claim needs receipts and `SHA256SUMS`
  under `results/`. Name the harness and version, measure baselines on the
  same Spark, report every repetition and say which options were on. Never
  edit a sealed result; add a new one and update `results/README.md`.
- Every claim about another project names its source and says whether it is
  a vendor claim or an independent measurement.
- Defaults are the measured best. Say what any non-default option costs.
- Keep host names, user names, private addresses and home paths out of the
  repository, except in unedited receipts under `results/*/raw/`.
- New files carry an SPDX header. Credit ideas and code taken from other
  projects (origin, license) in code, docs and NOTICE.
- Before sending a change, run the checks in `.github/workflows/static.yml`.
