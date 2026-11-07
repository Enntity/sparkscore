<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Contributing

Issues and pull requests are welcome. Please read [AGENTS.md](AGENTS.md)
first: it is short and applies to people too.

- **Harness, install, demo, docs:** a pull request here. Run the checks in
  `.github/workflows/static.yml` (`python3 -m unittest discover -s tests`).
- **Engine (kernels, the score path):** these live in Atlas. Open them
  against [Atlas-Inf/atlas](https://github.com/Atlas-Inf/atlas), which has its
  own CLA. Once merged, SparkScore moves its pin with `install/repin.sh`.
- **Results:** a new dated directory under `results/` with raw receipts,
  `SHA256SUMS` and a `RESULT.md` that names the build, settings and
  hardware. Never edit an existing one.

By contributing you agree that your contribution is licensed under
AGPL-3.0-only.
