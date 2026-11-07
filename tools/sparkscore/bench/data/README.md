<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Reference data

`d3-bf16-reference.jsonl` holds d3-ultra's own answers, recorded with d3's
BF16 runtime (its `_meta` line names the runtime, versions and file hashes):
the exact prompt token ids, the state-block boundary and the readout logits
for each question. The requests are SparkScore's smoke fixture
(`tests/fixtures/smoke.jsonl`), JevBench public items (MIT,
github.com/fstandhartinger/jevbench) and a few hand-written edge cases.
`../oracle_check.py` compares Atlas against it.
