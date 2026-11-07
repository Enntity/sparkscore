<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Release replication: from a clean checkout, by `start.sh`

This run checks that the documented install reproduces the published
numbers. Everything was done on one DGX Spark from a fresh clone of this
repository at `8597e49`, using only `install/` and `start.sh`:

1. **Image.** `install/build.sh` built Atlas at the pin
   (`Enntity/atlas` `upstream/systemone-score-20261105` at `b594151da`, the
   reviewed head of the upstream proposal), giving image
   `sparkscore:28d408ac5cbc`.
2. **Weights.** `install/convert.sh` downloaded d3-ultra and the format
   reference at their pinned revisions into an empty directory, converted
   and validated in 15 minutes, and matched every file of
   `install/d3-ultra-nvfp4.sha256` (`raw/convert.txt`).
3. **Serving.** `./start.sh serve` served those fresh weights with the
   default settings (W4A4).

## Results

| Check | This run | Published (`2026-11-07-d3-ultra-w4a4`) |
|---|---|---|
| Weights | byte-identical to the reference build | |
| JevBench public, 3 runs | 210/231 (90.9%) every run, Brier 0.1175 | 210/231, Brier 0.1175 |
| JevBench p50 / p95 | 144–145 ms / 1.07–1.13 s | 146–148 ms / 1.05–1.06 s |
| BF16 oracle, `prefix_bidir` | 47/49, mean KL 0.01475 | 47/49, mean KL 0.01475 |
| Engine, 288 rows | 134.4 ms | 139 ms (at `6e8c06a44`) |
| Engine, 4,128 rows | 1,656 ms in the run, 1,570 ms in the A/B below | 1,579 ms |
| Concurrency, merged vs solo | max \|dp\| 0 at every K | 0 |

The answers are identical to the published run. Latency matches within
run-to-run noise.

**The A/B.** The run's 4,128-row median came out 5% above the published one,
while its minimum did not move. So the image before the review fixes
(`56e49be1c`) and this one were served in turn on the same Spark and weights,
twice each, interleaved (`raw/ab_sweep.txt`). They agree within 1% at every
size: 1,569–1,610 ms against 1,570–1,575 ms at 4,128 rows. The fixes cost no
speed, and the run's high medians were variance.

## Merged vs solo, and the endpoint's refusals

`merge_identity.py` covers five request shapes, causal and `prefix_bidir`,
each run alone and then six times racing the other four:

- **Shapes:** from 13 rows (below the 32-row floor that score passes now
  lift) up to 2,000-token states, which put the attention on 64-row tiles for
  the whole pass.
- **Merging:** 51 of the 60 results came from merged passes.
- **Identity:** every result equals its solo run exactly.
- **Refusals:**

  | Request | Response |
  |---|---|
  | An option id or prefix id outside the vocab | 400 |
  | 257 branches | 400 |
  | Unknown attention mode | 400 |
  | A single branch longer than one pass | 422 |
  | A burst of 80 concurrent requests | 65 answered, 15 refused with 503 |

  The burst refusals come from the backlog cap of 64 queued jobs.

## Receipts

`raw/` holds the fingerprint (`fingerprint.txt`), conversion log, smoke test,
`start.sh status`, oracle, sweep, concurrency, merge-identity output, the
A/B sweep and the three JevBench runs with their own `SHA256SUMS`.
