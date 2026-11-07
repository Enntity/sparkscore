<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Results

Each directory is one dated measurement with its receipts (`raw/`) and
`SHA256SUMS`. For publication, the receipts were redacted in two ways
only: the test Spark's admin user name became `user` (in paths such as
`/home/user/...`) and a private host name became `spark-b`. Every
`SHA256SUMS` was recomputed afterwards; hashes recorded inside a harness's
own manifest refer to the unredacted files. Sealed results are never edited; a later result supersedes an
earlier one, and this index says which is current.

| Result | What it measured | Status |
|---|---|---|
| [2026-11-07-release-replication](2026-11-07-release-replication/RESULT.md) | The documented install from a clean checkout: byte-identical weights, JevBench ×3, oracle, engine, merged vs solo, refusals | **Current**: the pinned engine |
| [2026-11-07-d3-ultra-w4a4](2026-11-07-d3-ultra-w4a4/RESULT.md) | The default (W4A4): JevBench ×3, kev, Decision Index sample, engine sweep, concurrency, BF16 oracle | **Current** |
| [2026-11-07-activation-formats](2026-11-07-activation-formats/RESULT.md) | Which 4-bit formats keep d3-ultra's accuracy (MXFP4, NVFP4, two-term) | Current. Its `scripts/` ran on the development setup, not `start.sh`. |
| [2026-11-07-d3-ultra-final-latency](2026-11-07-d3-ultra-final-latency/RESULT.md) | W4A8 latency (16K arena) | Current for `--precision w4a8`. Its "defaults" are no longer the defaults. |
| [2026-11-07-d3-ultra-final-accuracy](2026-11-07-d3-ultra-final-accuracy/RESULT.md) | W4A8 accuracy: kev, Decision Index sample | Current for `--precision w4a8`. Its "release candidate" is now the W4A8 option. |
| [2026-11-06-d3-ultra-nvfp4-lowp](2026-11-06-d3-ultra-nvfp4-lowp/RESULT.md), [-bf16mma](2026-11-06-d3-ultra-nvfp4-bf16mma/RESULT.md), [-nvfp4](2026-11-06-d3-ultra-nvfp4/RESULT.md) | Earlier d3-ultra kernels; the d3 BF16 runtime baseline | Superseded, except the BF16 runtime row (633 ms, 91.3%), which is the baseline. |
| [2026-11-06-knowline-*](2026-11-06-knowline-35b-a3b-gen1.5-nvfp4/RESULT.md) | KnowLine-35B-A3B, the earlier candidate | Superseded by d3-ultra. Kept as a baseline. |
| [2026-11-06-zero-shot-qwen3.6-35b-a3b-nvfp4](2026-11-06-zero-shot-qwen3.6-35b-a3b-nvfp4/RESULT.md) | Stock Qwen3.6-35B-A3B, zero-shot | Baseline. |

The engine commits named in each result are on
[Enntity/atlas](https://github.com/Enntity/atlas); the current pin is in
`install/atlas-source.json`.
