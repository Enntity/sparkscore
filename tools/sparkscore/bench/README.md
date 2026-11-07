<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Benchmarks against a SparkScore endpoint

Each public harness runs its own code, unmodified, against our
`/v1/systemone`. `run.py` only launches it with the right flags and keeps the
receipts:

- **Raw output:** everything the harness writes lands in `<out>/raw/`.
- **Manifest:** `<out>/raw/manifest.json` records the harness commit and
  whether its checkout was dirty, the exact commands with their exit codes
  and run times, and the endpoint's `/v1/models` and `/v1/limits`.
- **Checksums:** `<out>/SHA256SUMS` covers every file, so `results/<run>/`
  passes the receipt check in `.github/workflows/static.yml`.

These are unedited receipts. They may name local paths, which AGENTS.md
allows only under `results/*/raw/`. Use a fresh `--out` per run: the runner
refuses a directory that is not empty.

The runner was written against these harness commits. It warns on others.

| Harness | Repository | Commit |
|---|---|---|
| Decision Index 0.3 public suite (the 0.4 board's public part) | github.com/apolinario/decision-index (MIT) | `25c6ec7b6b5f` |
| JevBench v1.6.1 | github.com/fstandhartinger/jevbench (MIT) | `609a664e81f4` |
| kev-suites | github.com/jaredpalmer/kev (Apache-2.0), data hf.co/datasets/jaredpalmer/kev-suites | `2074cc442562` |

The runner itself is standard library only. Run it from the SparkScore
checkout:

```
python3 -m tools.sparkscore.bench.run smoke          --endpoint URL --model M [--out DIR]
python3 -m tools.sparkscore.bench.run jevbench       --endpoint URL --model M --harness DIR --out DIR [--python PY] [--limit N]
python3 -m tools.sparkscore.bench.run kev            --endpoint URL --model M --harness DIR --out DIR [--python PY] [--suite S] [--concurrency N]
python3 -m tools.sparkscore.bench.run decision-index --endpoint URL --model M --harness DIR --out DIR [--python PY] [--rows F] [--limit N]
```

- **Defaults:** `--endpoint` is `http://127.0.0.1:8008`, and `--model` is
  `sparkscore-latest`. Under `./start.sh`, pass
  `--endpoint http://127.0.0.1:8015 --model sparkscore-d3`.
- **`--python`:** the interpreter that has the harness installed. It
  defaults to the one running `run.py`.
- **Bearer key:** if `SPARKSCORE_API_KEY` is set, the runner sends it and
  passes it on to each harness.

## 1. The endpoint

`./start.sh` serves the setup of the receipts under
`results/2026-11-07-d3-ultra-w4a4`: the harness on `127.0.0.1:8015`, model
`sparkscore-d3`, template `d3-ultra` on the `score` backend, W4A4. The
readout temperatures are the fitted ones, `choice=1.636,noul=2.366,score=1.588`.
`install/checkpoints.json` has the full serve block, and
[docs/API.md](../../../docs/API.md) has the API.

Some receipts use **d3's own temperatures** instead: kev transfer-v9,
kev decision-v7 and the Decision Index sample. That is the same harness
without `--temperature`. To reproduce those runs, start a second harness
inside the container, on the same Atlas:

```sh
docker exec -d sparkscore /opt/venv/bin/python -m tools.sparkscore.server \
    --backend score --backend-url http://127.0.0.1:8081 --tokenizer /model \
    --template d3-ultra --model-id sparkscore-d3 --host 0.0.0.0 --port 8016 --quiet
D3OWN=http://$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' sparkscore):8016
curl -sf $D3OWN/health
```

Only port 8015 is published, so the second harness is reached at the
container's address. `./start.sh stop` removes it with the container.

## 2. Smoke (no harness, about a minute)

```sh
python3 -m tools.sparkscore.bench.run smoke --endpoint http://127.0.0.1:8015 \
    --model sparkscore-d3 --out results/$(date +%F)-smoke-d3
```

This sends the 30 decisions of `tests/fixtures/smoke.jsonl`: 22 requests,
all three question types, some with several questions on one state.

- **Checks:** every answer is checked the way Decision Index, JevBench and
  kev check them (keys, ranges, sums within 1e-3).
- **Output:** accuracy, Brier, top-label ECE, client latency and the
  server's own `Server-Timing` medians.
- **Exit code:** 1 if any answer fails a check.

It is a wiring check, not a benchmark. `--out` is optional here.

## 3. JevBench (public items)

**How it calls us.** The `typesafe` adapter posts one question per request,
`{"state", "model", "questions": {"decision": q}}`, to
`{endpoint}/v1/systemone`.

- **Calls:** serially, with no retries and a 120 s timeout.
- **Key:** `Authorization: Bearer $<key-env>`. The runner passes
  `SPARKSCORE_API_KEY` when it is set and `--key-env ""` otherwise, which
  sends no key.
- **Checks:** the probabilities must have exactly the option keys and sum to
  1 within 1e-3 (strict), or 2e-2 (headline, rescaled). A noul becomes
  `{"yes": p, "no": 1 − p}`.
- **Raw responses:** they must be stored outside the JevBench checkout. The
  runner puts them in `<out>/raw/responses/`.

**Install:** standard library only.

```sh
git clone https://github.com/fstandhartinger/jevbench ../jevbench
git -C ../jevbench checkout 609a664e81f46e03731600bd7bd9f1b7a61257c5
```

**Run** three times, as `results/2026-11-07-d3-ultra-w4a4` did:

```sh
for i in 1 2 3; do
  python3 -m tools.sparkscore.bench.run jevbench --harness ../jevbench \
      --endpoint http://127.0.0.1:8015 --model sparkscore-d3 \
      --out results/$(date +%F)-jevbench-d3-$i
done
```

This runs `jevbench.cli run` over `datasets/public/{easy,hard,original}.jsonl`:
231 decisions (48 + 111 + 72), of which 139 are choice, 74 noul and 18
score. It uses zero prices (cost basis `local_spark_compute_excluded`). Then
`jevbench.cli summarize` writes `raw/summary.json` and
`raw/public-export.json`.

**Expected,** from `results/2026-11-07-d3-ultra-w4a4`:

- **Every run:** 210 of 231 right (90.9%), macro 92.5%, Brier 0.118, ECE
  0.029. The score path is deterministic, so all three runs give the same
  answers.
- **Latency:** p50 146–148 ms, and p95 1.05–1.06 s.
- **Run time:** each run took 77 s (`raw/latency-lat-w4a4-r9-48k/jevbench-1/raw/manifest.json`).

**Scoring** (`jevbench/summarize.py`, `metrics.py`, `scoring.py`):

- accuracy (argmax, score items included), and macro accuracy over families;
- multi-class Brier summed over the label set (2-class for noul);
- top-label ECE over 10 equal bins;
- ordinal MAE of the expected level;
- p50 and p95 latency, measured at the harness per question;
- paraphrase agreement.

The board's Intelligence, Calibration and Capability scores also use 303
sealed items, which only the maintainers can run (`composite_v14.py`). A
local run gives the public metrics only.

## 4. kev-suites (`kev.benchmark --remote`)

**How it calls us.** `RemotePredictor` posts `{"state", "questions",
"model"}` to `{remote}/v1/systemone`, with a 120 s timeout.

- **Key:** `Authorization: Bearer $KEV_REMOTE_API_KEY`. The runner sets it
  from `SPARKSCORE_API_KEY`, or to `local` when that is unset.
- **Retries:** 408, 429 and 5xx are tried up to three times in all.
- **Refusals:** 400, 413 and 422 count as rejected records on eval-only
  suites and are fatal otherwise.
- **Checks:** noul answers become `{"true": p, "false": 1 − p}`. Choice and
  score probabilities are renormalised and must sum to 1 within
  `max(1e-5, 0.005·K)`.

**Install.** `kev.benchmark` imports torch, transformers and datasets even
for `--remote`, but a CPU torch is enough.

```sh
git clone https://github.com/jaredpalmer/kev ../kev
cd ../kev && git checkout 2074cc44256200bb37540103fa4572901faa10f3
uv sync          # Python 3.13 (.python-version); the lock has a torch 2.8.0 aarch64 wheel (about 100 MB)
cd -
```

The suites' development partitions are in the repository. Missing
partitions are fetched from hf.co/datasets/jaredpalmer/kev-suites at
revision `cc4bac80` and checked against the manifest's sha256.

**Run** as recorded in `results/2026-11-07-d3-ultra-w4a4/raw/kev-*/raw/manifest.json`.
transfer-v4 runs on the fitted temperatures (port 8015). transfer-v9 and
decision-v7 run on d3's own (`$D3OWN`, section 1).

```sh
python3 -m tools.sparkscore.bench.run kev --harness ../kev --python ../kev/.venv/bin/python \
    --endpoint http://127.0.0.1:8015 --model sparkscore-d3 --suite evals/v4/transfer-v4 \
    --out results/$(date +%F)-kev-transfer-v4-d3
for suite in evals/v9/transfer-v9 evals/v7/decision-v7; do
  python3 -m tools.sparkscore.bench.run kev --harness ../kev --python ../kev/.venv/bin/python \
      --endpoint $D3OWN --model sparkscore-d3 --suite $suite \
      --out results/$(date +%F)-kev-$(basename $suite)-d3-own
done
```

Without `--suite` the runner runs `evals/v4/transfer-v4`. `--concurrency N`
keeps N requests in flight; the receipts used 1.

**Expected,** clean rows, from `results/2026-11-07-d3-ultra-w4a4`:

| Suite | Temperatures | Records | Clean n | Accuracy | NLL | ECE | Coverage at 5% error | Run time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| transfer-v4 | fitted | 764 | 656 | 87.7% | 0.334 | 0.030 | 0.793 | 105 s |
| transfer-v9 | d3's own | 1,264 | 1,046 | 83.7% | 0.549 | 0.089 | 0.682 | 185 s |
| decision-v7 | d3's own | 1,204 | 1,264 | 86.9% | 0.425 | 0.064 | 0.795 | 239 s |

The records column counts the lines of `predictions.jsonl`, and the clean
column is the `clean` block of `report.json`. The states have at most 384
tokens. The receipts' kev runs were on Atlas `8434214d8`, the first W4A4
kernels. Later kernel rounds were checked bit-identical to it.

**Scoring** (`kev/metrics.py`, `kev/benchmark.py`):

- accuracy, NLL, multi-class Brier, and top-label ECE over 10 bins;
- confident-error rate (wrong at p ≥ 0.9);
- coverage at a 5% and a 1% error budget, and AURC;
- score MAE and RPS;
- option-order flip rate on permuted copies, and isolation;
- over-confidence on "unknowable" records (transfer-v9).

`report.json` is the result.

## 5. Decision Index (public index)

**How it calls us.** `--engine http` posts `{"model", "state", "questions"}`
to `{base_url}/v1/systemone`.

- **Calls:** one request at a time after two warm-up requests, with a 600 s
  timeout.
- **Key:** `Authorization: Bearer $DECISION_INDEX_API_KEY` when that is set.
  The runner sets it from `SPARKSCORE_API_KEY`.
- **Capacity refusals:** a refusal whose body contains a known capacity
  marker is recorded as `unsupported`. Our 400 `max_tokens_exceeded` is
  recorded that way in the receipts. Other failures are `error`.
- **Questions:** only `choice` and `noul`, with 2 to 255 options.
- **Model name:** the engine's default is `default`, so the runner passes
  `--option model=<--model>`.

**Install** next to the SparkScore checkout. The suite can be built on
another machine and its `suite-0.3/` copied over.

```sh
git clone https://github.com/apolinario/decision-index ../decision-index
cd ../decision-index && git checkout 25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d
python3 -m venv .venv && .venv/bin/pip install -e ".[rebuild]"
export HF_HUB_DISABLE_XET=1      # and `hf auth login` after accepting the HLE terms on the Hub
.venv/bin/python -m decision_index suite rebuild --work work      # about 7 GB down, 17 GB of working space
.venv/bin/python -m decision_index suite import \
    --rows work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
    --added-rows work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz \
    --gsm8k-rows work/artifacts/benchmark-suite/release-v3-rebuilt/gsm8k-rows.jsonl.gz
cd -
```

**Run the 1,000-row sample** on d3's own temperatures, as recorded in
`results/2026-11-07-d3-ultra-w4a4/raw/di-sample1000-w4a4all/raw/manifest.json`:

```sh
(cd ../decision-index && .venv/bin/python -m decision_index suite sample --n 1000 --out sample-1000.jsonl.gz)
python3 -m tools.sparkscore.bench.run decision-index --harness ../decision-index \
    --python ../decision-index/.venv/bin/python --endpoint $D3OWN --model sparkscore-d3 \
    --rows ../decision-index/sample-1000.jsonl.gz --out results/$(date +%F)-di-sample1000-d3-own
```

To check that your sample has the same rows, compare its `run_id`s with the
receipt's `raw/run/results.jsonl`. For the full public suite, drop `--rows`.

This runs `decision_index pipeline` (run, then score). If restarted on the
same `--out/raw/run`, it resumes from its own `results.jsonl` and retries
the rows with status `error`.

**Expected,** from `results/2026-11-07-d3-ultra-w4a4`:

- **Macro:** 0.756 over the native metrics of 42 benchmarks, computed by a
  comparison script that is not in this repository. The kit's own outputs
  are `benchmark-summary.json`, `index.json` and `scores.json`.
- **Failures:** one, a BRIGHT request over the 32,768-token limit per
  question.
- **Run time:** 33 minutes.

The receipt's run was on a 32K arena. The BRIGHT requests too long for one
32K pass were retried at 48K, and the retries were appended to its results
file; the last record per request is the one scored. Under `./start.sh`'s
48K arena they run the first time.

**Scoring** (`decision_index/scoring/index02.py`, edition 0.3):

1. Per benchmark, the native metric (accuracy, macro-F1, nDCG@10, case-exact
   accuracy, ...) times coverage. Unanswered counts as wrong.
2. Chance-corrected `skill = clip((raw − chance)/(1 − chance), 0, 1)`.
3. Benchmarks are averaged inside five areas; gold benchmarks weigh 1.2.
4. The areas are weighted 25.8/25.8/20.0/18.3/10 into the index (0–100).

The kit computes no ECE or Brier for the public index. The board's ECE and
its Full score (private parts only) are computed by the maintainers on
submission.

**Runtime of the full suite.** The full public run is 140,178 requests
(109,759 + 30,419 scoreable), sent one at a time. At t seconds per request it
takes 39·t hours. The sample's 1,000 requests averaged about 2 s each,
because the suite has long states (BRIGHT reaches 30K tokens).

## 6. Engine-level checks

Three scripts next to `run.py` talk to Atlas's `/v1/score` directly. Run
them inside the container, where Atlas listens on `127.0.0.1:8081`.
[docs/REPRODUCE.md](../../../docs/REPRODUCE.md) gives the commands and the
expected tables.

- **`engine_sweep.py`:** engine time per pass, by prefix length and branch
  count.
- **`concurrency.py`:** K concurrent requests, how many merged per pass, and
  the largest difference from solo runs.
- **`oracle_check.py`:** agreement and KL against d3-ultra's own BF16
  runtime, on its exact prompt tokens (`data/`).

## Other harness options

The harness also serves the older results' setups. `python3 -m
tools.sparkscore.server --help` lists its options.

- **Templates:** `--template chat-v1` (zero-shot Qwen3.6) and
  `--template knowline-gen1.5` (PelaAI/KnowLine-35B-A3B-Gen1.5, with
  `--rotations` and `--read-twice`).
- **Echo backend:** `--backend echo` scores through Atlas's
  `/v1/completions` prompt logprobs, one prompt per question or per option.
  It is slow and is the exact oracle for causal templates. d3-ultra needs
  `--backend score`, because only `/v1/score` takes the `prefix_bidir` mask.
- **Temperature:** name any `--temperature` in a result that uses it.
