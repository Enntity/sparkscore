<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# API reference

SparkScore's HTTP API is served by the harness, `tools/sparkscore/server.py`.
The harness renders each request into token ids with the model's prompt
template (`tools/sparkscore/templates/d3_ultra.py`), sends one `/v1/score`
call to Atlas (`tools/sparkscore/backends.py`), and turns the option
probabilities into answers (`tools/sparkscore/answers.py`). This page
describes the setup that `./start.sh` serves:

- **Address:** `http://127.0.0.1:8015` (`PORT`, `BIND`).
- **Model names:** `sparkscore-d3`, with the aliases `sparkscore-latest` and
  `jev-latest`.
- **Template:** `d3-ultra`, on the `score` backend.
- **Readout temperatures:** `choice=1.636,noul=2.366,score=1.588`.

The serve block in `install/checkpoints.json` sets these.

## Compatibility with TypeSafe's Jev API

The request and response shapes follow TypeSafe's published API: the
OpenAPI document at api.typesafe.ai (version 0.2.0) and docs.typesafe.ai/api.
The status codes follow docs.typesafe.ai/api/errors. The answer fields and
confidence formulas follow TypeSafe's `system-one-adapter-python` 0.2.1 (MIT).
They are reimplemented here, not copied. A Jev client needs only a new base
URL and one of the model names above.

The shapes were written from those documents. They were not checked against
TypeSafe's hosted service. The differences that the code documents:

| Area | SparkScore |
|---|---|
| `GET /v1/limits` | An extension. It is not part of TypeSafe's API. |
| Images | `assets` and `box` questions are refused with 422: d3-ultra is served text only. |
| d3-ultra's own rules | A `choice` with an empty option key, and a `score` whose level texts are empty or repeated, are refused with 422. |
| `zstd` bodies | Accepted only when the harness runs on Python 3.14 or later. The container's Python is older, so it answers 415; use `gzip`. |
| `model` in the response | Always `sparkscore-d3`, even when the request names an alias. |
| Extra headers | `Server-Timing`, `X-SparkScore-Passes` and `X-SparkScore-Prefix-Tokens`; with `knowledge` on packs, also `X-SparkScore-Pack-Tokens` and a `pack` entry in `Server-Timing`. |
| `knowledge` | An extension. It is not part of TypeSafe's API. |

## Routes

| Route | Auth | Returns |
|---|---|---|
| `POST /v1/systemone` | bearer key, if set | One answer per question. |
| `GET /v1/models` | bearer key, if set | The model names. |
| `GET /v1/limits` | bearer key, if set | This deployment's limits. |
| `GET /health` | none | `{"status": "ok"}` once the harness is up. `./start.sh` starts the harness only after Atlas is healthy. |

Any other path is 404 with `{"detail": "Not Found"}`.

### `GET /v1/models`

```json
{"models": [
  {"name": "sparkscore-d3", "description": "SparkScore: local:model with prompt template d3-ultra on the Atlas score backend", "release_date": "2026-11-05"},
  {"name": "sparkscore-latest", "description": "…", "release_date": "2026-11-05"},
  {"name": "jev-latest", "description": "…", "release_date": "2026-11-05"}
]}
```

The description names the tokenizer source, the template and the backend.
In the container, the tokenizer is the mounted checkpoint, `/model`.

### `GET /v1/limits`

```json
{"max_choice_options": 255, "max_score_levels": 10, "max_body_bytes": 33554432,
 "per_question_context": 32768, "per_request_context": 65536, "max_concurrent_requests": 16}
```

These are the harness defaults. `./start.sh` does not change them. The same
values are recorded in every run manifest under
`results/2026-11-07-d3-ultra-w4a4/raw/`.

## `POST /v1/systemone`

### Request

```json
{
  "model": "sparkscore-d3",
  "state": "…",
  "questions": {
    "<name>": {"type": "noul" | "choice" | "score", "instructions": "…", "criteria": …},
    …
  }
}
```

| Field | Required | Type | Meaning |
|---|---|---|---|
| `model` | yes | string | `sparkscore-d3`, `sparkscore-latest` or `jev-latest`. |
| `state` | yes | string, object or array | The data the questions are about. A string is shown verbatim, anything else as JSON. An empty string is shown as `(empty)`. |
| `questions` | yes | object, at least 1 entry | Question name to question. Answers come back under the same names, in the same order. |
| `assets` | no | must be absent or `null` | Images are not supported. |
| `knowledge` | no | nonempty string | A SparkScore extension: text the model reads before the state, such as a policy or a handbook. See [Knowledge](#knowledge). |

Other top-level fields are ignored.

Every question has these fields:

| Field | Required | Type | Meaning |
|---|---|---|---|
| `type` | yes | `"noul"`, `"choice"` or `"score"` | `"box"` is refused (422). |
| `instructions` | no | string, object, array or `null` | The question. When missing or empty, the prompt says "Choose the best matching option." |
| `criteria` | depends on `type` | see below | The options. |

The criteria for each type:

- **`noul`** (yes or no): `criteria` is optional. It is an object with
  optional `"true"` and `"false"` descriptions, each a string, object, array
  or `null`. Missing descriptions are shown as "Yes / true" and "No / false".
- **`choice`** (one of named options): `criteria` is required. It is an
  object of option key to description, with 1 to 255 entries. A key with a
  `null` description is shown as the key alone. Otherwise the option is shown
  as `key: description`, with a non-string description as JSON. Keys must be
  nonempty.
- **`score`** (an ordinal level): `criteria` is required. It is a list of 1 to
  10 levels, lowest first. Each level is a string, object, array or `null`,
  and is shown by its text alone, with a non-string as JSON. With two or more
  levels the texts must be nonempty and distinct.

Some questions are answered without the model. A `choice` with one option or
a `score` with one level gets probability 1 for that option, as TypeSafe
does. If no question in a request needs the model, Atlas is not called.

### Knowledge

`knowledge` puts text after d3-ultra's system prompt, ahead of the state, and the answers follow it. Typical
uses are routing rules, policies and catalogs that every request on a domain shares.

On an Atlas started with `--score-packs N`, the knowledge is a **pack**:
- Atlas encodes it once and keeps its attention keys and values and its recurrent state.
- Later requests on the same text read that state instead of recomputing it.
- The pack is one bidirectional block that never reads the state, so it is the same for every request.
- Atlas holds up to N packs and evicts the least recently used first. The harness re-sends an evicted
  pack's tokens by itself.

On an Atlas without packs, the knowledge is part of the state block, which gives the same answers more
slowly.

| One DGX Spark, W4A4, 40 tickets × 3 questions | p50 |
|---|---:|
| No knowledge | 168 ms |
| 1.2K-token handbook in the state | 533 ms |
| The same handbook as `knowledge` (pack) | 173 ms |
| 12K-token handbook in the state | 5.45 s |
| The same handbook as `knowledge` (pack) | 280 ms |

Knowledge that contains one of the tokenizer's special tokens (`<|im_end|>` and the like) is refused
with 422.

### Response

```json
{
  "model": "sparkscore-d3",
  "answers": {"<name>": {…}, …},
  "usage": {"input_tokens": 316, "output_tokens": 0}
}
```

- **`usage.input_tokens`:** the tokens Atlas reads. The state block is
  counted once, plus each question's own tokens.
- **`usage.output_tokens`:** always 0. Nothing is generated.

Each answer has its type's shape:

| Type | Answer |
|---|---|
| `noul` | `{"type": "noul", "noul": P(true)}` |
| `choice` | `{"type": "choice", "choice": key, "confidence": c, "probabilities": {key: p, …}}` |
| `score` | `{"type": "score", "score": s, "confidence": c, "legend": {"0": level, …}, "probabilities": {"0": p, …}}` |

- **`choice`:** the first key with the highest probability. Its confidence
  is `(p_max − 1/K) / (1 − 1/K)` over K options, and 1 when K = 1.
- **`score`:** the expected level, `Σ i·p_i`. Its confidence is
  `max(0, 1 − Σ p_i·|i − m| / D)`, where m is the first most likely level and
  D is the mean of `|i − (K − 1)/2|` over the K levels. It is 1 when K = 1.
- **`legend`:** maps each level index to its `criteria` entry, as sent.
- **`probabilities`:** over the options of one question, summing to 1.

### Example

This is the request from the README:

```bash
curl -s http://127.0.0.1:8015/v1/systemone -H 'content-type: application/json' -d '{
  "model": "sparkscore-d3",
  "state": "Refund policy: refunds within 30 days; annual plans are refunded pro rata.\n\nTicket from Mina (annual plan, bought 5 months ago): \"We barely use it anymore. I would like a refund for the rest of the year.\"",
  "questions": {
    "team":    {"type": "choice", "instructions": "Which team should handle this?",
                "criteria": {"billing": "Billing", "infrastructure": "Infrastructure", "sales": "Sales"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["Can wait", "This week", "Today", "Right now"]},
    "refund":  {"type": "noul", "instructions": "Is a refund allowed under the policy?"}
  }
}'
```

The response from the live server, with values rounded to four places:

```json
{
  "model": "sparkscore-d3",
  "answers": {
    "team": {"type": "choice", "choice": "billing", "confidence": 0.9915,
             "probabilities": {"billing": 0.9943, "infrastructure": 0.0001, "sales": 0.0055}},
    "urgency": {"type": "score", "score": 0.4831, "confidence": 0.5169,
                "legend": {"0": "Can wait", "1": "This week", "2": "Today", "3": "Right now"},
                "probabilities": {"0": 0.5414, "1": 0.4386, "2": 0.0155, "3": 0.0045}},
    "refund": {"type": "noul", "noul": 0.9542}
  },
  "usage": {"input_tokens": 316, "output_tokens": 0}
}
```

The urgency fields check out by hand:

- **Score:** 1·0.4386 + 2·0.0155 + 3·0.0045 = 0.4831.
- **Confidence:** the mode is level 0, so the spread is also 0.4831. D is 1
  for four levels, so the confidence is 1 − 0.4831 = 0.5169.

The three questions are one score call: a 167-token state block and three
branches of 51, 51 and 47 tokens, 316 rows in all.

### A minimal Python client (standard library only)

```python
import json, os, urllib.error, urllib.request

def decide(state, questions, base="http://127.0.0.1:8015", model="sparkscore-d3"):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("SPARKSCORE_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["SPARKSCORE_API_KEY"]
    body = json.dumps({"model": model, "state": state, "questions": questions}).encode()
    request = urllib.request.Request(base + "/v1/systemone", body, headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)["answers"]
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode()}") from None

answers = decide("The invoice is 45 days overdue.", {
    "overdue": {"type": "noul", "instructions": "Is the invoice overdue?"},
})
print(answers["overdue"]["noul"])
```

`install/smoke.py` is a longer example that also prints the timing headers.

### Response headers

Every 200 response carries these headers:

| Header | Value |
|---|---|
| `Server-Timing` | `render;dur=…, backend;dur=…, answer;dur=…, total;dur=…`, in milliseconds |
| `X-SparkScore-Passes` | `1` when a question needed the model, `0` when none did |
| `X-SparkScore-Prefix-Tokens` | Tokens in the shared state block |

The timings in `Server-Timing`:

- **`render`:** the prompt template and tokenization.
- **`backend`:** the `/v1/score` call to Atlas, queueing included.
- **`answer`:** building the answer fields.
- **`total`:** the sum of the three. It does not include reading and
  validating the body.

`X-SparkScore-Passes` counts the harness's calls to `/v1/score`. If Atlas
splits a large request into several passes (see [ENGINE.md](ENGINE.md)), the
header still says 1.

## Authentication

Set `SPARKSCORE_API_KEY` in the environment or in `.env` before `./start.sh`.
Every route except `/health` then requires `Authorization: Bearer <key>`. A
missing or wrong key gets 401 with `{"detail": "missing or invalid API key"}`.
The key is compared in constant time. Without the variable, no key is
required.

## Request bodies

| Rule | Limit | Status |
|---|---|---|
| `Content-Length` header | required; chunked bodies are not read | 411 |
| Body size | 33,554,432 bytes (32 MiB), before and after decompression | 413 |
| `Content-Encoding` | `identity`, `gzip` (`x-gzip`), or `zstd` on Python 3.14+; one layer at most | 415 |
| Encoding | UTF-8, no byte order mark | 400 |
| JSON | valid; no `NaN` or `Infinity`, no non-finite numbers | 400 |
| Nesting | `state`, `instructions` and `criteria` at most 64 levels deep | 400 |

## Status codes

Every error body is `{"detail": …}`.

| Status | When | `detail` |
|---|---|---|
| 400 | A broken request rule (table above) | a string |
| 400 | Over the context length | `{"type": "max_tokens_exceeded", "msg": "…"}` |
| 401 | Wrong or missing bearer key | a string |
| 404 | Unknown path | `"Not Found"` |
| 411 | No `Content-Length` | a string |
| 413 | Body too large | a string |
| 415 | Unsupported `Content-Encoding` | a string |
| 422 | Schema failure, unknown model, or a question the template refuses | a list of errors |
| 500 | A fault in the harness | `"internal error: <type>"` |
| 502 | Atlas answered with an error or malformed output | a string beginning `backend …` |
| 503 | Atlas cannot be reached | a string; `Retry-After: 1` |
| 504 | Atlas did not answer within 600 s | a string |
| 529 | All 16 request slots are busy, or Atlas reported overload (429, 503 or 529) | a string; `Retry-After: 1` |

The 422 list is FastAPI-style. It holds every schema error in the request:

```json
{"detail": [{"loc": ["body", "questions", "team", "criteria"], "msg": "Field required", "type": "missing", "input": null}]}
```

The `type` values are FastAPI's (`missing`, `dict_type`, `too_short`, …),
plus `unsupported` (`assets`, `box`, or a question d3-ultra refuses) and
`unknown_model`. A question d3-ultra refuses is reported with
`loc: ["body", "state"]`, whatever question caused it.

A context-length refusal, as recorded in
`results/2026-11-07-d3-ultra-w4a4/raw/di-sample1000-w4a4all/raw/run/results.jsonl`:

```json
{"detail": {"type": "max_tokens_exceeded", "msg": "request exceeds the maximum context length: 32853 tokens for the longest question (limit 32768), 63520 for the request (limit 65536)"}}
```

## What the d3-ultra template supports

The template reproduces vllm-sr/d3-ultra's own prompts, token for token, at
revision `ccdcfa611649a59e095e0b5e1732a15cedafadd6`. `tests/test_d3.py`
checks it against d3's code.

- **Input:** text only. `assets`, `box` questions, videos and track questions
  are not served.
- **Options:** a `choice` takes 1 to 255 options. They are shown under the
  codes `A`, `B`, …, then two-letter codes, each a single token of d3's
  tokenizer. A `noul` is `A` = false, `B` = true. A `score` takes 1 to 10
  levels, `A` = level 0.
- **Readout:** d3's code readout at the last prompt token, over the codes of
  the shown options only.
- **Prompt layout:** every request starts with a header that lists all its
  questions. A choice question's header entry also lists its option texts.
  So the state block is the same for every question, and Atlas reads it
  once. A one-level `score` is left out of the header.
- **Attention:** `prefix_bidir`, as d3-ultra was trained. The state block
  attends to itself in both directions. Each question block attends to the
  state block and to itself in both directions. See [ENGINE.md](ENGINE.md).
- **Temperature:** the option logits are divided by d3's own temperature,
  1.046 for a request with one question and 1.0 otherwise. They are then
  divided by the harness's per-type temperature, `choice=1.636`,
  `noul=2.366`, `score=1.588` under `./start.sh`. Those were fitted on kev
  decision-v4 (`results/2026-11-07-d3-ultra-final-accuracy`).
- **Permutation average:** off, as d3 serves it. The harness option
  `--permutation-average on` also scores every choice with its options
  reversed, in the same call, and averages the two. `./start.sh` does not
  set it.

### Context limits

- **One question:** at most 32,768 tokens, counting its state block, its own
  tokens and one more.
- **One request:** at most 65,536 tokens, counting the state block once plus
  every question's tokens.

Both counts include d3's system prompt and the header. A longer request is
refused with the 400 `max_tokens_exceeded` above, never truncated.

### Engine limits

Atlas adds limits of its own (`/v1/score` at the commit pinned in
`install/atlas-source.json`):

- **Questions:** at most 256 that need the model per request. With
  `--permutation-average on`, each choice counts twice. The harness does not
  check this limit: a request over it fails in Atlas and comes back as 502.
- **Pass size:** one pass reads about 38K rows under the shipped
  `--max-prefill-tokens 49152` (`results/2026-11-07-d3-ultra-w4a4`) and at
  most 32 sequences (`--max-batch-size 32`). A request with more questions
  than one pass holds runs as several passes, each over the state and a
  subset of the questions. Each such pass reads the state again.
- **No truncation:** a state that does not fit one pass with a question is
  not split. With the shipped settings, the 32,768-token limit per question
  keeps every accepted question inside one pass.
