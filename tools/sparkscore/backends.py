# SPDX-License-Identifier: AGPL-3.0-only
"""Atlas backends. Each scores (prefix, branches, option ids) and returns,
per branch, the probabilities of its options normalised over those options,
plus the number of prompts the engine had to run.

- echo:  /v1/completions with prompt token ids, echo, logprobs and
         max_tokens 0. Exists today; slow, and the exact oracle. Atlas does
         not score the position after the last prompt token, so an option is
         scored by appending its token and reading that token's logprob.
- score: /v1/score, the one-pass score path of docs/ENGINE.md. A template
         that needs another attention mask (d3-ultra: "prefix_bidir") names
         it in the request; the others send nothing and get causal attention.
"""

import http.client
import json
import math
import threading
from urllib.parse import urlsplit

TOP_LOGPROBS = 20  # Atlas clamps completion logprobs to 20


class BackendError(Exception):
    def __init__(self, message, status=502):
        super().__init__(message)
        self.status = status


def _softmax(logprobs):
    peak = max(logprobs)
    if not math.isfinite(peak):
        raise BackendError("backend returned no finite option logprob")
    weights = [math.exp(x - peak) for x in logprobs]
    total = math.fsum(weights)
    return [w / total for w in weights]


class _Http:
    """JSON POSTs over one keep-alive connection per thread."""

    def __init__(self, base_url, timeout):
        url = urlsplit(base_url)
        if url.scheme not in ("http", "https"):
            raise ValueError(f"backend URL must be http or https: {base_url!r}")
        self.scheme, self.netloc, self.path = url.scheme, url.netloc, url.path.rstrip("/")
        self.timeout = timeout
        self.local = threading.local()

    def _connection(self):
        conn = getattr(self.local, "conn", None)
        if conn is None:
            kind = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
            conn = self.local.conn = kind(self.netloc, timeout=self.timeout)
        return conn

    def post(self, path, body):
        data = json.dumps(body, separators=(",", ":")).encode()
        headers = {"Content-Type": "application/json"}
        for attempt in range(2):  # the second try only replaces a stale keep-alive connection
            conn = self._connection()
            try:
                conn.request("POST", self.path + path, data, headers)
                response = conn.getresponse()
                payload = response.read()
                break
            except TimeoutError as error:
                conn.close()
                self.local.conn = None
                raise BackendError(f"backend timed out: {error}", 504) from error
            except (OSError, http.client.HTTPException) as error:
                conn.close()
                self.local.conn = None
                if attempt:
                    raise BackendError(f"cannot reach backend: {error}", 503) from error
        if response.status != 200:
            # 529: every slot busy. 422: the engine refused this request (for
            # example, a single question longer than one score pass). 502: anything else.
            status = 529 if response.status in (429, 503, 529) else 422 if response.status in (400, 413, 422) else 502
            raise BackendError(f"backend HTTP {response.status}: {payload[:300]!r}", status)
        try:
            return json.loads(payload)
        except ValueError as error:
            raise BackendError(f"backend returned invalid JSON: {error}") from error


class EchoBackend:
    name = "echo"

    def __init__(self, base_url, option_text, model="default", timeout=600.0, topk=True):
        """option_text: {token id: decoded string}, which is how completion
        top_logprobs name tokens."""
        self.http = _Http(base_url, timeout)
        self.option_text = option_text
        self.model = model
        self.topk = topk

    def _last_logprobs(self, prompts):
        """For each prompt: (logprob of its last token, top logprobs there)."""
        body = {
            "model": self.model,
            "prompt": prompts,
            "max_tokens": 0,
            "echo": True,
            "logprobs": TOP_LOGPROBS,
            "temperature": 1.0,
        }
        choices = self.http.post("/v1/completions", body).get("choices")
        if not isinstance(choices, list) or len(choices) != len(prompts):
            raise BackendError("backend returned the wrong number of completion choices")
        out = []
        for choice in sorted(choices, key=lambda c: c.get("index", 0)):
            try:
                logprobs = choice["logprobs"]
                last = float(logprobs["token_logprobs"][-1])
                top = logprobs["top_logprobs"][-1] or {}
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise BackendError(f"backend returned no prompt logprobs: {error!r}") from error
            out.append((last, top))
        return out

    def score(self, prefix, branches, options):
        logprobs = [None] * len(branches)
        first = {}  # branch -> exact logprob of its first option
        passes = 0
        if self.topk:
            # One prompt per question; its top logprobs at the answer slot
            # are enough when every option is among them.
            todo = [j for j, o in enumerate(options) if len(o) <= TOP_LOGPROBS]
            results = self._last_logprobs([prefix + branches[j] + [options[j][0]] for j in todo]) if todo else []
            passes += len(todo)
            for j, (last, top) in zip(todo, results):
                first[j] = last
                names = [self.option_text[i] for i in options[j]]
                # The appended token's own logprob checks the string mapping.
                if all(n in top for n in names) and abs(top[names[0]] - last) <= 1e-3:
                    logprobs[j] = [float(top[n]) for n in names]
        work = [
            (j, k)
            for j, o in enumerate(options)
            if logprobs[j] is None
            for k in range(len(o))
            if not (k == 0 and j in first)
        ]
        if work:
            results = self._last_logprobs([prefix + branches[j] + [options[j][k]] for j, k in work])
            passes += len(work)
            exact = {j: [None] * len(o) for j, o in enumerate(options) if logprobs[j] is None}
            for j, value in first.items():
                if j in exact:
                    exact[j][0] = value
            for (j, k), (last, _) in zip(work, results):
                exact[j][k] = last
            for j, values in exact.items():
                logprobs[j] = values
        return [_softmax(lp) for lp in logprobs], passes


class VllmBackend:
    """A reference engine (vLLM's OpenAI server, e.g. on BF16 weights).

    One request per question reads the top logprobs at the answer slot
    (`max_tokens: 1`); questions with an option outside them fall back to one
    prompt per option, scored with `prompt_logprobs`."""

    name = "vllm"

    def __init__(self, base_url, option_text, model="default", timeout=600.0):
        self.http = _Http(base_url, timeout)
        self.option_text = option_text
        self.model = model

    def score(self, prefix, branches, options):
        body = {"model": self.model, "prompt": [prefix + b for b in branches], "max_tokens": 1,
                "logprobs": TOP_LOGPROBS, "temperature": 0.0}
        choices = sorted(self.http.post("/v1/completions", body)["choices"], key=lambda c: c.get("index", 0))
        out, passes = [], len(branches)
        for b, opts, choice in zip(branches, options, choices):
            top = (choice.get("logprobs") or {}).get("top_logprobs") or [{}]
            names = [self.option_text[o] for o in opts]
            if all(n in top[0] for n in names):
                out.append(_softmax([float(top[0][n]) for n in names]))
                continue
            body = {"model": self.model, "prompt": [prefix + b + [o] for o in opts], "max_tokens": 1,
                    "prompt_logprobs": 0, "temperature": 0.0}
            rows = sorted(self.http.post("/v1/completions", body)["choices"], key=lambda c: c.get("index", 0))
            passes += len(opts)
            out.append(_softmax([float(r["prompt_logprobs"][-1][str(o)]["logprob"]) for r, o in zip(rows, opts)]))
        return out, passes


class ScoreBackend:
    name = "score"

    def __init__(self, base_url, timeout=600.0, attention=None):
        self.http = _Http(base_url, timeout)
        self.attention = attention

    def score(self, prefix, branches, options):
        body = {"prefix": prefix, "branches": branches, "options": options}
        if self.attention:
            body["attention"] = self.attention
        response = self.http.post("/v1/score", body)
        probs = response.get("probs") if isinstance(response, dict) else None
        if not isinstance(probs, list) or len(probs) != len(branches):
            raise BackendError("backend returned the wrong number of branches")
        out = []
        for row, o in zip(probs, options):
            if not isinstance(row, list) or len(row) != len(o):
                raise BackendError("backend returned the wrong number of options")
            values = [float(p) for p in row]
            total = math.fsum(values)
            if not all(math.isfinite(p) and p >= 0 for p in values) or total <= 0:
                raise BackendError("backend returned invalid probabilities")
            out.append([p / total for p in values])
        return out, 1
