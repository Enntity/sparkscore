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

import hashlib
import http.client
import json
import math
import threading
from urllib.parse import urlsplit

TOP_LOGPROBS = 20  # Atlas clamps completion logprobs to 20


class BackendError(Exception):
    def __init__(self, message, status=502, upstream=None):
        super().__init__(message)
        self.status = status
        self.upstream = upstream  # the engine's own HTTP status, when it answered


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
            raise BackendError(f"backend HTTP {response.status}: {payload[:300]!r}", status, response.status)
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
    """Atlas's `/v1/score`. Knowledge packs: the first request on a pack sends its tokens and Atlas encodes
    them once; later ones send the `pack_id` it answered with (the tokens again if Atlas no longer holds
    it). An Atlas without packs gets the pack as the start of the prefix instead (the same tokens, read as
    part of the state).

    Images go the same way: the state block with its images is an image pack (a photo asked many
    questions is encoded once, and Atlas's pack store keeps it compressed), later requests on the same
    state and images send only its `pack_id`. An Atlas without image packs gets them in the prefix."""

    name = "score"
    packs = True
    images = True

    def __init__(self, base_url, timeout=600.0, attention=None):
        self.http = _Http(base_url, timeout)
        self.attention = attention
        self.pack_ids = {}  # pack key (_pack_key) -> Atlas pack id
        self.image_packs = True  # until Atlas refuses images with a pack
        self.lock = threading.Lock()

    @staticmethod
    def _pack_key(pack, images=()):
        """A fixed-size key for a pack's tokens and images (a pack can be thousands of tokens)."""
        digest = hashlib.sha256(len(images).to_bytes(8, "little"))
        digest.update(b"".join(int(t).to_bytes(4, "little") for t in pack))
        for image in images:
            digest.update(len(image.data).to_bytes(8, "little") + image.data)
        return digest.hexdigest()

    def _post_pack(self, body, pack, report, images=()):
        key = self._pack_key(pack, images)
        with self.lock:
            known = self.pack_ids.get(key)
        if known is not None:
            try:
                response = self.http.post("/v1/score", dict(body, pack_id=known))
                report["pack"] = response.get("pack", "held") if isinstance(response, dict) else "held"
                return response
            except BackendError as error:
                if error.upstream != 404:
                    raise
                with self.lock:
                    self.pack_ids.pop(key, None)
        wire = {"images": [image.wire() for image in images]} if images else {}
        try:
            response = self.http.post("/v1/score", dict(body, pack=pack, **wire))
        except BackendError as error:
            # Atlas refuses packs it was not started for, naming the flag (`--score-packs N`), and an
            # Atlas without image packs refuses images with a pack.
            refusals = ("--score-packs", "packs are off", "images cannot be combined with pack")
            if error.upstream != 422 or not any(s in str(error) for s in refusals):
                raise
            if images:
                self.image_packs = False
            report["pack"] = "off"
            return self.http.post("/v1/score", dict(body, prefix=pack + body["prefix"], **wire))
        if isinstance(response, dict) and isinstance(response.get("pack_id"), str):
            with self.lock:
                if len(self.pack_ids) >= 4096:
                    self.pack_ids.pop(next(iter(self.pack_ids)))
                self.pack_ids[key] = response["pack_id"]
            report["pack"] = response.get("pack", "built")
        return response

    def score(self, prefix, branches, options, pack=None, report=None, images=None):
        body = {"prefix": prefix, "branches": branches, "options": options}
        if self.attention:
            body["attention"] = self.attention
        if images and self.image_packs:
            # The state block and its images as one image pack, the questions on it.
            response = self._post_pack(dict(body, prefix=[]), prefix, report if report is not None else {}, images)
        elif images:
            # An Atlas without image packs: the images in the prefix, one call.
            if report is not None:
                report["pack"] = "off"
            response = self.http.post("/v1/score", dict(body, images=[image.wire() for image in images]))
        elif pack:
            response = self._post_pack(body, pack, report if report is not None else {})
        else:
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
