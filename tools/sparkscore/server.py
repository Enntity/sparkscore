# SPDX-License-Identifier: AGPL-3.0-only
"""A Jev-compatible System One server in front of Atlas.

    python3 -m tools.sparkscore.server --backend echo --backend-url http://127.0.0.1:8000

Routes:
- POST /v1/systemone  TypeSafe's request and response shapes (api.typesafe.ai
                      openapi.json 0.2.0; docs.typesafe.ai/api)
- GET  /v1/models     TypeSafe's model list shape
- GET  /v1/limits     this deployment's limits (an openjev-sglang extension,
                      not part of TypeSafe's API)
- GET  /v1/knowledge, PUT|DELETE /v1/knowledge/NAME
                      named knowledge documents (`tools.sparkscore.knowledge`;
                      a SparkScore extension)
- GET  /health

Status codes follow docs.typesafe.ai/api/errors: 400 for a broken request
rule (invalid UTF-8 or JSON, a byte order mark, nesting over 64, a
non-finite number, or a request over the context length, with
detail.type "max_tokens_exceeded"), 401 for a wrong bearer key, 413 for a
body over the size limit, 415 for an unsupported Content-Encoding, 422 for a
schema failure (FastAPI-style detail list), 529 when every slot is busy.

Every answer carries a Server-Timing header: render, backend, answer and
total in milliseconds. X-SparkScore-Passes is the number of prompts the
engine ran (1 on the score path).
"""

import argparse
import hmac
import json
import math
import os
import threading
import time
import zlib
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tools.sparkscore import answers, knowledge as kb, render
from tools.sparkscore.backends import BackendError, EchoBackend, ScoreBackend, VllmBackend

try:  # Python 3.14+; the official Python SDK sends zstd bodies when it can
    from compression import zstd
except ImportError:
    zstd = None

MAX_CHOICE_OPTIONS = 255
MAX_QUESTIONS = 256  # Atlas /v1/score takes at most 256 branches per request
MAX_SCORE_LEVELS = 10
MAX_NESTING = 64
KINDS = ("noul", "choice", "score")


class RequestError(Exception):
    def __init__(self, status, detail, headers=None):
        super().__init__(str(detail))
        self.status, self.detail, self.headers = status, detail, headers or {}


def parse_temperature(spec):
    """'1.2' -> {'*': 1.2}; 'choice=1.4,noul=2' -> {'choice': 1.4, 'noul': 2.0} (others 1)."""
    out = {"*": 1.0}
    for part in str(spec).split(","):
        key, _, value = part.rpartition("=")
        key = key.strip() or "*"
        if key not in ("*", "noul", "choice", "score"):
            raise ValueError(f"unknown question type {key!r}")
        t = float(value)
        if not (t > 0 and math.isfinite(t)):
            raise ValueError("temperatures must be positive numbers")
        out[key] = t
    return out


@dataclass
class Config:
    model_id: str = "sparkscore-qwen3.6-35b-a3b"
    aliases: list = field(default_factory=lambda: ["sparkscore-latest", "jev-latest"])
    release_date: str = "2026-11-05"
    # readout temperature per question type; "*" is the default for any type
    temperature: dict = field(default_factory=lambda: {"*": 1.0})
    per_question_context: int = 32768
    per_request_context: int = 65536
    max_body_bytes: int = 32 * 1024 * 1024
    max_concurrent: int = 16
    api_key: str | None = None

    @property
    def names(self):
        return [self.model_id, *self.aliases]


# ------------------------------------------------------------------ request rules (400, 413, 415)
def decode_body(raw, content_encoding, limit):
    codings = [c.strip().lower() for c in (content_encoding or "").split(",") if c.strip()]
    codings = [c for c in codings if c != "identity"]
    if len(codings) > 1:
        raise RequestError(415, "use at most one Content-Encoding layer")
    if codings:
        coding = codings[0]
        if coding in ("gzip", "x-gzip"):
            decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        elif coding == "zstd" and zstd is not None:
            decoder = zstd.ZstdDecompressor(options={zstd.DecompressionParameter.window_log_max: 23})  # 8 MiB
        else:
            raise RequestError(415, f"unsupported Content-Encoding {coding!r}")
        try:
            raw = decoder.decompress(raw, limit + 1)
        except Exception as error:  # zlib.error, zstd.ZstdError
            raise RequestError(400, f"malformed {coding} body") from error
        if len(raw) > limit:
            raise RequestError(413, f"the decompressed body is over {limit} bytes")
        if not decoder.eof:
            raise RequestError(400, f"malformed {coding} body")
    if raw.startswith(b"\xef\xbb\xbf"):
        raise RequestError(400, "send UTF-8 JSON without a byte order mark")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RequestError(400, f"the body is not valid UTF-8: {error}") from error

    def finite(token):
        value = float(token)
        if not math.isfinite(value):
            raise ValueError(f"number {token} is not finite")
        return value

    def constant(token):
        raise ValueError(f"{token} is not a finite number")

    try:
        body = json.loads(text, parse_float=finite, parse_constant=constant)
    except RecursionError as error:
        raise RequestError(400, f"nesting is deeper than {MAX_NESTING} levels") from error
    except ValueError as error:
        raise RequestError(400, f"invalid JSON: {error}") from error
    # The per-field limit is checked in validate(); this bound keeps every
    # value the server may echo back in an error shallow enough to encode.
    if nesting(body) > MAX_NESTING + 4:
        raise RequestError(400, f"nesting is deeper than {MAX_NESTING} levels")
    return body


def nesting(value):
    """Levels of objects and arrays: 0 for a scalar, 1 for {"a": 1} or []."""
    deepest, stack = 0, [(value, 0)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, (dict, list)):
            depth += 1
            deepest = max(deepest, depth)
            stack.extend((v, depth) for v in (item.values() if isinstance(item, dict) else item))
    return deepest


# ------------------------------------------------------------------ schema (422)
def _text(value, nullable=True):
    return isinstance(value, (str, dict, list)) or (nullable and value is None)


def validate(body, config):
    """-> (state, questions, knowledge). Mirrors TypeSafe's SystemOneRequest schema, plus SparkScore's
    optional `knowledge`: text the model reads before the state (a policy, a handbook), held by the engine
    as a knowledge pack where it can."""
    errors = []

    def error(loc, msg, kind, value):
        errors.append({"loc": ["body", *loc], "msg": msg, "type": kind, "input": value})

    if not isinstance(body, dict):
        raise RequestError(422, [{"loc": ["body"], "msg": "Input should be a valid dictionary", "type": "dict_type", "input": body}])
    for key in ("model", "state", "questions"):
        if key not in body:
            error([key], "Field required", "missing", None)
    model, state, questions = body.get("model"), body.get("state"), body.get("questions")
    if "model" in body and not isinstance(model, str):
        error(["model"], "Input should be a valid string", "string_type", model)
    if "state" in body and not _text(state, nullable=False):
        error(["state"], "Input should be a string, an object or an array", "state_type", state)
    if body.get("assets") is not None:
        error(["assets"], "assets (images) are not supported by this model", "unsupported", None)
    knowledge, knowledge_name = body.get("knowledge"), body.get("knowledge_name")
    if knowledge is not None and (not isinstance(knowledge, str) or not knowledge):
        error(["knowledge"], "Input should be a nonempty string", "string_type", knowledge)
    if knowledge_name is not None and not isinstance(knowledge_name, str):
        error(["knowledge_name"], "Input should be a valid string", "string_type", knowledge_name)
    if knowledge is not None and knowledge_name is not None:
        error(["knowledge_name"], "send knowledge or knowledge_name, not both", "value_error", knowledge_name)
    if "questions" in body:
        if not isinstance(questions, dict):
            error(["questions"], "Input should be a valid dictionary", "dict_type", questions)
        elif not questions:
            error(["questions"], "Dictionary should have at least 1 item", "too_short", questions)
        elif len(questions) > MAX_QUESTIONS:
            error(["questions"], f"a request takes at most {MAX_QUESTIONS} questions, got {len(questions)}", "too_long", len(questions))
        else:
            for name, q in questions.items():
                _validate_question(name, q, error)
    if errors:
        raise RequestError(422, errors)
    if model not in config.names:
        raise RequestError(422, [{"loc": ["body", "model"], "msg": f"unknown model {model!r}; GET /v1/models lists the names", "type": "unknown_model", "input": model}])
    for loc, value in [(["state"], state)] + [
        (["questions", name, part], q.get(part)) for name, q in questions.items() for part in ("instructions", "criteria")
    ]:
        if nesting(value) > MAX_NESTING:
            raise RequestError(400, f"{'.'.join(map(str, loc))} is nested deeper than {MAX_NESTING} levels")
    return state, questions, knowledge, knowledge_name


def _validate_question(name, q, error):
    loc = ["questions", name]
    if not isinstance(q, dict):
        return error(loc, "Input should be a valid dictionary", "dict_type", q)
    kind = q.get("type")
    if kind == "box":
        return error([*loc, "type"], "box questions need a vision model; this model takes noul, choice and score", "unsupported", kind)
    if kind not in KINDS:
        return error([*loc, "type"], "Input tag should be one of 'noul', 'choice', 'score'", "union_tag_invalid", kind)
    if not _text(q.get("instructions")):
        error([*loc, "instructions"], "Input should be a string, an object, an array or null", "text_type", q.get("instructions"))
    criteria = q.get("criteria")
    cloc = [*loc, "criteria"]
    if kind == "noul":
        if criteria is not None:
            if not isinstance(criteria, dict):
                error(cloc, "Input should be a valid dictionary or null", "dict_type", criteria)
            else:
                for side in ("true", "false"):
                    if not _text(criteria.get(side)):
                        error([*cloc, side], "Input should be a string, an object, an array or null", "text_type", criteria.get(side))
    elif kind == "choice":
        if "criteria" not in q:
            error(cloc, "Field required", "missing", None)
        elif not isinstance(criteria, dict):
            error(cloc, "Input should be a valid dictionary", "dict_type", criteria)
        elif not 1 <= len(criteria) <= MAX_CHOICE_OPTIONS:
            error(cloc, f"a choice takes 1 to {MAX_CHOICE_OPTIONS} options per choice, got {len(criteria)}", "too_long" if criteria else "too_short", len(criteria))
        else:
            for key, value in criteria.items():
                if not _text(value):
                    error([*cloc, key], "Input should be a string, an object, an array or null", "text_type", value)
    else:
        if "criteria" not in q:
            error(cloc, "Field required", "missing", None)
        elif not isinstance(criteria, list):
            error(cloc, "Input should be a valid list", "list_type", criteria)
        elif not 1 <= len(criteria) <= MAX_SCORE_LEVELS:
            error(cloc, f"a score takes 1 to {MAX_SCORE_LEVELS} levels, got {len(criteria)}", "too_long" if criteria else "too_short", len(criteria))
        else:
            for i, value in enumerate(criteria):
                if not _text(value):
                    error([*cloc, i], "Input should be a string, an object, an array or null", "text_type", value)


# ------------------------------------------------------------------ the service
class Service:
    def __init__(self, renderer, backend, config):
        self.renderer, self.backend, self.config = renderer, backend, config
        self.slots = threading.BoundedSemaphore(config.max_concurrent)
        self.library = kb.Library(self.warm)

    def warm(self, text):
        """One small request on `text`, so its knowledge pack is encoded before it is needed."""
        _, _, _, _, pack = self._evaluate(kb.WARM_STATE, kb.WARM_QUESTIONS, text)
        return pack if pack else (None, 0)

    def systemone(self, body):
        """-> (response, timings in ms, prompts the engine ran, prefix tokens, pack status or None)."""
        state, questions, knowledge, knowledge_name = validate(body, self.config)
        if knowledge_name is not None:
            try:
                knowledge = self.library.text(knowledge_name)
            except kb.KnowledgeError as error:
                raise RequestError(error.status, [{"loc": ["body", "knowledge_name"], "msg": error.detail,
                                                   "type": "unknown_knowledge", "input": knowledge_name}]) from error
        if not self.slots.acquire(blocking=False):
            raise RequestError(529, "every slot is busy; retry shortly", {"Retry-After": "1"})
        try:
            return self._evaluate(state, questions, knowledge)
        finally:
            self.slots.release()

    def _evaluate(self, state, questions, knowledge=None):
        cfg = self.config
        t0 = time.perf_counter()
        try:
            if knowledge is None:
                rendered = self.renderer.render(state, questions)
            elif not hasattr(self.renderer, "knowledge_head"):
                raise render.Unsupported("knowledge is supported by the d3-ultra template only")
            else:
                rendered = self.renderer.render(state, questions, knowledge, pack=getattr(self.backend, "packs", False))
        except render.Unsupported as error:
            raise RequestError(422, [{"loc": ["body", "state"], "msg": str(error), "type": "unsupported", "input": None}]) from error
        longest = max(len(b.ids) for b in rendered.branches) + len(rendered.pack) + len(rendered.prefix) + 1
        if longest > cfg.per_question_context or rendered.input_tokens > cfg.per_request_context:
            raise RequestError(400, {
                "type": "max_tokens_exceeded",
                "msg": (f"request exceeds the maximum context length: {longest} tokens for the longest question "
                        f"(limit {cfg.per_question_context}), {rendered.input_tokens} for the request "
                        f"(limit {cfg.per_request_context})"),
            })
        t1 = time.perf_counter()
        # One score call for every branch: each question, at each option
        # rotation of the first pass. A template's second pass, if any, is a
        # second call with the same prefix.
        shown, passes, input_tokens = {}, 0, rendered.input_tokens  # shown: name -> {offset: probs}
        report = {}

        def score(branches):
            ids, opts = [b.ids for b in branches], [b.option_ids for b in branches]
            if rendered.pack:
                rows, n = self.backend.score(rendered.prefix, ids, opts, pack=rendered.pack, report=report)
            else:
                rows, n = self.backend.score(rendered.prefix, ids, opts)
            for b, p in zip(branches, rows):
                t = cfg.temperature.get(questions[b.name]["type"], cfg.temperature.get("*", 1.0)) * rendered.temperature
                shown.setdefault(b.name, {})[b.offset] = answers.temper(p, t)
            return n

        scored = [b for b in rendered.branches if len(b.keys) > 1]
        if scored:
            passes = score(scored)
            more = rendered.second_pass(shown) if rendered.second_pass else []
            if more:
                passes += score(more)
                input_tokens += sum(len(b.ids) for b in more)
        t2 = time.perf_counter()
        keys = {b.name: b.keys for b in rendered.branches}
        result = {
            name: answers.answer(q, keys[name], render.combine(shown[name]) if name in shown else [1.0])
            for name, q in questions.items()
        }
        response = {
            "model": cfg.model_id,
            "answers": result,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }
        t3 = time.perf_counter()
        ms = lambda a, b: (b - a) * 1000
        timings = {"render": ms(t0, t1), "backend": ms(t1, t2), "answer": ms(t2, t3), "total": ms(t0, t3)}
        pack = (report.get("pack", "held"), len(rendered.pack)) if rendered.pack else None
        return response, timings, passes, len(rendered.prefix), pack

    def models(self):
        cfg, r = self.config, self.renderer
        description = (f"SparkScore: {getattr(r, 'source', '?')} with prompt template "
                       f"{getattr(r, 'version', '?')} on the Atlas {getattr(self.backend, 'name', '?')} backend")
        return {"models": [{"name": n, "description": description, "release_date": cfg.release_date} for n in cfg.names]}

    def limits(self):
        cfg = self.config
        return {
            "max_questions": MAX_QUESTIONS,
            "max_choice_options": MAX_CHOICE_OPTIONS,
            "max_score_levels": MAX_SCORE_LEVELS,
            "max_body_bytes": cfg.max_body_bytes,
            "per_question_context": cfg.per_question_context,
            "per_request_context": cfg.per_request_context,
            "max_concurrent_requests": cfg.max_concurrent,
        }


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "SparkScore"

    @property
    def service(self):
        return self.server.service

    def log_message(self, fmt, *args):
        if not getattr(self.server, "quiet", False):
            super().log_message(fmt, *args)

    def _send(self, status, payload, headers=None, close=False):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self):
        key = self.service.config.api_key
        sent = self.headers.get("Authorization", "").encode()
        if key and not hmac.compare_digest(sent, f"Bearer {key}".encode()):
            self._send(401, {"detail": "missing or invalid API key"})
            return False
        return True

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health":
            return self._send(200, {"status": "ok"})
        if path == "/v1/knowledge":
            if self._authorized():
                self._send(200, self.service.library.list())
            return None
        if path in ("/v1/models", "/v1/limits"):
            if self._authorized():
                self._send(200, self.service.models() if path == "/v1/models" else self.service.limits())
            return None
        return self._send(404, {"detail": "Not Found"})

    def _knowledge(self, method):
        """PUT or DELETE /v1/knowledge/NAME."""
        name = self.path.split("?", 1)[0][len("/v1/knowledge/"):]
        body = None
        if method == "PUT":
            raw = self._read_body()
            if raw is None:
                return None
        if not self._authorized():
            return None
        try:
            if method == "DELETE":
                self.service.library.delete(name)
                return self._send(200, {"deleted": name})
            body = decode_body(raw, self.headers.get("Content-Encoding"), self.service.config.max_body_bytes)
            text = body.get("text") if isinstance(body, dict) else None
            return self._send(200, self.service.library.put(name, text))
        except kb.KnowledgeError as error:
            return self._send(error.status, {"detail": error.detail})
        except render.Unsupported as error:
            return self._send(422, {"detail": str(error)})
        except RequestError as error:
            return self._send(error.status, {"detail": error.detail}, error.headers)
        except BackendError as error:
            return self._send(error.status, {"detail": str(error)})

    def do_PUT(self):
        if not self.path.split("?", 1)[0].startswith("/v1/knowledge/"):
            return self._send(404, {"detail": "Not Found"}, close=True)
        return self._knowledge("PUT")

    def do_DELETE(self):
        if not self.path.split("?", 1)[0].startswith("/v1/knowledge/"):
            return self._send(404, {"detail": "Not Found"}, close=True)
        return self._knowledge("DELETE")

    def _read_body(self):
        """The request body, or None after answering 411 / 413."""
        limit = self.service.config.max_body_bytes
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = -1
        if length < 0:
            self._send(411, {"detail": "Content-Length is required"}, close=True)
            return None
        if length > limit:
            self._send(413, {"detail": f"the body is over {limit} bytes"}, close=True)
            return None
        return self.rfile.read(length)

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/v1/systemone":
            return self._send(404, {"detail": "Not Found"}, close=True)
        limit = self.service.config.max_body_bytes
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = -1
        if length < 0:
            return self._send(411, {"detail": "Content-Length is required"}, close=True)
        if length > limit:
            return self._send(413, {"detail": f"the body is over {limit} bytes"}, close=True)
        raw = self.rfile.read(length)
        if not self._authorized():
            return None
        try:
            body = decode_body(raw, self.headers.get("Content-Encoding"), limit)
            response, timings, passes, prefix, pack = self.service.systemone(body)
        except RequestError as error:
            return self._send(error.status, {"detail": error.detail}, error.headers)
        except BackendError as error:
            headers = {"Retry-After": "1"} if error.status in (503, 529) else None
            return self._send(error.status, {"detail": str(error)}, headers)
        except Exception as error:  # a server fault, never a client error
            self.log_error("internal error: %r", error)
            return self._send(500, {"detail": f"internal error: {type(error).__name__}"})
        timing = ", ".join(f"{k};dur={v:.3f}" for k, v in timings.items())
        headers = {"Server-Timing": timing, "X-SparkScore-Passes": str(passes), "X-SparkScore-Prefix-Tokens": str(prefix)}
        if pack:
            headers["Server-Timing"] += f", pack;desc={pack[0]}"
            headers["X-SparkScore-Pack-Tokens"] = str(pack[1])
        self._send(200, response, headers)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128  # the default of 5 refuses bursts of clients


def serve(service, host="127.0.0.1", port=8008, quiet=False):
    """A started server thread; call shutdown() to stop it."""
    server = _Server((host, port), Handler)
    server.service, server.quiet = service, quiet
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
    return server


def main(argv=None):
    from tools.sparkscore import templates

    defaults = Config()
    ap = argparse.ArgumentParser(prog="python3 -m tools.sparkscore.server", description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", choices=("echo", "score", "vllm"), required=True)
    ap.add_argument("--backend-url", default="http://127.0.0.1:8000", help="Atlas base URL")
    ap.add_argument("--backend-model", default="default", help="model field sent to /v1/completions (echo)")
    ap.add_argument("--backend-timeout", type=float, default=600.0)
    ap.add_argument("--no-topk", action="store_true", help="echo: always score options one prompt each")
    ap.add_argument("--tokenizer", help="HF id or local directory (default: the template's pinned tokenizer)")
    ap.add_argument("--revision", help="HF revision of --tokenizer")
    ap.add_argument("--template", default=templates.DEFAULT, help="chat-v1, knowline-gen1.5 or d3-ultra")
    ap.add_argument("--rotations", choices=("off", "first", "gated"),
                    help="knowline-gen1.5: option rotations to score (default off, as KnowLine serves)")
    ap.add_argument("--read-twice", choices=("on", "off"), help="knowline-gen1.5: read short requests twice (default on)")
    ap.add_argument("--permutation-average", choices=("on", "off"),
                    help="d3-ultra: also score choices with their options reversed (default off, as d3 serves)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--model-id", default=defaults.model_id)
    ap.add_argument("--alias", action="append", help=f"accepted model name (repeatable; default {defaults.aliases})")
    ap.add_argument("--temperature", default="1",
                    help="readout temperature on option logits (1 = off): one number for every type, "
                         "or per type, e.g. choice=1.45,noul=2.2,score=2.0 (tools.sparkscore.calibrate fits them); "
                         "it multiplies the template's own (d3-ultra: 1.046 for one question)")
    ap.add_argument("--per-question-context", type=int, default=defaults.per_question_context)
    ap.add_argument("--per-request-context", type=int, default=defaults.per_request_context)
    ap.add_argument("--max-body-bytes", type=int, default=defaults.max_body_bytes)
    ap.add_argument("--max-concurrent", type=int, default=defaults.max_concurrent)
    ap.add_argument("--knowledge", action="append", default=[], metavar="NAME=PATH",
                    help="a named knowledge document loaded and warmed at startup (repeatable)")
    ap.add_argument("--knowledge-dir", help="every *.md and *.txt file here, named by its file name, loaded at startup")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    try:
        temperature = parse_temperature(a.temperature)
    except ValueError as error:
        ap.error(f"--temperature: {error}")
    config = Config(
        model_id=a.model_id,
        aliases=a.alias if a.alias is not None else defaults.aliases,
        temperature=temperature,
        per_question_context=a.per_question_context,
        per_request_context=a.per_request_context,
        max_body_bytes=a.max_body_bytes,
        max_concurrent=a.max_concurrent,
        api_key=os.environ.get("SPARKSCORE_API_KEY") or None,
    )
    options = {}  # template options, only when given
    if a.rotations:
        options["rotations"] = a.rotations
    if a.read_twice:
        options["read_twice"] = a.read_twice == "on"
    if a.permutation_average:
        options["permutation_average"] = a.permutation_average == "on"
    try:
        renderer = render.load(a.template, a.tokenizer, a.revision, **options)
    except (TypeError, ValueError) as error:
        ap.error(str(error))
    attention = getattr(renderer, "attention", None)
    if attention and a.backend != "score":
        ap.error(f"template {a.template} needs {attention} attention, which only --backend score requests")
    if a.backend == "echo":
        backend = EchoBackend(a.backend_url, renderer.option_text, a.backend_model, a.backend_timeout, topk=not a.no_topk)
    elif a.backend == "vllm":
        backend = VllmBackend(a.backend_url, renderer.option_text, a.backend_model, a.backend_timeout)
    else:
        backend = ScoreBackend(a.backend_url, a.backend_timeout, attention)
    service = Service(renderer, backend, config)
    for doc in service.library.load(a.knowledge, a.knowledge_dir):
        print(f"knowledge {doc['name']}: {doc['chars']} chars, pack {doc['pack'] or 'none'} ({doc['pack_tokens']} tokens)", flush=True)
    server = serve(service, a.host, a.port, a.quiet)
    print(f"SparkScore on http://{a.host}:{a.port} ({config.model_id}, {renderer.version}, {a.backend} -> {a.backend_url})", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
