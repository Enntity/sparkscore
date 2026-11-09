# SPDX-License-Identifier: AGPL-3.0-only
"""Server request handling and the two Atlas backends, with a fake renderer
and a fake Atlas. No GPU, tokenizer or network needed."""

import gzip
import http.client
import json
import math
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.sparkscore import server  # noqa: E402
from tools.sparkscore.backends import EchoBackend, ScoreBackend  # noqa: E402
from tools.sparkscore.bench import run as bench  # noqa: E402
from tools.sparkscore.render import Branch, Rendered  # noqa: E402
from tools.sparkscore.templates import chat_v1  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "smoke.jsonl"


class FakeRenderer:
    """One token per character of state; option k is token 100 + k."""

    tokenizer, version = "fake-tokenizer", "chat-v1"

    def render(self, state, questions):
        prefix = [1] * (1 + len(chat_v1.text(state)))
        branches = []
        for name, q in questions.items():
            keys, _ = chat_v1.options(q)
            branches.append(Branch(name, keys, [2] * (4 + len(json.dumps(q))), [100 + k for k in range(len(keys))]))
        return Rendered(prefix, branches)


class FakeBackend:
    name = "fake"

    def __init__(self):
        self.calls = []
        self.block = None

    def score(self, prefix, branches, options):
        if self.block:
            self.block.wait()
        self.calls.append((prefix, branches, options))
        # Option k gets weight 2^-k, so the first option always wins.
        rows = [[2.0 ** -k for k in range(len(o))] for o in options]
        return [[w / sum(r) for w in r] for r in rows], 1


def post(port, body, headers=None, raw=None, path="/v1/systemone"):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    data = raw if raw is not None else json.dumps(body).encode()
    conn.request("POST", path, data, {"Content-Type": "application/json", **(headers or {})})
    resp = conn.getresponse()
    payload = json.loads(resp.read() or b"null")
    conn.close()
    return resp.status, payload, resp


def get(port, path, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request("GET", path, headers=headers or {})
    resp = conn.getresponse()
    payload = json.loads(resp.read())
    conn.close()
    return resp.status, payload


def request(**questions):
    return {"model": "sparkscore-latest", "state": "I was charged twice.", "questions": questions}


NOUL = {"type": "noul", "instructions": "Is this about billing?"}
CHOICE = {"type": "choice", "instructions": "Tone?", "criteria": {"angry": "Upset", "calm": None, "excited": "Eager"}}
SCORE = {"type": "score", "instructions": "Urgency?", "criteria": ["Can wait", "This week", "Today"]}


class ServerTest(unittest.TestCase):
    config = {}

    def setUp(self):
        self.backend = FakeBackend()
        self.service = server.Service(FakeRenderer(), self.backend, server.Config(**self.config))
        self.http = server.serve(self.service, "127.0.0.1", 0, quiet=True)
        self.port = self.http.server_address[1]

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()

    def assert_detail(self, status, payload, code, loc=None, text=None):
        self.assertEqual(status, code, payload)
        if loc is not None:
            self.assertIn(["body", *loc], [d["loc"] for d in payload["detail"]], payload)
        if text is not None:
            self.assertIn(text, json.dumps(payload["detail"]))


class Systemone(ServerTest):
    def test_answers_have_the_jev_shape(self):
        status, body, resp = post(self.port, request(billing=NOUL, tone=CHOICE, urgency=SCORE))
        self.assertEqual(status, 200, body)
        self.assertEqual(body["model"], "sparkscore-qwen3.6-35b-a3b")
        self.assertEqual(list(body["answers"]), ["billing", "tone", "urgency"])
        noul, choice, score = (body["answers"][k] for k in ("billing", "tone", "urgency"))
        self.assertEqual(noul, {"type": "noul", "noul": 2 / 3})
        self.assertEqual(choice["type"], "choice")
        self.assertEqual(choice["choice"], "angry")
        self.assertEqual(list(choice["probabilities"]), ["angry", "calm", "excited"])
        self.assertAlmostEqual(choice["confidence"], (4 / 7 - 1 / 3) / (2 / 3))
        self.assertEqual(score["legend"], {"0": "Can wait", "1": "This week", "2": "Today"})
        self.assertAlmostEqual(score["score"], (2 / 7) + 2 * (1 / 7))
        self.assertEqual(set(score), {"type", "score", "confidence", "legend", "probabilities"})
        self.assertEqual(body["usage"]["output_tokens"], 0)
        prefix, branches, options = self.backend.calls[0]
        self.assertEqual(body["usage"]["input_tokens"], len(prefix) + sum(len(b) for b in branches))
        self.assertEqual(options, [[100, 101], [100, 101, 102], [100, 101, 102]])
        timing = dict(p.strip().split(";dur=") for p in resp.getheader("Server-Timing").split(","))
        self.assertEqual(set(timing), {"render", "backend", "answer", "total"})
        self.assertEqual(resp.getheader("X-SparkScore-Passes"), "1")
        self.assertEqual(resp.getheader("X-SparkScore-Prefix-Tokens"), str(len(prefix)))

    def test_one_option_needs_no_backend(self):
        status, body, _ = post(self.port, request(only={"type": "choice", "criteria": {"x": None}}, one={"type": "score", "criteria": ["a"]}))
        self.assertEqual(status, 200, body)
        self.assertEqual(body["answers"]["only"], {"type": "choice", "choice": "x", "confidence": 1.0, "probabilities": {"x": 1.0}})
        self.assertEqual(body["answers"]["one"]["confidence"], 1.0)
        self.assertEqual(self.backend.calls, [])

    def test_structured_state_and_null_fields(self):
        q = {"type": "noul", "instructions": None, "criteria": {"true": {"rule": "spam"}, "false": None}}
        for state in ({"message": "hi"}, ["a", {"b": 1}], ""):
            status, body, _ = post(self.port, {**request(q=q), "state": state, "extra_field": 1})
            self.assertEqual(status, 200, body)

    def test_every_smoke_fixture_passes_the_harness_checks(self):
        for line in FIXTURES.read_text().splitlines():
            row = json.loads(line)
            status, body, _ = post(self.port, {"model": "jev-latest", "state": row["state"], "questions": row["questions"]})
            self.assertEqual(status, 200, (row["id"], body))
            for name, q in row["questions"].items():
                bench.check_answer(q, body["answers"][name])
            self.assertEqual(set(row["expected"]), set(row["questions"]))

    def test_validation_errors_are_422_with_locations(self):
        cases = [
            ({"state": "s", "questions": {"q": NOUL}}, ["model"], "Field required"),
            ({"model": "sparkscore-latest", "questions": {"q": NOUL}}, ["state"], "Field required"),
            ({**request(q=NOUL), "state": None}, ["state"], None),
            ({**request(), "questions": {}}, ["questions"], "at least 1"),
            ({**request(), "questions": []}, ["questions"], None),
            (request(q={"type": "maybe"}), ["questions", "q", "type"], None),
            (request(q={"type": "box", "instructions": "x"}), ["questions", "q", "type"], "vision"),
            (request(q={"type": "choice"}), ["questions", "q", "criteria"], "Field required"),
            (request(q={"type": "choice", "criteria": {}}), ["questions", "q", "criteria"], "options per choice"),
            (request(q={"type": "choice", "criteria": {str(i): None for i in range(256)}}), ["questions", "q", "criteria"], "options per choice"),
            (request(q={"type": "choice", "criteria": {"a": 3}}), ["questions", "q", "criteria", "a"], None),
            (request(q={"type": "score", "criteria": [str(i) for i in range(11)]}), ["questions", "q", "criteria"], "1 to 10 levels"),
            (request(q={"type": "score", "criteria": "abc"}), ["questions", "q", "criteria"], None),
            (request(q={"type": "noul", "criteria": {"true": 1}}), ["questions", "q", "criteria", "true"], None),
            (request(q={"type": "noul", "instructions": 5}), ["questions", "q", "instructions"], None),
            ({**request(q=NOUL), "assets": {"img": {"type": "image"}}}, ["assets"], "not supported"),
            ({**request(q=NOUL), "model": "gpt-9"}, ["model"], "unknown model"),
        ]
        for body, loc, text in cases:
            with self.subTest(loc=loc, text=text):
                status, payload, _ = post(self.port, body)
                self.assert_detail(status, payload, 422, loc, text)
        status, payload, _ = post(self.port, ["not", "an", "object"])
        self.assertEqual(status, 422)
        self.assertEqual(self.backend.calls, [])

    def test_too_many_questions_is_422(self):
        body = request(q=NOUL)
        body["questions"] = {f"q{i}": NOUL for i in range(257)}
        status, payload, _ = post(self.port, body)
        self.assert_detail(status, payload, 422, ["questions"], "at most 256 questions")
        self.assertEqual(self.backend.calls, [])

    def test_choice_at_the_limit_is_accepted(self):
        q = {"type": "choice", "criteria": {f"o{i}": None for i in range(255)}}
        status, body, _ = post(self.port, request(q=q))
        self.assertEqual(status, 200)
        self.assertEqual(len(body["answers"]["q"]["probabilities"]), 255)

    def test_request_rules_are_400(self):
        good = json.dumps(request(q=NOUL)).encode()
        for raw, text in [
            (b"\xef\xbb\xbf" + good, "byte order mark"),
            (b'{"model": "x", "state": "\xff"}', "UTF-8"),
            (b"{not json", "invalid JSON"),
            (json.dumps({**request(q=NOUL), "n": 1}).encode().replace(b'"n": 1', b'"n": NaN'), "finite"),
            (json.dumps({**request(q=NOUL), "n": 1}).encode().replace(b'"n": 1', b'"n": 1e400'), "finite"),
        ]:
            with self.subTest(text=text):
                status, payload, _ = post(self.port, None, raw=raw)
                self.assert_detail(status, payload, 400, text=text)

    def test_nesting_limit(self):
        deep = "x"
        for _ in range(65):
            deep = [deep]
        status, payload, _ = post(self.port, {**request(q=NOUL), "state": deep})
        self.assert_detail(status, payload, 400, text="nested deeper than 64")
        status, _, _ = post(self.port, {**request(q=NOUL), "state": deep[0]})
        self.assertEqual(status, 200)
        self.assertEqual([server.nesting(v) for v in ("x", [], {"a": 1}, [[{}]], deep)], [0, 1, 1, 3, 65])
        status, payload, _ = post(self.port, None, raw=b"[" * 100000 + b"]" * 100000)
        self.assertEqual(status, 400)

    def test_content_encoding(self):
        good = json.dumps(request(q=NOUL)).encode()
        status, _, _ = post(self.port, None, raw=gzip.compress(good), headers={"Content-Encoding": "gzip"})
        self.assertEqual(status, 200)
        status, _, _ = post(self.port, None, raw=good, headers={"Content-Encoding": "identity"})
        self.assertEqual(status, 200)
        status, _, _ = post(self.port, None, raw=b"not gzip", headers={"Content-Encoding": "gzip"})
        self.assertEqual(status, 400)
        status, _, _ = post(self.port, None, raw=good, headers={"Content-Encoding": "br"})
        self.assertEqual(status, 415)
        status, _, _ = post(self.port, None, raw=good, headers={"Content-Encoding": "gzip, gzip"})
        self.assertEqual(status, 415)
        packed = server.zstd.compress(good) if server.zstd else good
        status, _, _ = post(self.port, None, raw=packed, headers={"Content-Encoding": "zstd"})
        self.assertEqual(status, 200 if server.zstd else 415)

    def test_context_limit_is_400_max_tokens_exceeded(self):
        status, payload, _ = post(self.port, {**request(q=NOUL), "state": "x" * 40000})
        self.assertEqual(status, 400)
        self.assertEqual(payload["detail"]["type"], "max_tokens_exceeded")
        self.assertIn("maximum context length", payload["detail"]["msg"])  # Decision Index's refusal marker
        self.assertEqual(self.backend.calls, [])

    def test_other_routes(self):
        status, body = get(self.port, "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual([m["name"] for m in body["models"]], ["sparkscore-qwen3.6-35b-a3b", "sparkscore-latest", "jev-latest"])
        self.assertEqual(set(body["models"][0]), {"name", "description", "release_date"})
        status, body = get(self.port, "/v1/limits")
        self.assertEqual((status, body["max_choice_options"], body["max_score_levels"]), (200, 255, 10))
        self.assertEqual(get(self.port, "/health"), (200, {"status": "ok"}))
        self.assertEqual(get(self.port, "/nope")[0], 404)
        self.assertEqual(post(self.port, request(q=NOUL), path="/v1/other")[0], 404)

    def test_capacity_is_529(self):
        self.backend.block = threading.Event()
        threads = [threading.Thread(target=post, args=(self.port, request(q=NOUL))) for _ in range(self.service.config.max_concurrent)]
        for t in threads:
            t.start()
        try:
            for _ in range(200):
                if self.service.slots._value == 0:
                    break
                threading.Event().wait(0.01)
            status, payload, resp = post(self.port, request(q=NOUL))
            self.assertEqual(status, 529)
            self.assertEqual(resp.getheader("Retry-After"), "1")
        finally:
            self.backend.block.set()
            for t in threads:
                t.join()


class SmallLimits(ServerTest):
    config = {"max_body_bytes": 2000, "max_concurrent": 1, "api_key": "k3y", "per_request_context": 500}

    def test_body_limit_is_413(self):
        auth = {"Authorization": "Bearer k3y"}
        status, _, _ = post(self.port, {**request(q=NOUL), "state": "x" * 3000}, headers=auth)
        self.assertEqual(status, 413)
        bomb = gzip.compress(json.dumps({**request(q=NOUL), "state": "x" * 100000}).encode())
        self.assertLess(len(bomb), 2000)
        status, _, _ = post(self.port, None, raw=bomb, headers={**auth, "Content-Encoding": "gzip"})
        self.assertEqual(status, 413)

    def test_bearer_key(self):
        self.assertEqual(post(self.port, request(q=NOUL))[0], 401)
        self.assertEqual(post(self.port, request(q=NOUL), headers={"Authorization": "Bearer nope"})[0], 401)
        self.assertEqual(post(self.port, request(q=NOUL), headers={"Authorization": "Bearer k3y"})[0], 200)
        self.assertEqual(get(self.port, "/v1/models")[0], 401)
        self.assertEqual(get(self.port, "/health")[0], 200)

    def test_request_context_limit(self):
        status, payload, _ = post(self.port, {**request(a=NOUL, b=NOUL, c=NOUL), "state": "x" * 400}, headers={"Authorization": "Bearer k3y"})
        self.assertEqual(status, 400)
        self.assertEqual(payload["detail"]["type"], "max_tokens_exceeded")


# ------------------------------------------------------------------ backends against a fake Atlas
LABELS = {100 + k: f" L{k}" for k in range(40)}


def option_logprob(token):
    return -0.5 * (token - 100) - 0.1


class FakeAtlas(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen = []
    hide = set()  # option tokens left out of the top logprobs
    held = set()  # knowledge packs held (ids)
    packs_off = False

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeAtlas.seen.append((self.path, body))
        if self.path == "/v1/completions":
            assert body["echo"] is True and body["max_tokens"] == 0 and body["logprobs"] == 20
            choices = []
            for i, prompt in enumerate(body["prompt"]):
                # The slot distribution ranks options by token id; the top 20 are returned.
                top = {LABELS[t]: option_logprob(t) for t in sorted(LABELS)[:20] if t not in FakeAtlas.hide}
                n = len(prompt)
                choices.append({"index": i, "text": "", "logprobs": {
                    "tokens": ["t"] * n,
                    "token_logprobs": [None] * (n - 1) + [option_logprob(prompt[-1])],
                    "top_logprobs": [None] * (n - 1) + [top],
                    "text_offset": list(range(n))}})
            payload = {"choices": choices[::-1]}  # order must come from "index"
        elif self.path == "/v1/score":
            payload = {"probs": [[option_logprob(t) and math.exp(option_logprob(t)) for t in row] for row in body["options"]]}
            if "pack" in body or "pack_id" in body:
                if FakeAtlas.packs_off:
                    msg = "pack: knowledge packs are off on this server (start it with --score-packs N)"
                    return self.reply(422, {"error": {"message": msg}})
                pid = body.get("pack_id") or f"{sum(body['pack']) * 7919:016x}"
                if "pack_id" in body and pid not in FakeAtlas.held:
                    return self.reply(404, {"error": {"message": f"pack_id {pid} is not held"}})
                payload["pack"] = "held" if pid in FakeAtlas.held else "built"
                payload["pack_id"] = pid
                FakeAtlas.held.add(pid)
        else:
            payload = {}
        self.reply(200, payload)

    def reply(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def expected(options):
    weights = [math.exp(option_logprob(t)) for t in options]
    return [w / sum(weights) for w in weights]


class Backends(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.atlas = ThreadingHTTPServer(("127.0.0.1", 0), FakeAtlas)
        threading.Thread(target=cls.atlas.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.atlas.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.atlas.shutdown()
        cls.atlas.server_close()

    def setUp(self):
        FakeAtlas.seen, FakeAtlas.hide, FakeAtlas.held, FakeAtlas.packs_off = [], set(), set(), False

    def check(self, probs, options):
        for row, o in zip(probs, options):
            for got, want in zip(row, expected(o)):
                self.assertAlmostEqual(got, want, places=6)

    def test_echo_topk_shortcut(self):
        options = [[100, 101], [100, 101, 102, 103]]
        probs, passes = EchoBackend(self.url, LABELS).score([1, 2], [[3], [4, 5]], options)
        self.check(probs, options)
        self.assertEqual(passes, 2)  # one prompt per question
        prompts = FakeAtlas.seen[0][1]["prompt"]
        self.assertEqual(prompts, [[1, 2, 3, 100], [1, 2, 4, 5, 100]])

    def test_echo_falls_back_per_option(self):
        FakeAtlas.hide = {102}
        many = list(range(100, 125))  # more options than top logprobs
        options = [[100, 101], [100, 101, 102], many]
        probs, passes = EchoBackend(self.url, LABELS).score([1], [[3], [4], [5]], options)
        self.check(probs, options)
        # shortcut: 2 prompts (K <= 20); fallback: options 1..2 of branch 1, all 25 of branch 2
        self.assertEqual(passes, 2 + 2 + 25)
        fallback = FakeAtlas.seen[1][1]["prompt"]
        self.assertEqual(fallback[:2], [[1, 4, 101], [1, 4, 102]])
        self.assertEqual(fallback[2], [1, 5, 100])

    def test_echo_without_shortcut(self):
        options = [[100, 101, 102]]
        probs, passes = EchoBackend(self.url, LABELS, topk=False).score([1], [[3]], options)
        self.check(probs, options)
        self.assertEqual(passes, 3)
        self.assertEqual(len(FakeAtlas.seen), 1)

    def test_score_backend(self):
        options = [[100, 101], [100, 101, 102]]
        probs, passes = ScoreBackend(self.url).score([1, 2], [[3], [4]], options)
        self.check(probs, options)
        self.assertEqual(passes, 1)
        self.assertEqual(FakeAtlas.seen[0], ("/v1/score", {"prefix": [1, 2], "branches": [[3], [4]], "options": options}))

    def test_score_backend_names_the_mask(self):
        """Only a template that needs another mask (d3-ultra) sends one."""
        options = [[100, 101]]
        probs, _ = ScoreBackend(self.url, attention="prefix_bidir").score([1, 2], [[3]], options)
        self.check(probs, options)
        self.assertEqual(FakeAtlas.seen[0][1], {"prefix": [1, 2], "branches": [[3]], "options": options, "attention": "prefix_bidir"})

    def test_score_backend_sends_a_pack_once_then_its_id(self):
        backend, options = ScoreBackend(self.url, attention="prefix_bidir"), [[100, 101]]
        first, second = {}, {}
        probs, _ = backend.score([1, 2], [[3]], options, pack=[9, 9, 9], report=first)
        self.check(probs, options)
        backend.score([1, 5], [[3]], options, pack=[9, 9, 9], report=second)
        self.assertEqual(FakeAtlas.seen[0][1]["pack"], [9, 9, 9])
        self.assertNotIn("pack_id", FakeAtlas.seen[0][1])
        self.assertEqual(FakeAtlas.seen[1][1]["pack_id"], f"{27 * 7919:016x}")
        self.assertNotIn("pack", FakeAtlas.seen[1][1])
        self.assertEqual(FakeAtlas.seen[1][1]["prefix"], [1, 5])
        self.assertEqual((first["pack"], second["pack"]), ("built", "held"))

    def test_score_backend_resends_a_pack_atlas_no_longer_holds(self):
        backend, options, report = ScoreBackend(self.url), [[100, 101]], {}
        backend.score([1], [[3]], options, pack=[4, 4])
        FakeAtlas.held.clear()  # evicted
        probs, _ = backend.score([1], [[3]], options, pack=[4, 4], report=report)
        self.check(probs, options)
        self.assertEqual([("pack_id" in b, "pack" in b) for _, b in FakeAtlas.seen], [(False, True), (True, False), (False, True)])
        self.assertEqual(report["pack"], "built")

    def test_score_backend_without_packs_reads_the_pack_as_prefix(self):
        FakeAtlas.packs_off = True
        backend, options, report = ScoreBackend(self.url), [[100, 101]], {}
        probs, _ = backend.score([1, 2], [[3]], options, pack=[7, 8], report=report)
        self.check(probs, options)
        self.assertEqual(FakeAtlas.seen[-1][1], {"prefix": [7, 8, 1, 2], "branches": [[3]], "options": options})
        self.assertEqual(report["pack"], "off")


class Knowledge(ServerTest):
    def test_a_template_without_knowledge_refuses_it(self):
        status, body, _ = post(self.port, dict(request(refund=NOUL), knowledge="Refunds within 30 days."))
        self.assertEqual(status, 422)
        self.assertIn("d3-ultra template only", json.dumps(body))
        self.assertEqual(self.backend.calls, [])

    def test_knowledge_must_be_a_nonempty_string(self):
        for bad in ("", " \n ", 5, ["x"]):
            status, body, _ = post(self.port, dict(request(refund=NOUL), knowledge=bad))
            self.assertEqual(status, 422)
            self.assertEqual(body["detail"][0]["loc"], ["body", "knowledge"])


class PackRenderer(FakeRenderer):
    """FakeRenderer with knowledge: the knowledge is a pack of one token per character."""

    knowledge_head = True

    def render(self, state, questions, knowledge=None, pack=False):
        out = super().render(state, questions)
        if not knowledge:
            return out
        tokens = [7] * len(knowledge)
        return Rendered(out.prefix, out.branches, pack=tokens) if pack else Rendered(tokens + out.prefix, out.branches)


class PackBackend(FakeBackend):
    packs = True

    def score(self, prefix, branches, options, pack=None, report=None):
        self.calls.append((pack, prefix, branches, options))
        if report is not None:
            report["pack"] = "held" if any(c[0] == pack for c in self.calls[:-1]) else "built"
        rows = [[2.0 ** -k for k in range(len(o))] for o in options]
        return [[w / sum(r) for w in r] for r in rows], 1


def send(port, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    data = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, data, {"Content-Type": "application/json"} if data else {})
    resp = conn.getresponse()
    payload = json.loads(resp.read() or b"null")
    conn.close()
    return resp.status, payload, resp


class NamedKnowledge(ServerTest):
    def setUp(self):
        self.backend = PackBackend()
        self.service = server.Service(PackRenderer(), self.backend, server.Config(**self.config))
        self.http = server.serve(self.service, "127.0.0.1", 0, quiet=True)
        self.port = self.http.server_address[1]

    def test_a_registered_document_is_warmed_and_used_by_name(self):
        status, body, _ = send(self.port, "PUT", "/v1/knowledge/acme", {"text": "Helios goes to Infrastructure."})
        self.assertEqual(status, 200, body)
        self.assertEqual((body["name"], body["pack"], body["pack_tokens"]), ("acme", "built", 30))
        self.assertEqual(len(self.backend.calls), 1)  # the warm-up
        status, listing, _ = send(self.port, "GET", "/v1/knowledge")
        self.assertEqual([d["name"] for d in listing["knowledge"]], ["acme"])
        status, out, resp = post(self.port, dict(request(refund=NOUL), knowledge_name="acme"))
        self.assertEqual(status, 200, out)
        self.assertEqual(self.backend.calls[-1][0], [7] * 30)
        self.assertIn("pack;desc=held", resp.getheader("Server-Timing"))
        self.assertEqual(resp.getheader("X-SparkScore-Pack-Tokens"), "30")

    def test_unknown_names_and_both_fields_are_refused(self):
        status, body, _ = post(self.port, dict(request(refund=NOUL), knowledge_name="nope"))
        self.assert_detail(status, body, 422, ["knowledge_name"], "no knowledge named")
        status, body, _ = post(self.port, dict(request(refund=NOUL), knowledge="x", knowledge_name="nope"))
        self.assert_detail(status, body, 422, ["knowledge_name"], "not both")

    def test_bad_names_and_texts_are_refused_and_delete_works(self):
        self.assertEqual(send(self.port, "PUT", "/v1/knowledge/-bad", {"text": "x"})[0], 422)
        self.assertEqual(send(self.port, "PUT", "/v1/knowledge/ok", {"text": "  "})[0], 422)
        self.assertEqual(send(self.port, "PUT", "/v1/knowledge/ok", {"text": "Policy."})[0], 200)
        self.assertEqual(send(self.port, "DELETE", "/v1/knowledge/ok")[0], 200)
        self.assertEqual(send(self.port, "DELETE", "/v1/knowledge/ok")[0], 404)

    def test_a_server_fault_while_registering_answers_500(self):
        def broken(*args, **kwargs):
            raise ValueError("a template invariant broke")

        self.service.renderer.render = broken
        status, body, _ = send(self.port, "PUT", "/v1/knowledge/acme", {"text": "Policy."})
        self.assertEqual(status, 500, body)
        self.assertIn("internal error", body["detail"])

    def test_startup_documents_load_from_a_directory_and_specs(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            Path(d, "handbook.md").write_text("Handbook.")
            Path(d, "notes.json").write_text("{}")  # not a document
            extra = Path(d, "x.txt")
            extra.write_text("Extra policy.")
            loaded = self.service.library.load([f"policy={extra}"], d)
        self.assertEqual(sorted(doc["name"] for doc in loaded), ["handbook", "policy", "x"])
        with self.assertRaises(ValueError):
            self.service.library.load(["no-equals-sign"])


class KnowledgeWithoutPacks(ServerTest):
    def test_a_template_without_knowledge_cannot_register_it(self):
        status, body, _ = send(self.port, "PUT", "/v1/knowledge/acme", {"text": "Policy."})
        self.assertEqual(status, 422)
        self.assertIn("d3-ultra template only", json.dumps(body))


if __name__ == "__main__":
    unittest.main()
