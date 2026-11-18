# SPDX-License-Identifier: AGPL-3.0-only
"""Image inputs: header sizes, d3-ultra's resize geometry, data URL checks, and the server's images path."""

import base64
import struct
import sys
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.sparkscore import images  # noqa: E402
from tools.sparkscore.render import Unsupported  # noqa: E402


def png(width, height):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IEND", b"")


def jpeg(width, height):
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + app0 + sof + b"\xff\xd9"


def webp_vp8x(width, height):
    body = b"VP8X" + struct.pack("<I", 10) + b"\x00\x00\x00\x00" + (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WEBP" + body


def url(kind, data):
    return f"data:image/{kind};base64," + base64.b64encode(data).decode()


class Headers(unittest.TestCase):
    def test_sizes_from_headers(self):
        for kind, make in (("png", png), ("jpeg", jpeg), ("webp", webp_vp8x)):
            (img,) = images.parse([url(kind, make(640, 427))])
            self.assertEqual((img.format, img.width, img.height), (kind, 640, 427))

    def test_declared_type_may_differ_from_the_bytes_but_the_bytes_decide(self):
        (img,) = images.parse([url("jpg", png(300, 200))])
        self.assertEqual(img.format, "png")

    def test_refusals(self):
        for bad, text in (
            ("not a list", "list"),
            (["/tmp/x.png"], "data:image"),
            ([url("gif", png(8, 8))], "png, jpeg or webp"),
            (["data:image/png;base64,@@@"], "base64"),
            ([url("png", b"not an image")], "not a PNG"),
            ([url("png", png(5000, 5000))], "pixels"),
            ([url("png", png(64, 64))] * (images.MAX_IMAGES + 1), "at most"),
        ):
            with self.assertRaises(ValueError) as caught:
                images.parse(bad)
            self.assertIn(text, str(caught.exception))


class Geometry(unittest.TestCase):
    # Pad counts d3-ultra's own processor gave on the gate images (results/2026-11-18-image-scoring).
    def test_pads_match_d3s_processor(self):
        g = images.Geometry()
        for (w, h), pads in {(224, 224): 64, (512, 384): 192, (1024, 576): 576, (1280, 720): 880,
                             (1600, 900): 1400, (640, 426): 260}.items():
            self.assertEqual(g.pads(w, h), pads, (w, h))

    def test_extreme_aspect_is_refused(self):
        with self.assertRaises(Unsupported):
            images.Geometry().pads(10000, 40)


class ImageRenderer:
    """Prefix = one token per image pad; records the images it was given."""

    tokenizer, version = "fake-tokenizer", "d3-ultra"

    def knowledge_head(self, knowledge):
        raise AssertionError("not used")

    def render(self, state, questions, knowledge=None, pack=False, images=None):
        from tools.sparkscore.render import Branch, Rendered
        from tools.sparkscore.templates import chat_v1

        geometry = globals()["images"].Geometry()
        pads = sum(geometry.pads(i.width, i.height) for i in images) if images else 0
        branches = [Branch(n, chat_v1.options(q)[0], [2, 3], [100, 101]) for n, q in questions.items()]
        return Rendered([7] * pads + [1], branches, images=list(images or []))


class ImageBackend:
    name, images = "fake-images", True

    def __init__(self):
        self.calls = []

    def score(self, prefix, branches, options, images=None, **_):
        self.calls.append((prefix, images))
        return [[0.75, 0.25] for _ in options], 1


class Server(unittest.TestCase):
    def serve(self, renderer, backend):
        from tools.sparkscore import server

        service = server.Service(renderer, backend, server.Config())
        self.http = server.serve(service, "127.0.0.1", 0, quiet=True)
        self.addCleanup(self.http.server_close)
        self.addCleanup(self.http.shutdown)
        return self.http.server_address[1]

    def post(self, port, body):
        import http.client
        import json

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        conn.request("POST", "/v1/systemone", json.dumps(body).encode(), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        payload = json.loads(resp.read())
        conn.close()
        return resp.status, payload

    BODY = {"model": "sparkscore-latest", "state": "", "questions": {"q": {"type": "noul", "instructions": "A cat?"}}}

    def test_images_reach_the_backend(self):
        backend = ImageBackend()
        port = self.serve(ImageRenderer(), backend)
        status, body = self.post(port, {**self.BODY, "images": [url("png", png(512, 384))]})
        self.assertEqual(status, 200, body)
        self.assertEqual(body["answers"]["q"]["noul"], 0.75)  # chat_v1 lists "true" first
        prefix, sent = backend.calls[0]
        self.assertEqual(len(prefix), 192 + 1)
        self.assertEqual([(i.width, i.height) for i in sent], [(512, 384)])

    def test_bad_images_are_422(self):
        port = self.serve(ImageRenderer(), ImageBackend())
        status, body = self.post(port, {**self.BODY, "images": ["/etc/passwd"]})
        self.assertEqual(status, 422, body)
        self.assertEqual(body["detail"][0]["loc"], ["body", "images"])

    def test_a_backend_without_images_refuses_them(self):
        backend = ImageBackend()
        backend.images = False
        port = self.serve(ImageRenderer(), backend)
        status, body = self.post(port, {**self.BODY, "images": [url("png", png(64, 64))]})
        self.assertEqual(status, 422, body)
        self.assertIn("images need", str(body["detail"]))

    def test_score_backend_sends_images_as_an_image_pack(self):
        from tools.sparkscore.backends import BackendError, ScoreBackend

        backend = ScoreBackend("http://127.0.0.1:1", attention="prefix_bidir")
        sent, replies = [], []

        def post(path, body):
            sent.append((path, body))
            reply = replies.pop(0) if replies else {"probs": [[1.0, 3.0]], "pack_id": "00000000000000aa", "pack": "built"}
            if isinstance(reply, Exception):
                raise reply
            return reply

        backend.http.post = post
        (img,) = images.parse([url("png", png(64, 64))])
        wire = [{"data_base64": base64.b64encode(img.data).decode(), "format": "png"}]
        rows, _ = backend.score([1, 2], [[3]], [[100, 101]], images=[img])
        self.assertEqual(rows, [[0.25, 0.75]])
        path, body = sent[-1]
        self.assertEqual(path, "/v1/score")
        # The state block is the pack, with its images; the prefix is empty.
        self.assertEqual((body["pack"], body["prefix"], body["images"]), ([1, 2], [], wire))
        self.assertEqual(body["attention"], "prefix_bidir")
        # The same state and image again: only the pack id.
        replies.append({"probs": [[1.0, 1.0]], "pack": "held"})
        report = {}
        backend.score([1, 2], [[4]], [[100, 101]], images=[img], report=report)
        self.assertEqual(sent[-1][1].get("pack_id"), "00000000000000aa")
        self.assertNotIn("images", sent[-1][1])
        self.assertEqual(report["pack"], "held")
        # Another image on the same tokens is another pack.
        (other,) = images.parse([url("png", png(64, 65))])
        backend.score([1, 2], [[4]], [[100, 101]], images=[other])
        self.assertEqual(sent[-1][1]["pack"], [1, 2])
        # A pack Atlas no longer holds: the tokens and images again.
        replies.extend([BackendError("gone", upstream=404), {"probs": [[1.0, 1.0]], "pack_id": "00000000000000bb"}])
        backend.score([1, 2], [[4]], [[100, 101]], images=[img])
        self.assertEqual((sent[-1][1]["pack"], sent[-1][1]["images"]), ([1, 2], wire))
        # An Atlas without image packs: the images in the prefix.
        backend.pack_ids.clear()
        replies.extend([BackendError("score: images cannot be combined with pack or pack_id", upstream=422),
                        {"probs": [[1.0, 1.0]]}])
        report = {}
        backend.score([1, 2], [[4]], [[100, 101]], images=[img], report=report)
        self.assertEqual((sent[-1][1]["prefix"], sent[-1][1]["images"]), ([1, 2], wire))
        self.assertNotIn("pack", sent[-1][1])
        self.assertEqual(report["pack"], "off")
        # Remembered: later image requests go straight to the prefix, one call each.
        before = len(sent)
        report = {}
        backend.score([1, 2], [[5]], [[100, 101]], images=[other], report=report)
        self.assertEqual(len(sent), before + 1)
        self.assertEqual((sent[-1][1]["prefix"], sent[-1][1]["images"][0]["format"]), ([1, 2], "png"))
        self.assertEqual(report["pack"], "off")

    def test_pack_ids_are_kept_under_fixed_size_keys(self):
        from tools.sparkscore.backends import ScoreBackend

        backend = ScoreBackend("http://127.0.0.1:1")
        backend.http.post = lambda path, body: {"probs": [[1.0, 1.0]], "pack_id": "00000000000000aa", "pack": "built"}
        (img,) = images.parse([url("png", png(64, 64))])
        backend.score(list(range(5000)), [[3]], [[100, 101]], images=[img])
        backend.score([1], [[3]], [[100, 101]], pack=list(range(5000)))
        self.assertEqual(len(backend.pack_ids), 2)
        self.assertTrue(all(isinstance(k, str) and len(k) == 64 for k in backend.pack_ids))

    def test_image_pack_status_reaches_server_timing(self):
        import http.client

        class PackBackend(ImageBackend):
            def score(self, prefix, branches, options, images=None, report=None, **_):
                report["pack"] = "held"
                return super().score(prefix, branches, options, images=images)

        port = self.serve(ImageRenderer(), PackBackend())
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        import json
        conn.request("POST", "/v1/systemone", json.dumps({**self.BODY, "images": [url("png", png(512, 384))]}).encode(),
                     {"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertEqual(resp.status, 200)
        self.assertIn("pack;desc=held", resp.getheader("Server-Timing"))
        self.assertEqual(resp.getheader("X-SparkScore-Pack-Tokens"), str(192 + 1))


if __name__ == "__main__":
    unittest.main()
