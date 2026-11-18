# SPDX-License-Identifier: AGPL-3.0-only
"""demo/serve.py: the image demos' proxy answers CORS only for its own loopback names and port, refuses
requests addressed to another host, takes JSON bodies only, and never serves demo/out/."""

import http.client
import sys
import threading
import unittest
from functools import partial
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(DEMO))

import serve  # noqa: E402


class Proxy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import urllib.parse

        serve.Handler.upstream = urllib.parse.urlsplit("http://127.0.0.1:1")  # nothing there: proxied calls 502
        cls.server = serve.Server(("127.0.0.1", 0), partial(serve.Handler, directory=str(DEMO)))
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, method, path, headers=None, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        resp.read()
        conn.close()
        return resp

    def preflight(self, origin, path="/v1/systemone"):
        return self.request("OPTIONS", path, {"Origin": origin, "Access-Control-Request-Method": "POST"})

    def test_cors_only_for_its_own_loopback_names_and_port(self):
        for name in ("localhost", "s1.localhost", "s2.localhost", "127.0.0.1", "[::1]"):
            origin = f"http://{name}:{self.port}"
            resp = self.preflight(origin)
            self.assertEqual(resp.status, 204, origin)
            self.assertEqual(resp.getheader("Access-Control-Allow-Origin"), origin)
            self.assertEqual(resp.getheader("Access-Control-Allow-Methods"), "POST")
        for origin in (f"http://localhost:{self.port + 1}", f"http://evil.localhost:{self.port}", "http://localhost",
                       f"https://localhost:{self.port}", f"http://localhost.evil.com:{self.port}", "null"):
            resp = self.preflight(origin)
            self.assertEqual(resp.status, 404, origin)
            self.assertIsNone(resp.getheader("Access-Control-Allow-Origin"), origin)
        # Only the scoring call is shared across the names.
        self.assertEqual(self.preflight(f"http://s1.localhost:{self.port}", "/v1/knowledge/x").status, 404)

    def test_a_proxied_answer_carries_cors_for_an_allowed_origin_only(self):
        json = {"Content-Type": "application/json"}
        ok = self.request("POST", "/v1/systemone", {**json, "Origin": f"http://s2.localhost:{self.port}"}, b"{}")
        self.assertEqual(ok.status, 502)  # proxied (nothing upstream)
        self.assertEqual(ok.getheader("Access-Control-Allow-Origin"), f"http://s2.localhost:{self.port}")
        self.assertIn("server-timing", ok.getheader("Access-Control-Expose-Headers"))
        other = self.request("POST", "/v1/systemone", {**json, "Origin": "https://evil.example"}, b"{}")
        self.assertIsNone(other.getheader("Access-Control-Allow-Origin"))

    def test_requests_for_another_host_are_refused(self):
        resp = self.request("POST", "/v1/systemone", {"Host": "evil.example", "Content-Type": "application/json"}, b"{}")
        self.assertEqual(resp.status, 403)
        self.assertEqual(self.request("GET", "/health", {"Host": f"rebound.example:{self.port}"}).status, 403)

    def test_proxied_bodies_must_be_json(self):
        resp = self.request("POST", "/v1/systemone", {"Content-Type": "text/plain"}, b"{}")
        self.assertEqual(resp.status, 415)

    def test_static_files_answer_byte_ranges(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", "/vision.css", headers={"Range": "bytes=10-19"})
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        whole = (DEMO / "vision.css").read_bytes()
        self.assertEqual(resp.status, 206)
        self.assertEqual(body, whole[10:20])
        self.assertEqual(resp.getheader("Content-Range"), f"bytes 10-19/{len(whole)}")
        self.assertEqual(resp.getheader("Accept-Ranges"), "bytes")
        # An open end, and a range past the end.
        self.assertEqual(self.request("GET", "/vision.css", {"Range": f"bytes={len(whole) - 5}-"}).status, 206)
        self.assertEqual(self.request("GET", "/vision.css", {"Range": f"bytes={len(whole) + 5}-"}).status, 416)

    def test_out_is_never_served(self):
        out = DEMO / "out"
        out.mkdir(exist_ok=True)
        probe = out / "test-probe.txt"
        probe.write_text("recording")
        self.addCleanup(probe.unlink)
        for path in ("/out/test-probe.txt", "/%6Fut/test-probe.txt", "/./out/test-probe.txt", "/photos/../out/test-probe.txt"):
            self.assertEqual(self.request("GET", path).status, 404, path)
            self.assertEqual(self.request("HEAD", path).status, 404, path)


if __name__ == "__main__":
    unittest.main()
