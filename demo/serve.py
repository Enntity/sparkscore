# SPDX-License-Identifier: AGPL-3.0-only
"""Serve the SparkScore demo page and proxy the API to a SparkScore server.

    python3 demo/serve.py [--port 8780] [--upstream http://127.0.0.1:8015]

Static files come from this directory; /v1/* and /health are forwarded to the
upstream unchanged (Server-Timing and X-SparkScore-* headers included).

The image demos spread their scoring calls over this machine's loopback names
(vision.js: a browser holds six connections a host), so a POST /v1/systemone
from one of those names on this port, and nothing else, gets CORS. Proxied
requests must name this server in Host (a page on another site rebound to
127.0.0.1 is refused) and send JSON bodies. demo/out/ (recordings) is never
served. Static files answer single byte ranges, so a video can seek.
"""

import argparse
import http.client
import os
import re
import urllib.parse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PASS_HEADERS = ("content-type", "server-timing", "retry-after", "x-sparkscore-passes", "x-sparkscore-prefix-tokens",
                "x-sparkscore-pack-tokens")
OUT = os.path.join(HERE, "out")
# The names a page reaches this server by (vision.js's HOSTS); --host adds its own.
LOOPBACK_NAMES = ("localhost", "s1.localhost", "s2.localhost", "127.0.0.1", "[::1]")


class _Slice:
    """`n` bytes of an open file, for copyfile."""

    def __init__(self, f, n):
        self.f, self.left = f, n

    def read(self, size=-1):
        size = self.left if size is None or size < 0 else min(size, self.left)
        data = self.f.read(size)
        self.left -= len(data)
        return data

    def close(self):
        self.f.close()


class Handler(SimpleHTTPRequestHandler):
    upstream = None  # urllib.parse.SplitResult
    names = LOOPBACK_NAMES

    def _proxied(self):
        return self.path.startswith("/v1/") or self.path == "/health"

    def _host_ok(self):
        host = (self.headers.get("Host") or "").lower()
        name = host[: host.index("]") + 1] if host.startswith("[") and "]" in host else host.split(":", 1)[0]
        return name in self.names

    def _cors_origin(self):
        """The Origin, if it is one of this server's names on its port and the call is the scoring one."""
        origin = self.headers.get("Origin") or ""
        port = self.server.server_address[1]
        ok = self.path.split("?", 1)[0] == "/v1/systemone" and origin in (f"http://{n}:{port}" for n in self.names)
        return origin if ok else None

    def _proxy(self, method):
        if not self._host_ok():
            return self.send_error(403, "unknown Host")
        if method in ("POST", "PUT") and not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return self.send_error(415, "JSON bodies only")
        body = None
        if method in ("POST", "PUT"):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        # A knowledge pack's first encoding or a handbook in the request can take seconds.
        conn = http.client.HTTPConnection(self.upstream.hostname, self.upstream.port, timeout=600)
        try:
            conn.request(method, self.path, body=body, headers={"Content-Type": self.headers.get("Content-Type", "application/json")})
            r = conn.getresponse()
            data = r.read()
        except OSError as error:
            data = f'{{"detail": "upstream unavailable: {type(error).__name__}"}}'.encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        finally:
            conn.close()
        self.send_response(r.status)
        for name in PASS_HEADERS:
            value = r.getheader(name)
            if value is not None:
                self.send_header(name, value)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self._proxied():
            return self._proxy("GET")
        return super().do_GET()

    def do_POST(self):
        if self._proxied():
            return self._proxy("POST")
        return self.send_error(404)

    def do_PUT(self):  # PUT /v1/knowledge/NAME
        if self._proxied():
            return self._proxy("PUT")
        return self.send_error(404)

    def do_DELETE(self):
        if self._proxied():
            return self._proxy("DELETE")
        return self.send_error(404)

    def send_head(self):
        """Static files, with one byte range when asked (`Range: bytes=a-b`, `a-`, `-n`)."""
        path = self.translate_path(self.path)
        # The translated path, so no spelling of the URL (%6Fut, ./out, x/../out), GET or HEAD, reaches
        # the recordings.
        if os.path.commonpath([os.path.realpath(path), OUT]) == OUT:
            self.send_error(404)
            return None
        spec = (self.headers.get("Range") or "").strip()
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", spec)
        if not m or m.groups() == ("", "") or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        first, last = m.groups()
        if first:
            start, end = int(first), min(int(last) if last else size - 1, size - 1)
        else:
            start, end = max(0, size - int(last)), size - 1
        if start >= size or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        return _Slice(f, end - start + 1)

    def do_OPTIONS(self):  # CORS preflight of the scoring call
        if not (self._host_ok() and self._cors_origin()):
            return self.send_error(404)
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "POST")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def end_headers(self):
        if not self._proxied():
            self.send_header("Cache-Control", "no-store")
            self.send_header("Accept-Ranges", "bytes")
        elif origin := self._cors_origin():
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Expose-Headers", ", ".join(PASS_HEADERS[1:]))
            self.send_header("Vary", "Origin")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64  # a page's calls arrive together (socketserver's default backlog is 5)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--upstream", default="http://127.0.0.1:8015")
    args = ap.parse_args()
    Handler.upstream = urllib.parse.urlsplit(args.upstream)
    if args.host not in ("0.0.0.0", "::", *LOOPBACK_NAMES):
        Handler.names = (*LOOPBACK_NAMES, args.host.lower())
    server = Server((args.host, args.port), partial(Handler, directory=HERE))
    print(f"SparkScore demo on http://{args.host}:{args.port}/ -> {args.upstream}")
    server.serve_forever()


if __name__ == "__main__":
    main()
