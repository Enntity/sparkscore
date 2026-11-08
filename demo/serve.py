# SPDX-License-Identifier: AGPL-3.0-only
"""Serve the SparkScore demo page and proxy the API to a SparkScore server.

    python3 demo/serve.py [--port 8780] [--upstream http://127.0.0.1:8015]

Static files come from this directory; /v1/* and /health are forwarded to the
upstream unchanged (Server-Timing and X-SparkScore-* headers included), so the
page is same-origin and needs no CORS.
"""

import argparse
import http.client
import os
import urllib.parse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PASS_HEADERS = ("content-type", "server-timing", "retry-after", "x-sparkscore-passes", "x-sparkscore-prefix-tokens",
                "x-sparkscore-pack-tokens")


class Handler(SimpleHTTPRequestHandler):
    upstream = None  # urllib.parse.SplitResult

    def _proxied(self):
        return self.path.startswith("/v1/") or self.path == "/health"

    def _proxy(self, method):
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
        if self.path.split("?", 1)[0].startswith("/out/"):
            return self.send_error(404)
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

    def end_headers(self):
        if not self._proxied():
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--upstream", default="http://127.0.0.1:8015")
    args = ap.parse_args()
    Handler.upstream = urllib.parse.urlsplit(args.upstream)
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, directory=HERE))
    server.daemon_threads = True
    print(f"SparkScore demo on http://{args.host}:{args.port}/ -> {args.upstream}")
    server.serve_forever()


if __name__ == "__main__":
    main()
