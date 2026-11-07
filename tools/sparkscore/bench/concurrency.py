#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""K concurrent /v1/score requests: latency, merge count, and agreement with solo runs.

usage: concurrency.py URL [K...]

Each request is a distinct 300-token prefix with 4 branches of 40 tokens.
`batched` is how many requests the engine merged into one pass, and the last
column the largest probability difference from the same request run alone
(0 means merged and solo passes agree exactly). Inside the container:

    docker exec sparkscore /opt/venv/bin/python tools/sparkscore/bench/concurrency.py http://127.0.0.1:8081 1 2 4 8 16
"""
import json, statistics, sys, threading, time, urllib.request

URL = sys.argv[1]
KS = [int(k) for k in sys.argv[2:]] or [1, 2, 4, 8, 16]


def post(body):
    r = urllib.request.Request(URL + "/v1/score", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(r, timeout=600) as f:
        out = json.loads(f.read())
    out["wall_ms"] = (time.time() - t) * 1e3
    return out


def body(i):
    # Distinct prefixes (no sharing across requests), 4 questions each.
    prefix = [1000 + (i * 7919 + j) % 50000 for j in range(300)]
    return {"prefix": prefix, "branches": [[2000 + 31 * q + i + j for j in range(40)] for q in range(4)],
            "options": [32, 33, 34, 35]}


solo = {i: post(body(i))["probs"] for i in range(max(KS))}
for i in range(max(KS)):
    post(body(i))  # warm every shape once
print(f"{'K':>3} {'wall_p50':>9} {'wall_max':>9} {'engine_p50':>10} {'batched':>8} {'max|dp| vs solo':>16}")
for k in KS:
    res = [None] * k
    def go(i):
        res[i] = post(body(i))
    th = [threading.Thread(target=go, args=(i,)) for i in range(k)]
    t0 = time.time()
    for t in th:
        t.start()
    for t in th:
        t.join()
    total = (time.time() - t0) * 1e3
    walls = [r["wall_ms"] for r in res]
    dp = max(abs(a - b) for i, r in enumerate(res) for pa, pb in zip(r["probs"], solo[i]) for a, b in zip(pa, pb))
    print(f"{k:>3} {statistics.median(walls):>9.1f} {max(walls):>9.1f} {statistics.median(r['engine_us'] / 1e3 for r in res):>10.1f} "
          f"{max(r.get('batched', 1) for r in res):>8} {dp:>16.4f}   (all {k} done in {total:.0f} ms)")
