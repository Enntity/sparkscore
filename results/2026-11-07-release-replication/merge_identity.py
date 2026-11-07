# SPDX-License-Identifier: AGPL-3.0-only
"""Merged vs solo bit identity across pass shapes: short (<32 rows), medium and
long (chunk >= 256) requests, alone and racing each other, causal and
prefix_bidir. Also probes the endpoint's refusals.

usage: merge_identity.py [ATLAS_URL]
"""
import json, sys, threading, urllib.error, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8081"


def post(body, timeout=600):
    r = urllib.request.Request(URL + "/v1/score", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as f:
            return 200, json.loads(f.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]


def req(seed, plen, nb, blen, mode):
    prefix = [1000 + (seed * 7919 + j) % 50000 for j in range(plen)]
    branches = [[2000 + 31 * q + seed + j for j in range(blen)] for q in range(nb)]
    b = {"prefix": prefix, "branches": branches, "options": [32, 33, 34, 35]}
    if mode:
        b["attention"] = mode
    return b


SHAPES = {"tiny": (10, 1, 3), "short": (20, 2, 5), "mid": (120, 3, 20), "long": (600, 2, 40), "xlong": (2000, 4, 60)}
worst, merged_seen, cases = 0.0, 0, 0
for mode in (None, "prefix_bidir"):
    solo = {}
    for i, (name, (p, n, b)) in enumerate(SHAPES.items()):
        st, out = post(req(i, p, n, b, mode))
        assert st == 200, (name, st, out)
        solo[name] = out["probs"]
    names = list(SHAPES)
    for rep in range(6):
        res = {}
        def go(name, i):
            res[name] = post(req(i, *SHAPES[name], mode))
        th = [threading.Thread(target=go, args=(n, i)) for i, n in enumerate(names)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        for name, (st, out) in res.items():
            assert st == 200, (name, st, out)
            cases += 1
            merged_seen += out.get("batched", 1) > 1
            d = max(abs(a - b) for pa, pb in zip(out["probs"], solo[name]) for a, b in zip(pa, pb))
            worst = max(worst, d)
            if d:
                print(f"DIFF mode={mode} rep={rep} {name} batched={out.get('batched')} max|dp|={d:.3e}")
print(f"merge identity: {cases} results, {merged_seen} merged, worst |dp| vs solo {worst:.3e}")

# Refusals.
print("bad token id:", post({"prefix": [1, 2, 3], "branches": [[4]], "options": [32, 10**9]})[0])
print("prefix token past vocab:", post({"prefix": [10**8], "branches": [[4]], "options": [32, 33]})[0])
print("too many branches:", post({"prefix": [1] * 10, "branches": [[4]] * 257, "options": [32, 33]})[0])
print("branch too long for a pass:", post({"prefix": [1] * 10, "branches": [[5] * 60000], "options": [32, 33]})[0])
print("unknown attention:", post({"prefix": [1] * 10, "branches": [[4]], "options": [32, 33], "attention": "x"})[0])
# Backlog: 80 concurrent tiny requests; past 64 waiting some may get 503.
codes = []
def burst():
    codes.append(post(req(7, 10, 1, 3, None))[0])
th = [threading.Thread(target=burst) for _ in range(80)]
for t in th:
    t.start()
for t in th:
    t.join()
print("burst of 80:", {c: codes.count(c) for c in set(codes)})
