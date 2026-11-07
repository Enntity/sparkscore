# SPDX-License-Identifier: AGPL-3.0-only
"""Engine time of one /v1/score pass: prefix length x branch count (median of 5).

    engine_sweep.py [ATLAS_URL] [LENGTHS] [BRANCH_COUNTS]

Each branch is 32 tokens. `rows` is the pass's row count and `engine_ms` the
engine's own time for it. The JevBench median request is about 288 rows
(L=256, N=1). Run it inside the container, where Atlas listens on 8081:

    docker exec sparkscore /opt/venv/bin/python tools/sparkscore/bench/engine_sweep.py http://127.0.0.1:8081 64,256,1024,4096 1
"""
import json, statistics, sys, time, urllib.request
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8081"
LENS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "64,256,1024,4096").split(",")]
NS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,4,16,64").split(",")]
BR, REPS = 32, 5
def post(path, body):
    r = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as f:
        return json.loads(f.read())
base = list(range(1000, 1000 + 8192))
print(f"{'L':>6} {'N':>4} {'rows':>6} {'engine_ms':>10} {'wall_ms':>8} {'min_ms':>7}")
for L in LENS:
    for N in NS:
        prefix = base[:L]
        branches = [[2000 + 37 * i + j for j in range(BR)] for i in range(N)]
        body = {"prefix": prefix, "branches": branches, "options": [32, 33, 34, 35], "attention": "prefix_bidir"}
        post("/v1/score", body)  # warm this shape
        eng, wall = [], []
        for _ in range(REPS):
            t = time.time(); r = post("/v1/score", body); wall.append((time.time() - t) * 1e3)
            eng.append(r["engine_us"] / 1e3)
        rows = (L // 16) * 16 + N * (L % 16 + BR)
        print(f"{L:>6} {N:>4} {rows:>6} {statistics.median(eng):>10.1f} {statistics.median(wall):>8.1f} {min(eng):>7.1f}", flush=True)
