# SPDX-License-Identifier: AGPL-3.0-only
"""Requests on two different packs, merged into shared passes, score exactly as alone.

    python3 mixed_packs.py REFERENCE.jsonl http://127.0.0.1:8015

Alternates 12 handbook and 12 rules4 requests (each with its own `knowledge`), runs them one at a time,
then eight in flight, and prints the largest probability difference.
"""
import json, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

ROWS = [json.loads(l) for l in open(sys.argv[1])]
URL = sys.argv[2] + "/v1/systemone"
by = {}
for r in ROWS:
    by.setdefault(r["set"], []).append(r)
mix = [x for pair in zip(by["handbook"][:12], by["rules4"][:12]) for x in pair]


def post(r):
    body = {"model": "sparkscore-d3", "state": r["state"], "questions": r["questions"], "knowledge": r["knowledge"]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())["answers"]


def flat(a):
    out = []
    for q in sorted(a):
        x = a[q]
        out += [x["noul"]] if "noul" in x else [x["probabilities"][k] for k in sorted(x["probabilities"])]
    return out


alone = [post(r) for r in mix]
with ThreadPoolExecutor(8) as pool:
    merged = list(pool.map(post, mix))
worst = max(abs(a - b) for x, y in zip(alone, merged) for a, b in zip(flat(x), flat(y)))
print(f"{len(mix)} requests on 2 packs, 8 in flight vs alone: max |dp| {worst:.6f}")
