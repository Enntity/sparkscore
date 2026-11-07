#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Compare Atlas's /v1/score with d3-ultra's own BF16 runtime, question by question.

    oracle_check.py [ATLAS_URL] [REFERENCE.jsonl] [decision_config.json]

Defaults: http://127.0.0.1:8081 (Atlas inside the SparkScore container),
data/d3-bf16-reference.jsonl next to this file (d3's own token ids and BF16
logits, recorded with d3's runtime on one DGX Spark) and
/model/decision_config.json. Feeds d3's exact prompt tokens with the
prefix_bidir mask and with plain causal attention, and prints agreement, mean
and max |dp| and mean KL against the reference. Run it inside the container:

    docker exec sparkscore /opt/venv/bin/python tools/sparkscore/bench/oracle_check.py
"""
import json, math, os, statistics, sys, urllib.error, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8081"
REF = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "data", "d3-bf16-reference.jsonl")
CFG = sys.argv[3] if len(sys.argv) > 3 else "/model/decision_config.json"
ids = json.load(open(CFG))["token_ids"]
rows = [json.loads(l) for l in open(REF)]
rows = [r for r in rows if not r.get("_meta")]
def score(prefix, branch, opts, mode):
    b = {"prefix": prefix, "branches": [branch], "options": [opts], "attention": mode}
    q = urllib.request.Request(URL + "/v1/score", json.dumps(b).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(q, timeout=600).read())["probs"][0]
def soft(logits, t):
    m = max(logits); e = [math.exp((x - m) / t) for x in logits]; s = sum(e); return [x / s for x in e]
res = {"prefix_bidir": [], "causal": []}
for r in rows:
    if len(r["answers"]) > 4 or sum(len(a.get("prompt_token_ids", [])) for a in r["answers"].values()) > 12000:
        continue
    t = r.get("temperature", 1.0)
    for name, a in r["answers"].items():
        if "logits" not in a or "prompt_token_ids" not in a:
            continue
        p, s = a["prompt_token_ids"], a["state_block_tokens"]
        ref = soft([float(x) for x in a["logits"]], 1.0)  # untempered reference
        k = len(ref)
        for mode in res:
            try:
                got = score(p[:s], p[s:], ids[:k], mode)
            except urllib.error.HTTPError as e:
                print("ERR", mode, r["id"], e.read()[:100]); continue
            res[mode].append((max(abs(x - y) for x, y in zip(got, ref)),
                              sum(y * math.log(max(y, 1e-12) / max(x, 1e-12)) for x, y in zip(got, ref)),
                              max(range(k), key=ref.__getitem__) == max(range(k), key=got.__getitem__), r["id"] + "/" + name))
for mode, v in res.items():
    d = [x[0] for x in v]; kl = [x[1] for x in v]
    print(f"{mode:>12}: n={len(v)} agree {sum(x[2] for x in v)}/{len(v)} mean|dp| {statistics.mean(d):.4f} max|dp| {max(d):.4f} ({max(v)[3]}) meanKL {statistics.mean(kl):.5f}")
