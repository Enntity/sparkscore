"""G3/G4 on Atlas: knowledge packs against the d3 BF16 isolated-prefix reference, and their speed.

    python3 g3g4.py REFERENCE.jsonl http://127.0.0.1:8016 [rounds]

Conditions per item: none (state only), in-state (knowledge + state), pack (knowledge as a pack).
"""
import json, math, statistics, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ROWS = [json.loads(l) for l in open(sys.argv[1])]
URL = sys.argv[2] + "/v1/systemone"
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 1


def post(body):
    data = json.dumps(dict(body, model="sparkscore-d3")).encode()
    t = time.perf_counter()
    r = urllib.request.urlopen(urllib.request.Request(URL, data, {"content-type": "application/json"}), timeout=300)
    out = json.loads(r.read())
    return out["answers"], (time.perf_counter() - t) * 1e3, r.headers.get("Server-Timing", "")


def vec(q, a):
    if q["type"] == "noul":
        return [1 - a["noul"], a["noul"]]
    if q["type"] == "choice":
        return [a["probabilities"][k] for k in q["criteria"]]
    return [a["probabilities"][str(i)] for i in range(len(q["criteria"]))]


def body(row, cond):
    if cond == "none":
        return {"state": row["state"], "questions": row["questions"]}
    if cond == "in-state":
        return {"state": row["knowledge"] + "\n\n" + row["state"], "questions": row["questions"]}
    return {"state": row["state"], "questions": row["questions"], "knowledge": row["knowledge"]}


def kl(p, q):
    return sum(a * math.log(max(a, 1e-12) / max(b, 1e-12)) for a, b in zip(p, q))


def arg(p):
    return max(range(len(p)), key=p.__getitem__)


RULE_Q = {"helios": "team", "orchard": "team", "lattice": "team", "kestrel": "urgency", "osprey": "urgency",
          "tamarind": "legal", "wisteria": "legal"}


def gold_index(row, q):
    g = row["gold"][q]
    return list(row["questions"]["team"]["criteria"]).index(g) if q == "team" else g


results = {}
for cond in ("none", "in-state", "pack"):
    got, lat, packs = [], [], {}
    for rnd in range(ROUNDS):
        for row in ROWS:
            a, ms, st = post(body(row, cond))
            if rnd == ROUNDS - 1:
                got.append({q: vec(row["questions"][q], a[q]) for q in row["questions"]})
            lat.append(ms)
            for part in st.split(","):
                if part.strip().startswith("pack;desc="):
                    packs[part.split("=")[1]] = packs.get(part.split("=")[1], 0) + 1
    results[cond] = got
    lat.sort()
    print(f"{cond:<9} p50 {statistics.median(lat):6.1f} ms  p90 {lat[int(0.9 * len(lat))]:6.1f} ms  n={len(lat)}  packs={packs}", flush=True)

for name in sorted({r["set"] for r in ROWS}):
    idx = [i for i, r in enumerate(ROWS) if r["set"] == name]
    print(f"\n== {name} ({len(idx)} items)")
    for cond, ref_key in (("none", "probs_none"), ("in-state", "probs_instate"), ("pack", "probs_prefix")):
        rule_ok = rule_n = agree = n = 0
        kls = []
        for i in idx:
            row, out = ROWS[i], results[cond][i]
            ref = row[ref_key]
            for q, p in out.items():
                n += 1
                agree += arg(p) == arg(ref[q])
                kls.append(kl(ref[q], p))
            for t in row["triggers"]:
                q = RULE_Q[t]
                rule_n += 1
                rule_ok += arg(out[q]) == gold_index(row, q)
        print(f"{cond:<9} rules {rule_ok}/{rule_n}  vs BF16 {ref_key}: argmax {agree}/{n}, mean KL {sum(kls)/n:.4f}, max KL {max(kls):.3f}")
    agree = sum(arg(results["pack"][i][q]) == arg(results["in-state"][i][q]) for i in idx for q in results["pack"][i])
    total = sum(len(results["pack"][i]) for i in idx)
    print(f"pack vs in-state on Atlas: argmax {agree}/{total}")

# Merged passes score exactly as alone (pack requests, 8 in flight).
alone = results["pack"]
with ThreadPoolExecutor(8) as pool:
    merged = list(pool.map(lambda r: post(body(r, "pack"))[0], ROWS))
worst = max(abs(x - y) for row, a, m in zip(ROWS, alone, merged) for q in a
            for x, y in zip(a[q], vec(row["questions"][q], m[q])))
print(f"\nmerged (8 in flight) vs alone, pack requests: max |dp| {worst:.6f}")
