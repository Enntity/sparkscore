"""Atlas /v1/score on the d3 reference's own token ids: one request per image (shared state block, one branch per
question, prefix_bidir), probabilities at the reference's request temperature, compared with d3's.
    python img_atlas.py REF.jsonl IMGDIR OUT.json [--reps N]"""
import base64, json, math, sys, time
from urllib.request import Request, urlopen
ref_path, imgdir, out_path = sys.argv[1:4]
reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 3
URL = "http://127.0.0.1:8081/v1/score"
def post(body):
    t = time.perf_counter()
    with urlopen(Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600) as r:
        out = json.loads(r.read())
    out["wall_ms"] = (time.perf_counter() - t) * 1000
    return out
def tempered(p, T):
    w = [x ** (1 / T) if x > 0 else 0.0 for x in p]; z = sum(w); return [x / z for x in w]
rows, worst, flips, kls = [], 0.0, 0, []
for line in open(ref_path):
    r = json.loads(line)
    qs = r["questions"]; keys = list(qs)
    s0 = qs[keys[0]]["state"]; prefix = qs[keys[0]]["ids"][:s0]
    assert all(q["ids"][:q["state"]] == prefix for q in qs.values()), "questions have different state blocks"
    ext = r["image"].rsplit(".", 1)[1].lower()
    body = {"prefix": prefix, "branches": [qs[k]["ids"][s0:] for k in keys], "options": [qs[k]["code_ids"] for k in keys],
            "attention": "prefix_bidir",
            "images": [{"data_base64": base64.b64encode(open(f"{imgdir}/{r['image']}", "rb").read()).decode(),
                        "format": {"jpg": "jpeg"}.get(ext, ext)}]}
    runs = [post(body) for _ in range(reps)]
    probs = runs[0]["probs"]
    assert all(x["probs"] == probs for x in runs), "repeat runs differ"
    detail = {}
    for k, p in zip(keys, probs):
        a, d = tempered(p, r["T"]), qs[k]["probs"]
        dp = max(abs(x - y) for x, y in zip(a, d)); worst = max(worst, dp)
        kl = sum(y * math.log(y / max(x, 1e-12)) for x, y in zip(a, d) if y > 0); kls.append(kl)
        flip = max(range(len(a)), key=a.__getitem__) != max(range(len(d)), key=d.__getitem__); flips += flip
        detail[k] = {"atlas": [round(x, 4) for x in a], "d3": [round(x, 4) for x in d], "max_dp": round(dp, 4), "kl": round(kl, 5), "flip": flip}
    eng = sorted(x["engine_us"] / 1000 for x in runs)
    rows.append({"id": r["id"], "rows": len(qs[keys[0]]["ids"]), "engine_ms": eng[len(eng) // 2], "d3_wall_ms": round(r["wall_ms"], 1), "questions": detail})
    print(f"{r['id']:28s} rows={rows[-1]['rows']:5d} atlas={rows[-1]['engine_ms']:7.1f} ms  d3={r['wall_ms']:7.1f} ms  "
          f"max_dp={max(v['max_dp'] for v in detail.values()):.4f} flips={sum(v['flip'] for v in detail.values())}", flush=True)
summary = {"requests": len(rows), "questions": len(kls), "max_dp": round(worst, 4), "mean_kl": round(sum(kls) / len(kls), 5), "argmax_flips": flips}
print(json.dumps(summary))
json.dump({"summary": summary, "rows": rows}, open(out_path, "w"), indent=1)
