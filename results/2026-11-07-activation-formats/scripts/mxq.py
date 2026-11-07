# SPDX-License-Identifier: AGPL-3.0-only
"""Fake-quant sanity check on d3's own BF16 runtime: quantize the weights (and optionally the inputs) of the
matrices the engine runs in 4-bit (MLP gate/up/down, attention q/k/v/o), keep everything else BF16, and
compare each question's probabilities with d3's BF16 reference (same requests).

    python mxq.py MODEL_DIR CONVERTER_DIR REQUESTS.jsonl REFERENCE.jsonl OUT_PREFIX WEIGHTS ACTS

WEIGHTS: bf16 | nvfp4 | mx_ocp | mx_mse | mx_row   (see mxquant.py)
ACTS:    comma list of bf16 | e4m3tok | mx8 | nvfp4 | nvfp4x2; one pass over the requests each
"""
import json, math, os, sys, time
model_dir, conv_dir, req_path, ref_path, prefix, wvar, acts = sys.argv[1:8]
sys.path[:0] = [model_dir, conv_dir, os.path.dirname(os.path.abspath(__file__))]
import torch
from d3_runtime import D3
import mxquant

t0 = time.time()
m = D3.from_pretrained(model_dir, device="cuda:0", verify="fast")
rel = mxquant.apply(m, wvar)
mxquant.install_act_hooks(m)
print(wvar, "weights ready in", round(time.time() - t0), "s; relative weight RMS error:", {k: round(v, 4) for k, v in rel.items()}, flush=True)

ref = {}
for line in open(ref_path):
    r = json.loads(line)
    if not r.get("_meta"):
        ref[r["id"]] = r["answers"]
reqs = [json.loads(l) for l in open(req_path) if l.strip()]
for act in acts.split(","):
    mxquant.ACT["mode"] = act
    rows = []
    with torch.no_grad():
        for r in reqs:
            probs, _ = m.run(m.prepare(r["state"], r["questions"]))
            for k, p in probs.items():
                q = ref[r["id"]][k]["probabilities"]
                kl = sum(a * math.log(max(a, 1e-12) / max(b, 1e-12)) for a, b in zip(q, p) if a > 0)
                rows.append({"id": f"{r['id']}/{k}", "kl": kl, "max_dp": max(abs(a - b) for a, b in zip(q, p)),
                             "agree": max(range(len(q)), key=q.__getitem__) == max(range(len(p)), key=p.__getitem__)})
    kls = sorted(x["kl"] for x in rows)
    s = {"weights": wvar, "acts": act, "n": len(rows), "mean_kl": sum(kls) / len(kls),
         "trim3_kl": sum(kls[:-3]) / (len(kls) - 3), "median_kl": kls[len(kls) // 2], "max_kl": kls[-1],
         "agree": sum(x["agree"] for x in rows), "over_0.01": sum(k > 0.01 for k in kls), "weight_rel_rms": rel}
    json.dump({"summary": s, "rows": rows}, open(f"{prefix}-{wvar}-{act}.json", "w"), indent=1)
    print("MXQ_DONE", json.dumps({k: v for k, v in s.items() if k != "weight_rel_rms"}), flush=True)
