"""d3's own runtime on the image gate cases: per question the expanded prompt ids, state block size, code ids,
probabilities (request temperature) and the request's wall time.   python img_ref.py MODEL IMGDIR OUT.jsonl"""
import json, sys, time
model, imgdir, out = sys.argv[1:4]
sys.path.insert(0, model); sys.path.insert(0, "/w/d3work/img")
import torch
from d3_runtime import D3
import importlib, os; cases = importlib.import_module(os.environ.get("CASES", "img_cases")).cases
m = D3.from_pretrained(model, device="cuda:0", verify="fast")
print("loaded; state_once", m.state_once is not None, "blocked", m.blocked, flush=True)
with open(out, "w") as f:
    for rep in range(2):  # rep 0 warms up; rep 1 is timed and recorded
        for c in cases():
            path = f"{imgdir}/{c['image']}"
            torch.cuda.synchronize(); t0 = time.perf_counter()
            prep = m.prepare(c["state"], c["questions"], images=[path])
            probs, ntok = m.run(prep)
            torch.cuda.synchronize(); wall = time.perf_counter() - t0
            if rep == 0:
                continue
            T = m.temperature_for(len(prep.questions))
            rows = {}
            for k in prep.questions:
                enc = m.processor(text=[prep.texts[k]], images=prep.images, return_tensors="pt")
                n = len(prep.questions[k].keys)
                rows[k] = {"ids": enc["input_ids"][0].tolist(), "state": prep.states.get(k),
                           "code_ids": list(m.token_ids[:n]), "probs": probs[k]}
            f.write(json.dumps({"id": c["id"], "image": c["image"], "T": T, "wall_ms": wall * 1000,
                                "tokens": ntok, "errors": prep.errors, "questions": rows}) + "\n")
            print(c["id"], f"{wall*1000:.0f} ms", {k: [round(p, 3) for p in v["probs"]] for k, v in rows.items()}, flush=True)
