"""SparkScore's d3-ultra image rendering against d3's own runtime, token for token: every reference question's
expanded prompt ids and state block size (ref*.jsonl from img_ref.py).   python -m ... TOKENIZER IMGDIR REF..."""
import base64, json, sys
from tools.sparkscore import images
from tools.sparkscore.templates.d3_ultra import Renderer
tok, imgdir, refs = sys.argv[1], sys.argv[2], sys.argv[3:]
r = Renderer(tokenizer=tok, revision=None)
cases = __import__("importlib").import_module
ok = bad = 0
for ref in refs:
    for line in open(ref):
        row = json.loads(line)
        raw = open(f"{imgdir}/{row['image']}", "rb").read()
        kind = {"jpg": "jpeg"}.get(row["image"].rsplit(".", 1)[1], row["image"].rsplit(".", 1)[1])
        imgs = images.parse([f"data:image/{kind};base64," + base64.b64encode(raw).decode()])
        mod = cases("coco_cases" if "coco" in row["id"] else "img_cases")
        case = next(c for c in mod.cases() if c["id"] == row["id"])
        out = r.render(case["state"], case["questions"], images=imgs)
        for b in out.branches:
            q = row["questions"][b.name]
            whole, s = out.prefix + b.ids, len(out.prefix)
            good = whole == q["ids"] and s == q["state"] and b.option_ids == q["code_ids"]
            ok += good; bad += not good
            if not good:
                print("MISMATCH", row["id"], b.name, len(whole), len(q["ids"]), s, q["state"],
                      next((i for i, (x, y) in enumerate(zip(whole, q["ids"])) if x != y), None))
        if abs(out.temperature - row["T"]) > 1e-9:
            print("TEMPERATURE", row["id"], out.temperature, row["T"])
print(f"{ok} identical, {bad} different")
