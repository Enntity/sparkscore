"""Latency of ONE image question per reference image (first question only), median of N after one warm-up."""
import base64, json, statistics, sys
from urllib.request import Request, urlopen
ref, imgdir, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
for line in open(ref):
    r = json.loads(line); q = r["questions"]["oval"]; s0 = q["state"]
    ext = r["image"].rsplit(".", 1)[1].lower()
    body = {"prefix": q["ids"][:s0], "branches": [q["ids"][s0:]], "options": [q["code_ids"]], "attention": "prefix_bidir",
            "images": [{"data_base64": base64.b64encode(open(f"{imgdir}/{r['image']}", "rb").read()).decode(), "format": {"jpg": "jpeg"}.get(ext, ext)}]}
    us = []
    for i in range(n + 1):
        out = json.loads(urlopen(Request("http://127.0.0.1:8081/v1/score", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600).read())
        if i: us.append(out["engine_us"] / 1000)
    print(f"{r['id']:28s} rows={len(q['ids']):5d}  one question: {statistics.median(us):6.1f} ms", flush=True)
