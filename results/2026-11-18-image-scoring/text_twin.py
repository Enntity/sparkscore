"""Send the text twin of a reference image request (vision tokens -> token 64, no images) N times."""
import json, sys
from urllib.request import Request, urlopen
r = json.loads(open(sys.argv[1]).readline()); n = int(sys.argv[2])
qs = r["questions"]; keys = list(qs); s0 = qs[keys[0]]["state"]
VISION = {248053, 248056, 248054}
prefix = [64 if t in VISION else t for t in qs[keys[0]]["ids"][:s0]]
body = {"prefix": prefix, "branches": [qs[k]["ids"][s0:] for k in keys], "options": [qs[k]["code_ids"] for k in keys], "attention": "prefix_bidir"}
for _ in range(n):
    out = json.loads(urlopen(Request("http://127.0.0.1:8081/v1/score", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600).read())
    print("text twin engine_ms", out["engine_us"] / 1000)
