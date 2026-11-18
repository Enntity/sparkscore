"""Timing probe: image request vs the same token count as text, on /v1/score. Prints engine_us per variant."""
import importlib.util, json, sys, copy, statistics
from urllib.request import Request, urlopen
spec = importlib.util.spec_from_file_location("s", sys.argv[1]); s = importlib.util.module_from_spec(spec)
src = open(sys.argv[1]).read().split("landscape_runs =")[0]  # definitions + tokenizer + prepared requests
exec(compile(src, "smoke", "exec"), s.__dict__)
URL = "http://127.0.0.1:8081/v1/score"
def post(body):
    r = urlopen(Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=300)
    return json.loads(r.read())
for name in ("landscape", "wide"):
    body = json.loads(s.__dict__[name]["request"].data)
    pad = s.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vs, ve = (s.tokenizer.convert_tokens_to_ids(t) for t in ("<|vision_start|>", "<|vision_end|>"))
    text = dict(body); text.pop("images")
    text["prefix"] = [64 if t in (pad, vs, ve) else t for t in body["prefix"]]
    for label, b in (("image", body), ("text", text), ("image+bidir", dict(body, attention="prefix_bidir")), ("text+bidir", dict(text, attention="prefix_bidir"))):
        us = [post(b)["engine_us"] for _ in range(4)][1:]
        print(f"{name:9s} {label:12s} rows={len(b['prefix'])}+br  engine_ms={statistics.median(us)/1000:.1f}", flush=True)
