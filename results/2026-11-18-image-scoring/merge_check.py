"""Throughput + exactness of merged image passes: every reference request (all its questions) sent once
sequentially, then all at once from N threads. Merged answers must equal the sequential ones exactly.
    python concurrent.py REF.jsonl IMGDIR N"""
import base64, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen
ref, imgdir, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
bodies = []
for line in open(ref):
    r = json.loads(line); qs = r["questions"]; keys = list(qs); s0 = qs[keys[0]]["state"]
    ext = r["image"].rsplit(".", 1)[1].lower()
    bodies.append({"prefix": qs[keys[0]]["ids"][:s0], "branches": [qs[k]["ids"][s0:] for k in keys],
                   "options": [qs[k]["code_ids"] for k in keys], "attention": "prefix_bidir",
                   "images": [{"data_base64": base64.b64encode(open(f"{imgdir}/{r['image']}", "rb").read()).decode(),
                               "format": {"jpg": "jpeg"}.get(ext, ext)}]})
def post(b):
    return json.loads(urlopen(Request("http://127.0.0.1:8081/v1/score", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"}), timeout=600).read())
post(bodies[0])
t = time.perf_counter(); seq = [post(b) for b in bodies]; t_seq = time.perf_counter() - t
t = time.perf_counter()
with ThreadPoolExecutor(n) as ex: par = list(ex.map(post, bodies))
t_par = time.perf_counter() - t
same = sum(a["probs"] == b["probs"] for a, b in zip(seq, par)); dmax = max(abs(x - y) for a, b in zip(seq, par) for ra, rb in zip(a["probs"], b["probs"]) for x, y in zip(ra, rb)); print("max |dp| merged vs alone:", dmax)
batched = [p.get("batched") for p in par]
print(f"{len(bodies)} image requests: sequential {t_seq:.2f} s ({len(bodies)/t_seq:.2f} img/s), "
      f"{n} concurrent {t_par:.2f} s ({len(bodies)/t_par:.2f} img/s); identical {same}/{len(bodies)}; batched {batched}")
