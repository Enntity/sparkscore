"""G4: latency per knowledge set: none / in-state / pack (pack already built), 3 rounds, wall at the client."""
import json, statistics, sys, time, urllib.request
ROWS = [json.loads(l) for l in open(sys.argv[1])]
URL = sys.argv[2] + "/v1/systemone"
def post(body):
    data = json.dumps(dict(body, model="sparkscore-d3")).encode()
    t = time.perf_counter()
    urllib.request.urlopen(urllib.request.Request(URL, data, {"content-type": "application/json"}), timeout=300).read()
    return (time.perf_counter() - t) * 1e3
for name in sorted({r["set"] for r in ROWS}):
    rows = [r for r in ROWS if r["set"] == name]
    for cond in ("none", "in-state", "pack"):
        lat = []
        for _ in range(3):
            for r in rows:
                b = {"state": r["state"], "questions": r["questions"]}
                if cond == "in-state":
                    b["state"] = r["knowledge"] + "\n\n" + r["state"]
                if cond == "pack":
                    b["knowledge"] = r["knowledge"]
                lat.append(post(b))
        lat.sort()
        print(f"{name:<9} {cond:<9} p50 {statistics.median(lat):6.1f} ms  p90 {lat[int(.9*len(lat))]:6.1f} ms  mean {statistics.fmean(lat):6.1f}", flush=True)
