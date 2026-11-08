"""G5: long handbooks (~4K and ~16K tokens) on Atlas: pack vs in-state, rule accuracy, agreement and latency."""
import json, os, random, statistics, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import handbook as H

URL = "http://127.0.0.1:8016/v1/systemone"
rng = random.Random(7)
COUNTRIES = ["Ireland", "Portugal", "Chile", "Kenya", "Japan", "Canada", "Poland", "Vietnam", "Morocco", "Norway", "Peru",
             "Ghana", "Austria", "Thailand", "Mexico", "Finland", "Egypt", "Colombia", "Greece", "Malaysia", "Uruguay", "Latvia"]
WORDS = ["sync", "ledger", "relay", "atlas", "harbor", "signal", "orbit", "prism", "anchor", "summit", "meadow", "falcon", "ember"]
def filler(n_sections):
    out = []
    for i in range(n_sections):
        kind = i % 4
        if kind == 0:
            c = COUNTRIES[i % len(COUNTRIES)]
            out.append(f"Regional note ({c}). Customers in {c} are invoiced in local currency where we hold a local entity; "
                       f"VAT or sales tax follows the customer's billing address. Public holidays in {c} do not change response "
                       f"targets. Data-residency questions from {c} go to the account owner, who checks the contract annex.")
        elif kind == 1:
            name = (rng.choice(WORDS) + rng.choice(WORDS)).title()
            out.append(f"Integration note ({name}). The {name} connector syncs every {rng.choice([5, 15, 30, 60])} minutes and "
                       f"retries failed batches three times. Ask for the connector log and the time of the last good sync before "
                       f"escalating. Feature requests for {name} go to the product board, not to engineering directly.")
        elif kind == 2:
            m = f"{rng.choice(WORDS)}-{rng.choice(WORDS)}"
            out.append(f"Macro '{m}'. Use it when a customer asks how to {rng.choice(['export reports', 'rotate API keys', 'add a seat', 'change the account owner', 'download invoices', 'enable single sign-on'])}. "
                       f"Link the help article, confirm the account name, and offer a short call if the steps fail twice.")
        else:
            k = rng.randint(100, 999)
            out.append(f"Known issue KI-{k}. Some dashboards load slowly after midnight UTC while nightly aggregation runs. "
                       f"There is no workaround beyond waiting about twenty minutes; do not open an incident for it unless "
                       f"the dashboard fails to load at all.")
    return out
base = H.POLICY.split("\n\n")
def long_policy(per_gap):
    parts = [base[0]]
    for sec in base[1:]:
        parts += filler(per_gap) + [sec]
    return "\n\n".join(parts)
ITEMS = H.build_test(0, 5)
def post(b):
    data = json.dumps(dict(b, model="sparkscore-d3")).encode()
    t = time.perf_counter()
    r = urllib.request.urlopen(urllib.request.Request(URL, data, {"content-type": "application/json"}), timeout=600)
    return json.loads(r.read())["answers"], (time.perf_counter() - t) * 1e3, r.headers.get("X-SparkScore-Pack-Tokens")
def vec(q, a):
    if q["type"] == "noul": return [1 - a["noul"], a["noul"]]
    if q["type"] == "choice": return [a["probabilities"][k] for k in q["criteria"]]
    return [a["probabilities"][str(i)] for i in range(len(q["criteria"]))]
arg = lambda p: max(range(len(p)), key=p.__getitem__)
for per_gap in (3, 16):
    pol = long_policy(per_gap)
    res, lat, ptoks = {}, {}, None
    for cond in ("in-state", "pack"):
        res[cond], lat[cond] = [], []
        for rnd in range(2):
            for it in ITEMS:
                b = {"state": it["state"], "questions": H.QS}
                if cond == "in-state": b["state"] = pol + "\n\n" + it["state"]
                else: b["knowledge"] = pol
                a, ms, pt = post(b)
                ptoks = pt or ptoks
                lat[cond].append(ms)
                if rnd == 1: res[cond].append({q: vec(H.QS[q], a[q]) for q in H.QS})
    ok = {c: sum(arg(r[H.RULE_Q[t]]) == (list(H.QS["team"]["criteria"]).index(it["gold"]["team"]) if H.RULE_Q[t] == "team" else it["gold"][H.RULE_Q[t]])
                 for it, r in zip(ITEMS, res[c]) for t in it["triggers"]) for c in res}
    n_rules = sum(len(it["triggers"]) for it in ITEMS)
    agree = sum(arg(a[q]) == arg(b[q]) for a, b in zip(res["pack"], res["in-state"]) for q in H.QS)
    print(f"handbook ~{ptoks} pack tokens: rules in-state {ok['in-state']}/{n_rules}, pack {ok['pack']}/{n_rules}; "
          f"pack vs in-state argmax {agree}/{3 * len(ITEMS)}; p50 in-state {statistics.median(lat['in-state']):.0f} ms, "
          f"pack {statistics.median(lat['pack'][1:]):.0f} ms", flush=True)
