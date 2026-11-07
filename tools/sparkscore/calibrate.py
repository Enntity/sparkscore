# SPDX-License-Identifier: AGPL-3.0-only
"""Fit a readout temperature per question type on held-out kev rows.

    python3 -m tools.sparkscore.calibrate --fit RUN_DIR [RUN_DIR ...] --eval RUN_DIR [RUN_DIR ...]

RUN_DIR is a kev result directory (one holding `rows.json` from a run of
`tools.sparkscore.bench.run kev`). Only `clean` rows are used. A temperature
T rescales each answer distribution p to p**(1/T), renormalised: the same as
dividing the option logits by T. T is fitted per type by minimising NLL on the
--fit rows (golden-section search on log T), then reported on the --eval rows
together with accuracy (unchanged by T), NLL, Brier and top-label ECE (10 bins).
"""
import argparse
import json
import math
from pathlib import Path


def load_rows(run_dirs):
    rows = []
    for d in run_dirs:
        for f in Path(d).rglob("rows.json"):
            for r in json.loads(f.read_text()):
                if r.get("variant") == "clean" and isinstance(r.get("label"), int) and r.get("p"):
                    rows.append((r["type"], [float(x) for x in r["p"]], r["label"]))
    return rows


def temper(p, t):
    logs = [math.log(max(x, 1e-12)) / t for x in p]
    m = max(logs)
    e = [math.exp(x - m) for x in logs]
    s = math.fsum(e)
    return [x / s for x in e]


def metrics(rows, temps):
    n = len(rows)
    if not n:
        return {}
    nll = brier = correct = 0.0
    bins = [[0, 0.0, 0.0] for _ in range(10)]  # count, confidence sum, correct sum
    for typ, p, y in rows:
        q = temper(p, temps.get(typ, 1.0))
        top = max(range(len(q)), key=q.__getitem__)
        ok = float(top == y)
        correct += ok
        nll -= math.log(max(q[y], 1e-12))
        brier += sum((qi - (i == y)) ** 2 for i, qi in enumerate(q))
        b = bins[min(int(q[top] * 10), 9)]
        b[0] += 1
        b[1] += q[top]
        b[2] += ok
    ece = sum(abs(c - k) for _, c, k in bins) / n
    return {"n": n, "acc": correct / n, "nll": nll / n, "brier": brier / n, "ece": ece}


def fit(rows):
    lo, hi = math.log(0.25), math.log(8.0)
    f = lambda lt: metrics(rows, {rows[0][0]: math.exp(lt)})["nll"]
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(60):
        if f(c) < f(d):
            b, d = d, c
            c = b - g * (b - a)
        else:
            a, c = c, d
            d = a + g * (b - a)
    return math.exp((a + b) / 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", nargs="+", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    a = ap.parse_args()
    fit_rows, eval_rows = load_rows(a.fit), load_rows(a.eval)
    temps = {}
    for typ in sorted({t for t, _, _ in fit_rows}):
        sub = [r for r in fit_rows if r[0] == typ]
        if len(sub) >= 30:
            temps[typ] = fit(sub)
    print("temperatures:", {k: round(v, 3) for k, v in temps.items()})
    for name, t in (("before", {}), ("after", temps)):
        m = metrics(eval_rows, t)
        print(f"{name:>6}: " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))
        for typ in sorted({t for t, _, _ in eval_rows}):
            mt = metrics([r for r in eval_rows if r[0] == typ], t)
            print(f"        {typ:>6}: n={mt['n']} nll={mt['nll']:.4f} ece={mt['ece']:.4f}")


if __name__ == "__main__":
    main()
