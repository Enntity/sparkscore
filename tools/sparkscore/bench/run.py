# SPDX-License-Identifier: AGPL-3.0-only
"""Run a public harness against a /v1/systemone endpoint and keep receipts.

    python3 -m tools.sparkscore.bench.run smoke          --endpoint URL --model M [--out DIR]
    python3 -m tools.sparkscore.bench.run decision-index --endpoint URL --model M --harness DIR --out DIR [--rows F] [--limit N]
    python3 -m tools.sparkscore.bench.run jevbench       --endpoint URL --model M --harness DIR --out DIR [--limit N]
    python3 -m tools.sparkscore.bench.run kev            --endpoint URL --model M --harness DIR --out DIR [--suite S]

The three harnesses run their own code, unmodified, from a checkout given by
--harness (with --python naming that checkout's interpreter). Everything they
write goes to DIR/raw/, with DIR/raw/manifest.json: the harness commit, the
exact command, the endpoint's /v1/models and /v1/limits, and the wall time.
These are unedited receipts and may name local paths. DIR/SHA256SUMS covers
every file. See README.md next to this file.

`smoke` needs no harness: it sends tests/fixtures/smoke.jsonl, checks every
answer the way the harnesses do, and prints accuracy, Brier, ECE and latency.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SMOKE = ROOT / "tests" / "fixtures" / "smoke.jsonl"
JEVBENCH_TASKS = ["datasets/public/easy.jsonl", "datasets/public/hard.jsonl", "datasets/public/original.jsonl"]
# The harness revisions this runner was written against.
PINS = {
    "decision-index": "25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d",
    "jevbench": "609a664e81f46e03731600bd7bd9f1b7a61257c5",
    "kev": "2074cc44256200bb37540103fa4572901faa10f3",
}


# ------------------------------------------------------------------ HTTP
def _request(url, body=None, timeout=600):
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("SPARKSCORE_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="GET" if body is None else "POST")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, payload, server_timing = resp.status, resp.read(), resp.headers.get("Server-Timing")
    except urllib.error.HTTPError as error:
        status, payload, server_timing = error.code, error.read(), None
    wall = (time.perf_counter() - start) * 1000
    try:
        parsed = json.loads(payload)
    except ValueError:
        parsed = payload.decode(errors="replace")
    return status, parsed, wall, server_timing


def _endpoint_info(endpoint):
    info = {}
    for path in ("/v1/models", "/v1/limits"):
        status, body, _, _ = _request(endpoint + path, timeout=30)
        info[path] = body if status == 200 else {"http_status": status, "body": body}
    return info


# ------------------------------------------------------------------ smoke
def check_answer(question, answer):
    """Raise ValueError unless `answer` would pass Decision Index, JevBench
    (sum within 1e-3) and kev for this question; return (keys, probs)."""
    kind = question["type"]
    if answer.get("type") != kind:
        raise ValueError("answer type does not match the question")
    if kind == "noul":
        p = answer.get("noul")
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1:
            raise ValueError("noul must be a number in [0, 1]")
        return ["true", "false"], [float(p), 1 - float(p)]
    keys = list(question["criteria"]) if kind == "choice" else [str(i) for i in range(len(question["criteria"]))]
    probs = answer.get("probabilities")
    if not isinstance(probs, dict) or set(probs) != set(keys):
        raise ValueError("probability keys do not match the options")
    values = [probs[k] for k in keys]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in values):
        raise ValueError("probabilities must be numbers in [0, 1]")
    if abs(math.fsum(values) - 1) > 1e-3:
        raise ValueError("probabilities do not sum to 1")
    conf = answer.get("confidence")
    if not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
        raise ValueError("confidence must be in [0, 1]")
    if kind == "choice" and answer.get("choice") not in keys:
        raise ValueError("choice is not one of the options")
    if kind == "score":
        if set(answer.get("legend") or {}) != set(keys):
            raise ValueError("legend keys do not match the levels")
        if not 0 <= answer.get("score", -1) <= len(keys) - 1:
            raise ValueError("score is outside the levels")
    return keys, values


def ece(pairs, bins=10):
    """Top-label expected calibration error, 10 equal-width bins (as JevBench and kev)."""
    total, n = 0.0, len(pairs)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        group = [(c, ok) for c, ok in pairs if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if group:
            conf = sum(c for c, _ in group) / len(group)
            acc = sum(ok for _, ok in group) / len(group)
            total += len(group) / n * abs(acc - conf)
    return total


def _percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))] if values else None


def smoke(endpoint, model, out=None, path=SMOKE):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    decisions, failures, latency, server = [], [], [], []
    raw = []
    for row in rows:
        status, body, wall, timing = _request(endpoint + "/v1/systemone", {"model": model, "state": row["state"], "questions": row["questions"]})
        raw.append({"id": row["id"], "http_status": status, "response": body, "wall_ms": wall, "server_timing": timing})
        if status != 200:
            failures.append({"id": row["id"], "error": f"HTTP {status}: {str(body)[:200]}"})
            continue
        latency.append(wall)
        if timing:
            server.append({k: float(v.split("dur=")[1]) for k, v in (p.strip().split(";", 1) for p in timing.split(","))})
        for name, question in row["questions"].items():
            try:
                keys, probs = check_answer(question, (body.get("answers") or {})[name])
            except (KeyError, ValueError) as error:
                failures.append({"id": row["id"], "question": name, "error": str(error)})
                continue
            gold = row["expected"][name]
            gold = "true" if gold is True else "false" if gold is False else str(gold)
            top = max(range(len(probs)), key=probs.__getitem__)
            decisions.append({
                "type": question["type"],
                "correct": keys[top] == gold,
                "confidence": probs[top],
                "brier": math.fsum((p - (k == gold)) ** 2 for k, p in zip(keys, probs)),
            })
    n = len(decisions)
    summary = {
        "endpoint": endpoint,
        "model": model,
        "requests": len(rows),
        "decisions": sum(len(r["questions"]) for r in rows),
        "answered": n,
        "failures": failures,
        "accuracy": sum(d["correct"] for d in decisions) / n if n else None,
        "brier": math.fsum(d["brier"] for d in decisions) / n if n else None,
        "ece": ece([(d["confidence"], d["correct"]) for d in decisions]) if n else None,
        "by_type": {
            kind: sum(d["correct"] for d in decisions if d["type"] == kind) / max(1, sum(d["type"] == kind for d in decisions))
            for kind in ("noul", "choice", "score")
        },
        "latency_ms": {"p50": _percentile(latency, 0.5), "p95": _percentile(latency, 0.95)},
        "server_ms_p50": {k: _percentile([s[k] for s in server if k in s], 0.5) for k in ("render", "backend", "answer", "total")} if server else None,
    }
    if out:
        (Path(out) / "raw").mkdir(parents=True, exist_ok=True)
        (Path(out) / "raw" / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw))
        (Path(out) / "raw" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# ------------------------------------------------------------------ harnesses
def _harness_commands(a, raw):
    url, model, h = a.endpoint, a.model, Path(a.harness).resolve()
    key = os.environ.get("SPARKSCORE_API_KEY")
    if a.harness_name == "decision-index":
        opts = ["--engine", "http", "--option", f"base_url={url}", "--option", f"model={model}"]
        extra = (["--rows", str(Path(a.rows).resolve())] if a.rows else []) + (["--limit", str(a.limit)] if a.limit else [])
        env = {"DECISION_INDEX_API_KEY": key} if key else {}
        return h, env, [[a.python, "-m", "decision_index", "pipeline", *opts, *extra, "--out", str(raw / "run")]]
    if a.harness_name == "jevbench":
        tasks = ",".join(JEVBENCH_TASKS)
        limit = ["--limit", str(a.limit)] if a.limit else []
        run = [a.python, "-m", "jevbench.cli", "run", "--tasks", tasks, "--adapter", "typesafe",
               "--endpoint", url, "--model", model, "--key-env", "SPARKSCORE_API_KEY" if key else "",
               "--results", str(raw / "results.jsonl"), "--ledger", str(raw / "ledger.jsonl"),
               "--raw-dir", str(raw / "responses"), "--price-in-per-m", "0", "--price-out-per-m", "0",
               "--cost-basis", "local_spark_compute_excluded", "--run-label", model,
               "--manifest", str(raw / "run-manifest.json"), *limit]
        summarize = [a.python, "-m", "jevbench.cli", "summarize", "--tasks", tasks, "--results", str(raw / "results.jsonl"),
                     "--public-export", str(raw / "public-export.json")]
        return h, {}, [run, (summarize, raw / "summary.json")]
    suite = a.suite or "evals/v4/transfer-v4"
    cmd = [a.python, "-m", "kev.benchmark", "--remote", url, "--remote-model", model, "--suite", suite,
           "--out", str(raw / Path(suite).name), "--remote-concurrency", str(a.concurrency)]
    return h, {"KEV_REMOTE_API_KEY": key or "local"}, [cmd]


def _git(path, *args):
    try:
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def harness(a):
    out = Path(a.out).resolve()
    raw = out / "raw"
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty; every run gets a fresh directory")
    raw.mkdir(parents=True)
    cwd, env, commands = _harness_commands(a, raw)
    commit = _git(cwd, "rev-parse", "HEAD")
    if commit != PINS[a.harness_name]:
        print(f"note: {a.harness_name} is at {commit}, this runner was written against {PINS[a.harness_name]}", file=sys.stderr)
    manifest = {
        "harness": a.harness_name,
        "harness_commit": commit,
        "harness_dirty": bool(_git(cwd, "status", "--porcelain")),
        "endpoint": a.endpoint,
        "model": a.model,
        "endpoint_info": _endpoint_info(a.endpoint),
        "sparkscore_commit": _git(ROOT, "rev-parse", "HEAD"),
        "host_python": platform.python_version(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "commands": [],
    }
    status = 0
    for command in commands:
        command, stdout = command if isinstance(command, tuple) else (command, None)
        start = time.time()
        with open(stdout, "w") if stdout else open(raw / "harness.log", "a") as sink:
            status = subprocess.run(command, cwd=cwd, env={**os.environ, **env}, stdout=sink,
                                    stderr=None if stdout else subprocess.STDOUT).returncode
        manifest["commands"].append({"argv": command, "exit": status, "seconds": round(time.time() - start, 1)})
        if status:
            break
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_sums(out)
    return status


def write_sums(out):
    out = Path(out)
    lines = []
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(out).as_posix()}\n")
    (out / "SHA256SUMS").write_text("".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m tools.sparkscore.bench.run", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="harness_name", required=True)
    for name in ("smoke", "decision-index", "jevbench", "kev"):
        p = sub.add_parser(name)
        p.add_argument("--endpoint", default="http://127.0.0.1:8008")
        p.add_argument("--model", default="sparkscore-latest")
        p.add_argument("--out", required=name != "smoke")
        if name != "smoke":
            p.add_argument("--harness", required=True, help=f"checkout of the harness (written against {PINS[name][:12]})")
            p.add_argument("--python", default=sys.executable, help="interpreter with the harness installed")
        if name in ("decision-index", "jevbench"):
            p.add_argument("--limit", type=int)
        if name == "decision-index":
            p.add_argument("--rows", help="a rows file (e.g. from `suite sample`) instead of the full suite")
        if name == "kev":
            p.add_argument("--suite", help="suite directory relative to the kev checkout (default evals/v4/transfer-v4)")
            p.add_argument("--concurrency", type=int, default=1)
    a = ap.parse_args(argv)
    a.endpoint = a.endpoint.rstrip("/")
    if a.harness_name == "smoke":
        summary = smoke(a.endpoint, a.model, a.out)
        if a.out:
            (Path(a.out) / "raw" / "manifest.json").write_text(json.dumps({
                "harness": "sparkscore-smoke", "fixture": str(SMOKE.relative_to(ROOT)),
                "sparkscore_commit": _git(ROOT, "rev-parse", "HEAD"), "endpoint_info": _endpoint_info(a.endpoint),
                "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
            write_sums(a.out)
        print(json.dumps(summary, indent=2))
        return 1 if summary["failures"] else 0
    return harness(a)


if __name__ == "__main__":
    sys.exit(main())
