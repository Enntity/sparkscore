# SPDX-License-Identifier: AGPL-3.0-only
"""The SparkScore container entry point: Atlas on 127.0.0.1:8081, the harness
on 0.0.0.0:8015, both from the serve block in checkpoints.json.

    serve.py [--checkpoint d3-ultra-nvfp4] [--model /model] [--precision w4a4|w4a8] [--packs N]
             [--knowledge-dir /knowledge]
    serve.py --print       show the two command lines and the engine environment, then exit

Atlas stays on the container's loopback; only the harness port is published.
If either process exits, the other is stopped and the container exits with
its status. SPARKSCORE_API_KEY, if set, is passed to the harness (bearer key).
--packs N (SPARKSCORE_PACKS) replaces the serve block's --score-packs (0: knowledge
packs off). Documents in --knowledge-dir (default /knowledge, mounted by
./start.sh from KNOWLEDGE_DIR) are loaded by name and their packs built at startup.
"""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ATLAS_PORT = 8081
# The one setting --precision changes: W4A8 keeps E4M3 tokens for the MLP and
# attention (LOWP=1); W4A4 (the default) uses NVFP4 tokens there.
PRECISION = {"w4a4": "gu,dn,qkv,o,qkvz,gout", "w4a8": "1"}


def commands(args):
    with open(os.path.join(HERE, "checkpoints.json")) as f:
        ck = json.load(f)["checkpoints"][args.checkpoint]
    serve = ck["serve"]
    env = dict(serve["atlas_env"])
    if args.precision:
        env["ATLAS_DENSE_PREFILL_LOWP"] = PRECISION[args.precision]
    flags = shlex.split(serve["atlas_flags"])
    if args.packs is not None:
        if "--score-packs" in flags:
            i = flags.index("--score-packs")
            del flags[i:i + 2]
        flags += ["--score-packs", str(args.packs)]
    atlas = ["spark", "serve", "--model-from-path", args.model, "--model-name", "m",
             "--port", str(ATLAS_PORT), *flags]
    harness = [sys.executable, "-m", "tools.sparkscore.server", "--backend", "score",
               "--backend-url", f"http://127.0.0.1:{ATLAS_PORT}", "--backend-model", "m",
               "--tokenizer", args.model, "--host", "0.0.0.0", "--port", str(args.port),
               *shlex.split(serve["harness_args"])]
    if args.knowledge_dir and os.path.isdir(args.knowledge_dir):
        harness += ["--knowledge-dir", args.knowledge_dir]
    return env, atlas, harness


def wait_healthy(url, proc, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(1)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.environ.get("SPARKSCORE_CHECKPOINT", "d3-ultra-nvfp4"))
    ap.add_argument("--model", default="/model", help="converted checkpoint directory (mounted)")
    ap.add_argument("--port", type=int, default=8015)
    ap.add_argument("--precision", choices=sorted(PRECISION), default=os.environ.get("SPARKSCORE_PRECISION") or None)
    ap.add_argument("--packs", type=int, default=int(os.environ["SPARKSCORE_PACKS"]) if os.environ.get("SPARKSCORE_PACKS") else None,
                    help="knowledge packs Atlas holds (replaces the serve block's --score-packs; 0: off)")
    ap.add_argument("--knowledge-dir", default=os.environ.get("SPARKSCORE_KNOWLEDGE_DIR", "/knowledge"))
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()
    env, atlas, harness = commands(args)
    if args.print:
        print("env:", " ".join(f"{k}={v}" for k, v in env.items()))
        print("atlas:", shlex.join(atlas))
        print("harness:", shlex.join(harness))
        return 0
    if not os.path.isfile(os.path.join(args.model, "config.json")):
        print(f"no checkpoint at {args.model}: mount the converted model there (./start.sh does)", file=sys.stderr)
        return 2

    procs = []

    def stop(*_):
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()

    signal.signal(signal.SIGTERM, lambda *_: (stop(), sys.exit(0)))
    signal.signal(signal.SIGINT, lambda *_: (stop(), sys.exit(0)))

    print("atlas:", shlex.join(atlas), flush=True)
    procs.append(subprocess.Popen(atlas, env={**os.environ, **env}))
    if not wait_healthy(f"http://127.0.0.1:{ATLAS_PORT}/health", procs[0], 900):
        print("Atlas did not become healthy", file=sys.stderr)
        stop()
        return 1
    print("harness:", shlex.join(harness), flush=True)
    procs.append(subprocess.Popen(harness, cwd=HERE))
    while True:
        for p in procs:
            if p.poll() is not None:
                print(f"{p.args[0]} exited with {p.returncode}; stopping", file=sys.stderr, flush=True)
                stop()
                return p.returncode or 1
        time.sleep(1)


if __name__ == "__main__":
    sys.exit(main())
