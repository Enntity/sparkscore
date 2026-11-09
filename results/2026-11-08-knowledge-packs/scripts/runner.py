"""Persistent d3-ultra BF16 session: load once, then exec jobs/<name>.py in a shared namespace.

Each job's stdout/stderr goes to jobs/<name>.out; jobs/<name>.done marks completion.
"""
import contextlib, glob, io, os, sys, time, traceback

ROOT = "/w/d3work/wsp"
sys.path.insert(0, ROOT)
import wsp  # noqa: E402

G = {"__name__": "__wsp__"}
t0 = time.time()
G.update(wsp.boot())
print(f"loaded in {time.time() - t0:.1f}s fast={G['m'].fast!r} state_once={type(G['m'].state_once).__name__}", flush=True)
open(f"{ROOT}/READY", "w").write(str(time.time()))
while True:
    jobs = sorted(p for p in glob.glob(f"{ROOT}/jobs/*.py") if not os.path.exists(p[:-3] + ".done"))
    if not jobs:
        time.sleep(1)
        continue
    job = jobs[0]
    buf = open(job[:-3] + ".out", "w", buffering=1)
    t = time.time()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            exec(compile(open(job).read(), job, "exec"), G)
        except Exception:
            traceback.print_exc()
        print(f"[job {os.path.basename(job)} {time.time() - t:.1f}s]")
    buf.close()
    open(job[:-3] + ".done", "w").write("1")
