# SPDX-License-Identifier: AGPL-3.0-only
"""Record demo scenes in autoplay to H.264 clips with headless Chromium.

    python3 demo/record.py inbox guard typing claims title:intro title:outro
    python3 demo/record.py --montage
    python3 demo/record.py title:s5 packs title:packs --cut knowledge-packs   # the knowledge-pack feature video

Needs demo/serve.py running (default http://127.0.0.1:8780) and Python
Playwright. Frames come from a CDP screencast at high JPEG quality and are
re-timed to constant 30 fps by their capture timestamps, then encoded with
ffmpeg (H.264 high, yuv420p, CRF 18, faststart). Clips go to demo/out/.
"""

import argparse
import asyncio
import base64
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FPS = 30
ENCODE = ["-c:v", "libx264", "-profile:v", "high", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-color_range", "tv",
          "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-movflags", "+faststart"]


async def capture(url, frames_dir, tail_ms=700, timeout_s=180):
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--force-color-profile=srgb", "--disable-background-timer-throttling", "--disable-renderer-backgrounding"])
        page = await browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        cdp = await page.context.new_cdp_session(page)
        frames = []

        def on_frame(params):
            frames.append(params["metadata"]["timestamp"])
            n = len(frames) - 1
            with open(os.path.join(frames_dir, f"{n:06d}.jpg"), "wb") as f:
                f.write(base64.b64decode(params["data"]))
            ack = asyncio.ensure_future(cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]}))
            ack.add_done_callback(lambda t: t.cancelled() or t.exception())  # acks racing the close are fine

        cdp.on("Page.screencastFrame", on_frame)
        await page.goto(url)
        await page.wait_for_function("window.__ready === true", timeout=30000)
        await cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 95, "maxWidth": 1920, "maxHeight": 1080, "everyNthFrame": 1})
        await page.wait_for_function("window.__demoDone === true", timeout=timeout_s * 1000, polling=100)
        await asyncio.sleep(tail_ms / 1000)
        await cdp.send("Page.stopScreencast")
        calls = await page.evaluate("window.__calls")
        err = await page.evaluate("window.__demoError || null")
        await browser.close()
    if err or errors:
        raise RuntimeError(f"page error: {err or errors}")
    return frames, calls


def encode(frames, frames_dir, out_path, skip_s=0.0, fade=True):
    """Re-time screencast frames (capture timestamps) to constant FPS."""
    t0 = frames[0] + skip_s
    end = frames[-1]
    n_out = int((end - t0) * FPS)
    concat = os.path.join(frames_dir, "list.ffconcat")
    k = 0
    with open(concat, "w") as f:
        f.write("ffconcat version 1.0\n")
        last = None
        for i in range(n_out):
            t = t0 + i / FPS
            while k + 1 < len(frames) and frames[k + 1] <= t:
                k += 1
            f.write(f"file {k:06d}.jpg\nduration {1 / FPS:.6f}\n")
            last = k
        f.write(f"file {last:06d}.jpg\n")
    dur = n_out / FPS
    vf = ["scale=in_range=pc:out_range=tv", "format=yuv420p"]  # JPEG frames are full range; H.264 for the web is limited range
    if fade:
        vf = [f"fade=t=in:st=0:d=0.35", f"fade=t=out:st={max(0, dur - 0.45):.3f}:d=0.45"] + vf
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", concat,
                    "-vf", ",".join(vf), "-r", str(FPS), *ENCODE, out_path], check=True)
    return dur


def warm(base):
    """One untimed request first, so the GPU leaves its idle clocks before the
    clip starts; a serving box under traffic is never idle. Not recorded."""
    body = json.dumps({"model": "sparkscore-d3", "state": "Warm-up.",
                       "questions": {"q": {"type": "noul", "instructions": "Is this a warm-up?"}}}).encode()
    req = urllib.request.Request(f"{base}/v1/systemone", body, {"content-type": "application/json"})
    for _ in range(3):
        urllib.request.urlopen(req, timeout=60).read()


def record(name, base):
    scene, _, card = name.partition(":")
    if scene != "title":
        warm(base)
    query = f"scene={scene}&autoplay=1&lead=400" + (f"&card={card}" if card else "")
    if scene == "title":
        query += "&hold=" + ("5200" if card in ("outro", "packs") else "3600")
    out_path = os.path.join(OUT, f"{scene}-{card}.mp4" if card else f"{scene}.mp4")
    with tempfile.TemporaryDirectory(prefix="sparkscore-frames-") as tmp:
        frames, calls = asyncio.run(capture(f"{base}/?{query}", tmp))
        span = frames[-1] - frames[0]
        dur = encode(frames, tmp, out_path, skip_s=0.3 if scene != "title" else 0.0)
    with open(out_path.replace(".mp4", ".calls.json"), "w") as f:
        json.dump(calls, f)
    print(f"{out_path}: {dur:.1f}s, {len(frames)} frames captured ({len(frames) / span:.1f}/s), {len(calls)} live calls")


# Cuts: (clip, start s, end s) from the recorded clips, each with a short
# fade, joined with dips to black.
CUTS = {
    "montage": [
        ("title-intro", 0.0, None),
        ("inbox", 0.0, 19.5),
        ("guard", 0.0, 16.5),
        ("typing", 4.5, None),
        ("claims", 1.0, 22.0),
        ("title-outro", 0.0, None),
    ],
    "knowledge-packs": [
        ("title-s5", 0.0, None),
        ("packs", 0.0, None),
        ("title-packs", 0.0, None),
    ],
}


def probe(path):
    out = subprocess.run([FFMPEG.replace("ffmpeg", "ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True)
    return float(out.stdout)


def montage(cut="montage"):
    clips = CUTS[cut]
    inputs, parts = [], []
    for k, (clip, start, end) in enumerate(clips):
        path = os.path.join(OUT, f"{clip}.mp4")
        end = probe(path) if end is None else end
        d = end - start
        inputs += ["-i", path]
        parts.append(f"[{k}:v]trim=start={start}:end={end},setpts=PTS-STARTPTS,"
                     f"fade=t=in:st=0:d=0.3,fade=t=out:st={d - 0.35:.3f}:d=0.35[v{k}]")
    graph = ";".join(parts) + ";" + "".join(f"[v{k}]" for k in range(len(clips))) + f"concat=n={len(clips)}:v=1:a=0,format=yuv420p[out]"
    out_path = os.path.join(OUT, f"sparkscore-{cut}.mp4")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", *inputs, "-filter_complex", graph, "-map", "[out]", "-r", str(FPS), *ENCODE, out_path], check=True)
    print(f"{out_path}: {probe(out_path):.1f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--base", default="http://127.0.0.1:8780")
    ap.add_argument("--montage", action="store_true", help="cut the montage from recorded clips")
    ap.add_argument("--cut", choices=sorted(CUTS), help="cut a named sequence from recorded clips")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for name in args.scenes:
        record(name, args.base)
    if args.montage:
        montage()
    if args.cut:
        montage(args.cut)


if __name__ == "__main__":
    main()
