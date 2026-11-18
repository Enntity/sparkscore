# SPDX-License-Identifier: AGPL-3.0-only
"""Fetch the video demo's film: Tears of Steel (Blender Foundation, mango.blender.org, CC BY 3.0), 720p,
cut to its story (the credits dropped) as a browser-ready H.264 MP4: tears_of_steel.mp4 here.
    python3 demo/videos/fetch.py"""
import os
import shutil
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = "https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov"
STORY_S = 592  # the credits start at 9:52


def main():
    raw = os.path.join(HERE, "tears_of_steel_720p.mov")
    out = os.path.join(HERE, "tears_of_steel.mp4")
    if not os.path.exists(raw):
        print(f"downloading {SOURCE} (372 MB)")
        urllib.request.urlretrieve(SOURCE, raw + ".part")
        os.replace(raw + ".part", raw)
    ffmpeg = shutil.which("ffmpeg") or sys.exit("needs ffmpeg")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", raw, "-t", str(STORY_S),
                    "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out], check=True)
    print(f"{out}: {os.path.getsize(out) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
