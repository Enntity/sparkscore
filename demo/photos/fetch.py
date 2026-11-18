# SPDX-License-Identifier: AGPL-3.0-only
"""Fetch the photo-demo library: COCO val2017 photos under CC BY 2.0 or CC BY-SA 2.0, listed through the
phiyodr/coco2017 index on Hugging Face's dataset server. Writes NNN.jpg and manifest.json (file, size,
license, Flickr source) here.   python3 demo/photos/fetch.py [N]"""
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LICENSES = {4: "CC BY 2.0", 5: "CC BY-SA 2.0"}
ROWS = "https://datasets-server.huggingface.co/rows?dataset=phiyodr%2Fcoco2017&config=default&split=validation&offset={}&length=100"


def main(n):
    picked, offset = [], 0
    while len(picked) < n and offset < 5000:
        with urllib.request.urlopen(ROWS.format(offset), timeout=60) as r:
            rows = json.load(r)["rows"]
        picked += [row["row"] for row in rows if row["row"]["license"] in LICENSES]
        offset += 100
    manifest = []
    for i, row in enumerate(picked[:n]):
        name = f"{i:03d}.jpg"
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(row["coco_url"], path)
        manifest.append({"file": name, "width": row["width"], "height": row["height"],
                         "license": LICENSES[row["license"]], "source": row["flickr_url"], "coco_id": row["image_id"],
                         "caption": row["captions"][0]})
    with open(os.path.join(HERE, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{len(manifest)} photos")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
