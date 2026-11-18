"""Pre-resize the COCO photos to the processor's own target size (so neither engine resizes) and save as PNG."""
import glob, math, os
from PIL import Image
os.makedirs("/w/d3work/img/coco_aligned", exist_ok=True)
F, MIN, MAX = 32, 65536, 1638400
def fit(h, w):  # Qwen smart_resize
    hb, wb = round(h / F) * F, round(w / F) * F
    if hb * wb > MAX:
        b = math.sqrt(h * w / MAX); hb, wb = math.floor(h / b / F) * F, math.floor(w / b / F) * F
    elif hb * wb < MIN:
        b = math.sqrt(MIN / (h * w)); hb, wb = math.ceil(h * b / F) * F, math.ceil(w * b / F) * F
    return hb, wb
for f in sorted(glob.glob("/w/d3work/img/coco/*.jpg")):
    im = Image.open(f).convert("RGB"); h, w = fit(im.height, im.width)
    im.resize((w, h), Image.BICUBIC).save(f"/w/d3work/img/coco_aligned/{os.path.basename(f)[:-4]}.png")
    print(os.path.basename(f), im.size, "->", (w, h))
