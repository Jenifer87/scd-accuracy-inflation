"""
Regenerate the R3 annotation sheets LARGE enough to judge individual cells.

The original sheets packed ten pairs into one tall image; at that size the
only visible cue is colour, which is exactly what must not decide a label.
This writes one pair per image, side by side, near full resolution.

It reads the key only to find which class folder each file is in (file
names repeat across the Normal and Sickle folders). No similarity score is
printed or drawn, so labelling stays blind.

Run with the Tushabe dataset attached and the o5_closing folder present.
"""

import os
import glob
import zipfile
import pandas as pd
from PIL import Image

ROOT = "/kaggle/input"
SRC = "/kaggle/working/o5_closing"
OUT = "/kaggle/working/o5_annotate_large"
os.makedirs(OUT, exist_ok=True)


def find_dir(name):
    for depth in range(1, 8):
        for p in glob.glob(ROOT + "/*" * depth):
            if os.path.isdir(p) and os.path.basename(p).lower() == name.lower():
                return p


base = find_dir("Tushabe")
folders = {}
for d in glob.glob(f"{base}/*"):
    n = os.path.basename(d).lower()
    if "normal" in n:
        folders[0] = d
    elif "sickle" in n:
        folders[1] = d

keyf = glob.glob(f"{SRC}/annotation_key_*.csv") or \
       glob.glob("/kaggle/**/annotation_key_*.csv", recursive=True)
K = pd.read_csv(keyf[0])[["pair", "file_a", "file_b", "class_a", "class_b"]]


def locate(fname, cls):
    hits = glob.glob(f"{folders[int(cls)]}/**/{fname}", recursive=True)
    return hits[0] if hits else None


H = 900                                        # height of each image in the sheet
for r in K.itertuples():
    pa, pb = locate(r.file_a, r.class_a), locate(r.file_b, r.class_b)
    if not pa or not pb:
        print(f"  pair {r.pair}: file not found, skipped")
        continue
    ims = []
    for p in (pa, pb):
        im = Image.open(p).convert("RGB")
        im = im.resize((int(im.width * H / im.height), H), Image.LANCZOS)
        ims.append(im)
    gap = 40
    canvas = Image.new("RGB", (ims[0].width + ims[1].width + gap, H + 70), "white")
    canvas.paste(ims[0], (0, 70))
    canvas.paste(ims[1], (ims[0].width + gap, 70))
    try:
        from PIL import ImageDraw, ImageFont
        d = ImageDraw.Draw(canvas)
        try:
            f = ImageFont.truetype("DejaVuSans-Bold.ttf", 40)
        except Exception:
            f = ImageFont.load_default()
        d.text((20, 12), f"PAIR {r.pair}", fill="black", font=f)
    except Exception:
        pass
    canvas.save(f"{OUT}/pair_{r.pair:02d}.jpg", quality=92)

pd.read_csv(glob.glob(f"{SRC}/annotation_template.csv")[0]).to_csv(
    f"{OUT}/annotation_template.csv", index=False)
with zipfile.ZipFile(OUT + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(glob.glob(f"{OUT}/*")):
        z.write(p, os.path.basename(p))
print(f"wrote {len(glob.glob(f'{OUT}/pair_*.jpg'))} pair images -> {OUT}.zip")
print("""
LABELLING GUIDE
  SAME  the two images show the same field: the same cells in the same
        relative positions. Allow for shift, rotation, flip, zoom, blur and
        different exposure or colour.
  DIFF  different cells, even if the stain, density and brightness match.

  Check two or three distinctive landmarks -- a clump, an odd-shaped cell,
  a white cell, a gap -- and ask whether they reappear in the same pattern.
  If you cannot decide, write UNSURE rather than guessing; those are scored
  separately. Label all 60 before opening the key file.
""")
try:
    from IPython.display import FileLink, display
    display(FileLink(os.path.relpath(OUT + ".zip", "/kaggle/working")))
except Exception:
    pass
