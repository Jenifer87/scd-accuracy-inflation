"""
Tushabe -- which copy carries annotation boxes, on which images, and do
they change detection accuracy?

Two copies are attached:
  A  biomorphnet-scd-datasets/Tushabe   (Normal / Sickle)     -- used in O5
  B  tushabe/Tushabe                    (Normal_RBC / Sickled_RBC)

Method: match every image in A to its counterpart in B by content, then
subtract. Where an annotator drew a box, the copies differ -- and only there.
This isolates the marks directly, without assuming what a box looks like.

  1  inventory and exact-duplicate check between copies
  2  content matching A <-> B
  3  differencing: which images carry drawn marks, in which copy, which class
  4  contact sheet of flagged images for visual confirmation
  5  detection AUC on each copy, and on A with marked images removed

Output: /kaggle/working/tushabe_copies/ , zipped. ~10-15 minutes.
"""

import os
import glob
import hashlib
import zipfile
import warnings
import collections
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage.feature import local_binary_pattern, hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = "/kaggle/input"
OUT = "/kaggle/working/tushabe_copies"
os.makedirs(OUT, exist_ok=True)
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def tushabe_dirs():
    hits = [p for p in glob.glob(ROOT + "/**/*", recursive=True)
            if os.path.isdir(p) and os.path.basename(p).lower() == "tushabe"]
    good = []
    for h in hits:
        subs = [d for d in glob.glob(h + "/*") if os.path.isdir(d)]
        names = [os.path.basename(d).lower() for d in subs]
        if any("normal" in s for s in names) and any("sickle" in s for s in names):
            good.append(h)
    return good


def load_copy(root):
    rows = []
    for d in sorted(glob.glob(root + "/*")):
        if not os.path.isdir(d):
            continue
        n = os.path.basename(d).lower()
        lab = 0 if "normal" in n else (1 if "sickle" in n else None)
        if lab is None:
            continue
        rows += [(p, lab) for p in sorted(glob.glob(d + "/**/*", recursive=True))
                 if p.lower().endswith(IMG_EXT)]
    return pd.DataFrame(rows, columns=["path", "y"])


dirs = tushabe_dirs()
if len(dirs) < 2:
    raise SystemExit(f"Need both Tushabe copies attached; found {dirs}")
dirs.sort(key=lambda p: 0 if "biomorphnet" in p.lower() else 1)
A, B = load_copy(dirs[0]), load_copy(dirs[1])
print("=" * 76)
print("1  INVENTORY")
print("=" * 76)
for tag, root, D in (("A", dirs[0], A), ("B", dirs[1], B)):
    print(f"  copy {tag}: {root}")
    print(f"          {len(D)} images   sickle {int(D.y.sum())}   normal {int((1 - D.y).sum())}")


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


A["md5"] = [md5(p) for p in A.path]
B["md5"] = [md5(p) for p in B.path]
common = set(A.md5) & set(B.md5)
print(f"\n  byte-identical files present in both copies: {len(common)}")
for lab, nm in ((1, "sickle"), (0, "normal")):
    a_ = A[A.y == lab]
    print(f"     {nm:6s}: {int(a_.md5.isin(common).sum())} of {len(a_)} images in A "
          f"are byte-identical to an image in B")


# =====================================================================
# 2  content matching
# =====================================================================
print("\n" + "=" * 76)
print("2  MATCHING EACH IMAGE IN A TO ITS COUNTERPART IN B")
print("=" * 76)
T = 64


def thumb(p):
    a = np.asarray(Image.open(p).convert("L").resize((T, T), Image.BILINEAR), np.float32)
    a = (a - a.mean()) / (a.std() + 1e-9)
    return a.ravel() / np.sqrt(T * T)


VA = np.stack([thumb(p) for p in A.path])
VB = np.stack([thumb(p) for p in B.path])
match = np.full(len(A), -1)
score = np.zeros(len(A))
for lab in (0, 1):
    ia = np.where(A.y.values == lab)[0]
    ib = np.where(B.y.values == lab)[0]
    if not len(ib):
        continue
    C = VA[ia] @ VB[ib].T
    for k, i in enumerate(ia):
        j = int(np.argmax(C[k]))
        match[i], score[i] = ib[j], C[k, j]
A["match_b"] = match
A["match_score"] = score
good = A.match_score > 0.90
print(f"  matched with correlation > 0.90: {int(good.sum())} of {len(A)}")
print(f"  median match correlation: {np.median(score):.3f}")
print(f"  unmatched (< 0.90): {int((~good).sum())}  -- present in one copy only, or heavily changed")


# =====================================================================
# 3  differencing
# =====================================================================
print("\n" + "=" * 76)
print("3  WHERE DO THE COPIES DIFFER? -- drawn marks isolated by subtraction")
print("=" * 76)
W = 768


def marks(pa, pb):
    """Return (marked_in_A, marked_in_B, line_pixels_A, line_pixels_B)."""
    ia = Image.open(pa).convert("RGB")
    ib = Image.open(pb).convert("RGB").resize(ia.size, Image.BILINEAR)
    s = W / max(ia.size)
    size = (max(1, int(ia.width * s)), max(1, int(ia.height * s)))
    a = np.asarray(ia.resize(size, Image.BILINEAR), np.int16)
    b = np.asarray(ib.resize(size, Image.BILINEAR), np.int16)
    ga, gb = a.mean(2), b.mean(2)
    L = max(12, int(0.03 * size[0]))
    out = []
    for g_own, g_other, arr in ((ga, gb, a), (gb, ga, b)):
        dark = (arr.max(2) < 90) & ((arr.max(2) - arr.min(2)) < 45)
        new = dark & ((g_other - g_own) > 45)          # dark here, not dark in the other copy
        h = ndimage.binary_opening(new, structure=np.ones((1, L)))
        v = ndimage.binary_opening(new, structure=np.ones((L, 1)))
        out.append((h.sum() > 0 and v.sum() > 0, int(h.sum() + v.sum())))
    return out[0][0], out[1][0], out[0][1], out[1][1]


res = []
mk = A[good].index
for n_, i in enumerate(mk):
    ma, mb, la, lb = marks(A.path[i], B.path[A.match_b[i]])
    res.append({"a_path": A.path[i], "b_path": B.path[A.match_b[i]], "y": int(A.y[i]),
                "identical": A.md5[i] == B.md5[A.match_b[i]],
                "marked_in_A": ma, "marked_in_B": mb, "line_px_A": la, "line_px_B": lb})
    if (n_ + 1) % 100 == 0:
        print(f"   {n_ + 1}/{len(mk)}", flush=True)
R = pd.DataFrame(res)
R.to_csv(f"{OUT}/copy_differences.csv", index=False)

for lab, nm in ((1, "sickle"), (0, "normal")):
    r = R[R.y == lab]
    print(f"  {nm:6s}: {len(r)} matched pairs   marks only in A: {int((r.marked_in_A & ~r.marked_in_B).sum())}"
          f"   marks only in B: {int((r.marked_in_B & ~r.marked_in_A).sum())}"
          f"   identical files: {int(r.identical.sum())}")
labelled = "A" if R.marked_in_A.sum() > R.marked_in_B.sum() else "B"
print(f"\n  => the LABELLED copy is {labelled} "
      f"({dirs[0] if labelled == 'A' else dirs[1]})")


# =====================================================================
# 4  contact sheet
# =====================================================================
col = "marked_in_A" if labelled == "A" else "marked_in_B"
own = "a_path" if labelled == "A" else "b_path"
oth = "b_path" if labelled == "A" else "a_path"
F = R[R[col]].sort_values("line_px_" + labelled, ascending=False).head(8)
if len(F):
    fig, ax = plt.subplots(2, len(F), figsize=(3 * len(F), 6.4))
    ax = np.atleast_2d(ax)
    for c_, r in enumerate(F.itertuples()):
        for r_, p in enumerate((getattr(r, own), getattr(r, oth))):
            ax[r_, c_].imshow(Image.open(p).convert("RGB"))
            ax[r_, c_].set_xticks([]); ax[r_, c_].set_yticks([])
        ax[0, c_].set_title("sickle" if r.y else "normal", fontsize=9)
    ax[0, 0].set_ylabel("LABELLED copy", fontsize=10)
    ax[1, 0].set_ylabel("other copy", fontsize=10)
    fig.suptitle("Images flagged as carrying drawn marks -- confirm by eye", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/flagged_marks.png", dpi=100)
    plt.close(fig)
    print("  wrote flagged_marks.png -- check that the top row shows boxes the bottom row lacks")


# =====================================================================
# 5  do the marks change detection accuracy?
# =====================================================================
print("\n" + "=" * 76)
print("5  DETECTION ACCURACY: labelled copy vs clean copy")
print("=" * 76)


def feat(p):
    a = np.asarray(Image.open(p).convert("L").resize((96, 96), Image.BILINEAR), np.float32)
    a = (a - a.min()) / (np.ptp(a) + 1e-9)
    return np.concatenate([
        np.histogram(a, 32, (0, 1), density=True)[0],
        np.histogram(local_binary_pattern((a * 255).astype(np.uint8), 8, 1, "uniform"),
                     10, (0, 10), density=True)[0],
        hog(a, orientations=8, pixels_per_cell=(24, 24), cells_per_block=(1, 1),
            feature_vector=True),
        [a.mean(), a.std(), np.percentile(a, 10), np.percentile(a, 90)]])


def auc(paths, y, reps=10):
    X = np.stack([feat(p) for p in paths])
    v = [roc_auc_score(y, cross_val_predict(
        RandomForestClassifier(n_estimators=150, random_state=r, n_jobs=-1), X, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=r), method="predict_proba")[:, 1])
        for r in range(reps)]
    return np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5)


out = {}
for tag, D in (("A", A), ("B", B)):
    m, lo, hi = auc(D.path.tolist(), D.y.values)
    out[tag] = m
    print(f"  copy {tag}, all images          AUC {m:.3f} [{lo:.3f}, {hi:.3f}]")
marked_paths = set(R.loc[R[col], own])
Lab = A if labelled == "A" else B
keep = ~Lab.path.isin(marked_paths)
m, lo, hi = auc(Lab.path[keep].tolist(), Lab.y.values[keep])
print(f"  copy {labelled}, marked images removed ({int((~keep).sum())} dropped)"
      f"  AUC {m:.3f} [{lo:.3f}, {hi:.3f}]")
print(f"\n  labelled minus clean copy: {out[labelled] - out['B' if labelled == 'A' else 'A']:+.3f}")
pd.DataFrame([{"auc_A": out["A"], "auc_B": out["B"], "labelled_copy": labelled,
               "marked_images": int((~keep).sum()),
               "auc_labelled_marked_removed": m}]).to_csv(f"{OUT}/accuracy_by_copy.csv", index=False)
print("""
  Reading. If marks sit on sickle images only and the labelled copy scores
  higher than the clean copy, the drawn boxes are a label shortcut, and every
  O5 figure computed on the labelled copy carries that shortcut. Removing the
  marked images, or switching to the clean copy, is then required.
""")

with zipfile.ZipFile(OUT + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in glob.glob(OUT + "/*"):
        z.write(p, os.path.basename(p))
try:
    from IPython.display import FileLink, display
    display(FileLink(os.path.relpath(OUT + ".zip", "/kaggle/working")))
except Exception:
    pass
