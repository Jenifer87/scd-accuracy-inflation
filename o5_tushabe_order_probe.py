"""
Objective 5 -- is there participant-level dependence hidden in Tushabe?

Tushabe documents 140 participants with about three exposures per smear,
but distributes no participant identifiers, and re-captures taken at a
slightly different stage position are not similar enough in pixels to be
grouped. If filenames preserve capture order, exposures of the same smear
will sit next to each other. Two independent tests follow from that.

  T1  MODEL-FREE   Are files adjacent in filename order more alike than
                   random pairs from the same class? Similarity is computed
                   as a function of lag. Same-smear triples predict high
                   similarity at lags 1-2 falling away by lag 3.

  T2  MODEL-BASED  Hold out consecutive blocks of k files, against random
                   blocks of the same size k. If consecutive files share a
                   smear, the consecutive scheme removes that shared
                   information from training and its AUC falls below the
                   random-block AUC. Random blocks are the control: they
                   are exactly as large but carry no order information.

A null result is informative too. It means either there is no
participant-level dependence, or the filename order was not preserved at
deposit -- and the second cannot be excluded without the source study.

Needs the Tushabe dataset attached. About 5 minutes.
"""

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import mannwhitneyu
from skimage.feature import local_binary_pattern, hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
OUT = "/kaggle/working"
ROOT = "/kaggle/input"
IMG_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
REPEATS = 15
BLOCKS = [2, 3, 4, 5, 6]
LAGS = [1, 2, 3, 4, 5, 8, 12, 20]
SIM, CENTRE = 64, 0.60


# ------------------------------------------------------------- data
def find_tushabe():
    for depth in range(1, 7):
        for p in glob.glob(ROOT + "/*" * depth):
            if os.path.isdir(p) and os.path.basename(p).lower() == "tushabe":
                return p
    return None


def ordinal(p):
    m = re.search(r"(\d+)", os.path.splitext(os.path.basename(p))[0])
    return int(m.group(1)) if m else None


base = find_tushabe()
if not base:
    raise SystemExit("Tushabe not found -- attach the dataset and re-run.")

rows = []
for d in sorted(glob.glob(f"{base}/*")):
    name = os.path.basename(d).lower()
    lab = 0 if ("normal" in name or "absent" in name) else (
          1 if ("sickle" in name or "present" in name) else None)
    if lab is None:
        continue
    for p in glob.glob(f"{d}/**/*", recursive=True):
        if p.lower().endswith(IMG_EXT) and ordinal(p) is not None:
            rows.append((p, lab, ordinal(p)))
D = (pd.DataFrame(rows, columns=["path", "y", "ord"])
       .sort_values(["y", "ord"]).reset_index(drop=True))
print("=" * 74)
print("TUSHABE -- PARTICIPANT-LEVEL DEPENDENCE FROM CAPTURE ORDER")
print("=" * 74)
for lab, tag in [(1, "sickle"), (0, "normal")]:
    o = D.loc[D.y == lab, "ord"].values
    gaps = np.diff(np.sort(o))
    print(f"  {tag:7s} {len(o):4d} files   ordinals {o.min()}-{o.max()}   "
          f"contiguous {bool(np.all(gaps == 1))}   largest gap {gaps.max() if len(gaps) else 0}")


# ------------------------------------------------------------- similarity
def prep(p):
    im = Image.open(p).convert("L")
    w, h = im.size
    im = im.crop((int(w * (1 - CENTRE) / 2), int(h * (1 - CENTRE) / 2),
                  int(w * (1 + CENTRE) / 2), int(h * (1 + CENTRE) / 2)))
    a = np.asarray(im.resize((SIM, SIM), Image.BILINEAR), np.float32)
    a = a - gaussian_filter(a, SIM / 8)
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        for m in (r, np.fliplr(r)):
            out.append(((m - m.mean()) / (m.std() + 1e-9)).ravel())
    return np.stack(out)                      # (8, d)


print("\n  preparing images for similarity ...", flush=True)
V = np.stack([prep(p) for p in D.path])       # (n, 8, d)
d = V.shape[2]


def sim(i, j):
    return float((V[j] @ V[i, 0]).max() / d)


# ================================================================== T1
print("\n" + "=" * 74)
print("T1  SIMILARITY AS A FUNCTION OF LAG IN FILENAME ORDER")
print("=" * 74)
rng = np.random.default_rng(0)
t1 = []
for lab, tag in [(1, "sickle"), (0, "normal")]:
    idx = np.where(D.y.values == lab)[0]          # already in ordinal order
    n = len(idx)
    rand = [sim(idx[a], idx[b]) for a, b in
            (rng.choice(n, 2, replace=False) for _ in range(3000))]
    r_med = np.median(rand)
    print(f"\n  {tag}   random within-class pairs: median {r_med:.3f}")
    print(f"  {'lag':>5s}  {'median':>7s}  {'vs random':>9s}   {'p':>8s}")
    for L in LAGS:
        if L >= n:
            continue
        s = [sim(idx[a], idx[a + L]) for a in range(n - L)]
        _, pv = mannwhitneyu(s, rand, alternative="greater")
        t1.append({"class": tag, "lag": L, "median": np.median(s),
                   "random_median": r_med, "p": pv})
        star = "  <-- above random" if pv < 0.001 else ""
        print(f"  {L:5d}  {np.median(s):7.3f}  {np.median(s) - r_med:+9.3f}   "
              f"{pv:8.1e}{star}")
T1 = pd.DataFrame(t1)
T1.to_csv(f"{OUT}/o5_tushabe_lag_similarity.csv", index=False)

print("""
  Reading. Same-smear exposures predict similarity well above random at
  the smallest lags, falling to the random level within a few files. A
  flat profile at the random level means adjacent files are no more alike
  than any two files -- no recoverable order structure.
""")


# ================================================================== T2
print("=" * 74)
print("T2  CONSECUTIVE BLOCKS vs RANDOM BLOCKS OF THE SAME SIZE")
print("=" * 74)


def features(p):
    a = np.asarray(Image.open(p).convert("L").resize((96, 96), Image.BILINEAR),
                   np.float32)
    a = (a - a.min()) / (np.ptp(a) + 1e-9)
    h_int = np.histogram(a, bins=32, range=(0, 1), density=True)[0]
    lbp = local_binary_pattern((a * 255).astype(np.uint8), 8, 1, "uniform")
    h_lbp = np.histogram(lbp, bins=10, range=(0, 10), density=True)[0]
    h_hog = hog(a, orientations=8, pixels_per_cell=(24, 24),
                cells_per_block=(1, 1), feature_vector=True)
    return np.concatenate([h_int, h_lbp, h_hog,
                           [a.mean(), a.std(), np.percentile(a, 10),
                            np.percentile(a, 90)]])


print("  extracting features ...", flush=True)
X = np.stack([features(p) for p in D.path])
y = D.y.values


def held_auc(groups, seed):
    try:
        splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed)
                      .split(X, y, groups))
    except Exception:
        return np.nan
    if any(len(np.unique(y[te])) < 2 for _, te in splits):
        return np.nan
    pr = np.zeros(len(y))
    for tr, te in splits:
        m = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        pr[te] = m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return roc_auc_score(y, pr)


def blocks(k, order_within_class):
    """Group ids: consecutive runs of k within each class, in the given order."""
    g = np.empty(len(y), dtype=object)
    for lab in (0, 1):
        idx = order_within_class[lab]
        for pos, i in enumerate(idx):
            g[i] = f"{lab}_{pos // k}"
    return g


naive = []
for r in range(REPEATS):
    cv = StratifiedKFold(5, shuffle=True, random_state=r)
    pr = cross_val_predict(RandomForestClassifier(n_estimators=100, random_state=r,
                                                  n_jobs=-1), X, y, cv=cv,
                           method="predict_proba")[:, 1]
    naive.append(roc_auc_score(y, pr))
print(f"\n  naive image-level split      AUC {np.mean(naive):.3f}  "
      f"[{np.percentile(naive, 2.5):.3f}, {np.percentile(naive, 97.5):.3f}]")

ordered = {lab: np.where(y == lab)[0] for lab in (0, 1)}
t2 = []
print(f"\n  {'k':>3s}  {'consecutive':>13s}  {'random blocks':>15s}  "
      f"{'difference':>10s}   P(consec < random)")
for k in BLOCKS:
    cons, rnd = [], []
    for r in range(REPEATS):
        cons.append(held_auc(blocks(k, ordered), r))
        shuffled = {lab: np.random.default_rng(1000 + r).permutation(v)
                    for lab, v in ordered.items()}
        rnd.append(held_auc(blocks(k, shuffled), r))
    cons, rnd = np.array(cons, float), np.array(rnd, float)
    ok = np.isfinite(cons) & np.isfinite(rnd)
    cons, rnd = cons[ok], rnd[ok]
    diff = rnd.mean() - cons.mean()
    p_less = float(np.mean(cons[:, None] < rnd[None, :]))
    t2.append({"k": k, "consecutive": cons.mean(), "random": rnd.mean(),
               "difference": diff, "P_consec_below_random": p_less})
    print(f"  {k:3d}  {cons.mean():13.3f}  {rnd.mean():15.3f}  {diff:+10.3f}   "
          f"{p_less:.2f}")
T2 = pd.DataFrame(t2)
T2.to_csv(f"{OUT}/o5_tushabe_block_test.csv", index=False)

print("""
  Reading. If consecutive files share a smear, holding out consecutive
  blocks removes that shared information from training, so the consecutive
  AUC falls below the random-block AUC, and the gap should be largest near
  k = 3. P(consec < random) near 1 across k means the order carries
  participant structure. Near 0.5 means it does not.

  If T1 shows above-random similarity at small lags AND T2 shows the
  consecutive AUC below random, there is participant-level dependence in
  Tushabe that pixel-level source recovery missed, and naive evaluation on
  this corpus is inflated by roughly the random-minus-consecutive gap.
""")
print("wrote o5_tushabe_lag_similarity.csv, o5_tushabe_block_test.csv")
