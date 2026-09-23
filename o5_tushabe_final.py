"""
OBJECTIVE 5 -- final Tushabe estimate, threshold anchored to human labels

The annotation study showed that at 128 px:
    similarity < 0.45     P(same field) = 0.00
    0.45 - 0.55           P(same field) = 0.75
    >= 0.55               P(same field) = 1.00    (consensus, both raters)
    AUC 0.978 against the consensus reference

The earlier calibrated threshold, 0.648, came from SYNTHETIC copies and
recovered only 66% of genuine repeats. Real re-captures differ in exposure,
focus, zoom and vignette, so they score lower than planted copies. This cell
regroups Tushabe at human-anchored thresholds and re-estimates inflation.

Standalone: needs only the Tushabe dataset attached. ~15 minutes.
Output: /kaggle/working/o5_tushabe_final/ , zipped at the end.
"""

import os
import glob
import zipfile
import warnings
import collections
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.feature import local_binary_pattern, hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = "/kaggle/input"
OUT = "/kaggle/working/o5_tushabe_final"
os.makedirs(OUT, exist_ok=True)
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
REPEATS = 12
THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.648]     # 0.55 is the human-anchored choice
PRIMARY = 0.55
S = 128


def find_tushabe():
    """Pick the Tushabe folder that directly holds Normal and Sickle
    subfolders. Prefer the BioMorphNet copy so results stay consistent
    with every earlier run."""
    hits = [p for p in glob.glob(ROOT + "/**/*", recursive=True)
            if os.path.isdir(p) and os.path.basename(p).lower() == "tushabe"]
    good = []
    for h in hits:
        subs = [os.path.basename(d).lower() for d in glob.glob(h + "/*") if os.path.isdir(d)]
        if any("normal" in s for s in subs) and any("sickle" in s for s in subs):
            good.append(h)
    if not good:
        print("No Tushabe folder containing Normal and Sickle subfolders was found.")
        for h in hits:
            print("   candidate:", h, [os.path.basename(d) for d in glob.glob(h + "/*")])
        raise SystemExit("Attach biomorphnet-scd-datasets and re-run.")
    good.sort(key=lambda p: (0 if "biomorphnet" in p.lower() else 1, len(p)))
    if len(good) > 1:
        print("Several usable Tushabe copies found; using the first:")
        for g_ in good:
            print("   ", g_)
    return good[0]


base = find_tushabe()
print(f"Tushabe folder: {base}")
rows = []
for d in sorted(glob.glob(f"{base}/*")):
    if not os.path.isdir(d):
        continue
    n_ = os.path.basename(d).lower()
    lab = 0 if ("normal" in n_ or "absent" in n_) else (
          1 if ("sickle" in n_ or "present" in n_) else None)
    print(f"   subfolder {os.path.basename(d):12s} -> "
          f"{'normal' if lab == 0 else 'sickle' if lab == 1 else 'ignored'}")
    if lab is not None:
        rows += [(p, lab) for p in sorted(glob.glob(f"{d}/**/*", recursive=True))
                 if p.lower().endswith(IMG_EXT)]
if not rows:
    raise SystemExit("Found the Tushabe folder but no images inside Normal/Sickle subfolders.")
paths = [p for p, _ in rows]
y = np.array([l for _, l in rows])
print(f"Tushabe: {len(paths)} images (sickle {y.sum()}, normal {(1 - y).sum()})")

# ------------------------------------------------ similarity at 128 px, all pairs
ms = int(.15 * S)
mask = np.zeros((S, S), bool)
for dy in range(-ms, ms + 1):
    for dx in range(-ms, ms + 1):
        mask[dy % S, dx % S] = True
H = np.outer(np.hanning(S), np.hanning(S)).astype(np.float32)


def spec(p):
    im = Image.open(p).convert("L")
    w, h = im.size
    im = im.crop((int(w * .2), int(h * .2), int(w * .8), int(h * .8)))
    a = np.asarray(im.resize((S, S), Image.BILINEAR), np.float32)
    a = a - gaussian_filter(a, S / 8)
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        for m in (r, np.fliplr(r)):
            m = (m - m.mean()) / (m.std() + 1e-9) * H
            out.append(m / (np.sqrt((m ** 2).sum()) + 1e-9))
    return np.fft.rfft2(np.stack(out)).astype(np.complex64)


print("computing all-pairs similarity at 128 px ...", flush=True)
F = np.stack([spec(p) for p in paths])
n = len(paths)
Sim = np.full((n, n), -1, np.float32)
for i in range(n):
    xc = np.fft.irfft2(F * np.conj(F[i, 0])[None, None], s=(S, S))
    Sim[i] = xc[:, :, mask].max(axis=(1, 2))
    if (i + 1) % 100 == 0:
        print(f"   {i + 1}/{n}", flush=True)
np.fill_diagonal(Sim, -1)
np.save(f"{OUT}/similarity_128.npy", Sim)


def groups(thr, topk=None):
    """Connected components of pairs above thr. With topk, keep a link only
    if each image is among the other's topk matches (no chaining)."""
    parent = list(range(n))

    def f(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    if topk:
        top = [set(np.argsort(-Sim[i])[:topk]) for i in range(n)]
    for i, j in zip(*np.where(np.triu(Sim >= thr, 1))):
        if topk and not (j in top[i] and i in top[j]):
            continue
        a, b = f(i), f(j)
        if a != b:
            parent[b] = a
    return np.array([f(i) for i in range(n)])


def feat_gray(p):
    a = np.asarray(Image.open(p).convert("L").resize((96, 96), Image.BILINEAR), np.float32)
    a = (a - a.min()) / (np.ptp(a) + 1e-9)
    return np.concatenate([
        np.histogram(a, 32, (0, 1), density=True)[0],
        np.histogram(local_binary_pattern((a * 255).astype(np.uint8), 8, 1, "uniform"),
                     10, (0, 10), density=True)[0],
        hog(a, orientations=8, pixels_per_cell=(24, 24), cells_per_block=(1, 1),
            feature_vector=True),
        [a.mean(), a.std(), np.percentile(a, 10), np.percentile(a, 90)]])


def feat_color(p):
    a = np.asarray(Image.open(p).convert("RGB").resize((24, 24), Image.BILINEAR), np.float32) / 255
    return np.concatenate([np.concatenate([np.histogram(a[..., c], 16, (0, 1), density=True)[0]
                                           for c in range(3)]),
                           a.mean(2).ravel(), a.reshape(-1, 3).mean(0), a.reshape(-1, 3).std(0)])


def learner(kind, seed=0):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=-1)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


def naive(X, kind):
    return float(np.mean([roc_auc_score(y, cross_val_predict(
        learner(kind, r), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=r),
        method="predict_proba")[:, 1]) for r in range(REPEATS)]))


def held(X, g, kind):
    m = min(len(np.unique(g[y == c])) for c in (0, 1))
    if m < 3:
        return np.nan, np.nan, np.nan
    v = []
    for r in range(REPEATS):
        try:
            sp = list(StratifiedGroupKFold(int(min(5, m)), shuffle=True,
                                           random_state=r).split(X, y, g))
        except Exception:
            continue
        if any(len(np.unique(y[te])) < 2 for _, te in sp):
            continue
        pr = np.zeros(n)
        for tr, te in sp:
            pr[te] = learner(kind, r).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        v.append(roc_auc_score(y, pr))
    return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


print("extracting features ...", flush=True)
X = {"grey": np.stack([feat_gray(p) for p in paths]),
     "colour": np.stack([feat_color(p) for p in paths])}
NAIVE = {(fk, lk): naive(X[fk], lk) for fk in X for lk in ("rf", "lr")}

res = []
print("\n" + "=" * 78)
print("TUSHABE INFLATION BY GROUPING THRESHOLD")
print("=" * 78)
for thr in THRESHOLDS:
    for mode, tk in (("all links", None), ("mutual top-3", 3)):
        g = groups(thr, tk)
        ns = len(np.unique(g))
        sizes = collections.Counter(g)
        mixed = sum(1 for s in sizes if len(set(y[g == s])) > 1)
        rec = {"threshold": thr, "linking": mode, "sources": ns,
               "expansion": round(n / ns, 2), "largest_group": max(sizes.values()),
               "mixed_label_groups": mixed}
        tag = "  <-- human-anchored" if thr == PRIMARY else ""
        print(f"\n thr {thr:.3f}  {mode:12s}  sources {ns:3d}  expansion {n / ns:.2f}x  "
              f"largest group {max(sizes.values()):3d}  mixed-label {mixed}{tag}")
        for fk in X:
            for lk in ("rf", "lr"):
                h = held(X[fk], g, lk)
                inf = NAIVE[(fk, lk)] - h[0]
                rec[f"naive_{fk}_{lk}"] = round(NAIVE[(fk, lk)], 3)
                rec[f"held_{fk}_{lk}"] = round(h[0], 3)
                rec[f"infl_{fk}_{lk}"] = round(inf, 3)
                print(f"     {fk:6s} {lk}: naive {NAIVE[(fk, lk)]:.3f}  held {h[0]:.3f} "
                      f"[{h[1]:.3f}, {h[2]:.3f}]  inflation {inf:+.3f}")
        res.append(rec)
        pd.DataFrame(res).to_csv(f"{OUT}/tushabe_inflation_by_threshold.csv", index=False)

# the most-repeated fields, for the paper
g = groups(PRIMARY, 3)
big = [s for s, c in collections.Counter(g).most_common(8) if c > 1]
lines = []
for s in big:
    idx = np.where(g == s)[0]
    lines.append({"group": int(s), "size": len(idx),
                  "files": ";".join(os.path.basename(paths[i]) for i in idx),
                  "classes": ";".join(str(int(y[i])) for i in idx)})
pd.DataFrame(lines).to_csv(f"{OUT}/most_repeated_fields.csv", index=False)
print("\nmost-repeated fields at the human-anchored threshold:")
for L in lines:
    print(f"   group of {L['size']:2d}  classes {L['classes']}")

with zipfile.ZipFile(OUT + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in glob.glob(f"{OUT}/*"):
        if not p.endswith(".npy"):
            z.write(p, os.path.basename(p))
print(f"\nbundle: {OUT}.zip")
try:
    from IPython.display import FileLink, display
    display(FileLink(os.path.relpath(OUT + ".zip", "/kaggle/working")))
except Exception:
    pass
