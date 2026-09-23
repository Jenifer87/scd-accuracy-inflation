"""
OBJECTIVE 5 -- REMAINING EXPERIMENTS (resumable)

Sections A-C of the final run completed. This cell does what remains, with
the grouping fixed, and saves every result the moment it is produced so a
session timeout cannot lose it.

  G   VERIFIED GROUPING   candidates at 64 px, verified at 128 px where
                          genuine copies keep their cells aligned and
                          look-alikes fall apart; images linked only if each
                          is among the other's closest verified matches
                          (mutual nearest neighbours), so a chance match
                          cannot chain whole clusters together
  I   INFLATION           re-run on every image corpus with verified groups
  V2  ESTIMATOR RECOVERY  hard vs random planted duplicates
  V3  CALIBRATION         confounding index with permutation null
  E   EXTERNAL / SUBGROUP leave-one-phone-out on Tushabe
  O   OPTIMISM            bootstrap optimism correction vs source-held

Output: /kaggle/working/o5_remaining/ , zipped after every section.
Roughly 30-40 minutes. Set FAST = True to halve it.
"""

import os
import io
import re
import glob
import shutil
import zipfile
import warnings
import collections
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter
from skimage.feature import local_binary_pattern, hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                     GroupKFold, cross_val_predict)
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
FAST = False
ROOT = os.environ.get("O5_ROOT", "/kaggle/input")
OUT = os.environ.get("O5_OUT", "/kaggle/working/o5_remaining")
os.makedirs(OUT, exist_ok=True)
IMG_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
RNG = np.random.default_rng(0)
REPEATS = 6 if FAST else 12
CENTRE = 0.60
S1, S2 = 64, 128              # candidate and verification resolutions
TOPK_CAND = 10                # candidates per image from stage 1
TOPK_MUTUAL = 3               # mutual-nearest-neighbour depth
LOG = open(f"{OUT}/log.txt", "a")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def checkpoint(tag):
    zp = OUT + ".zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in glob.glob(f"{OUT}/*"):
            z.write(p, os.path.basename(p))
    say(f"   [saved after {tag} -> {zp}]")


# =====================================================================
# corpora
# =====================================================================
def find_dir(name):
    for depth in range(1, 8):
        for p in glob.glob(ROOT + "/*" * depth):
            if os.path.isdir(p) and os.path.basename(p).lower() == name.lower():
                return p
    return None


def load_redtell():
    hits = glob.glob(ROOT + "/**/scd/images", recursive=True)
    if not hits:
        return None
    b = os.path.dirname(os.path.dirname(hits[0]))
    return ([(p, 1) for p in sorted(glob.glob(f"{b}/scd/images/*.tif"))] +
            [(p, 0) for p in sorted(glob.glob(f"{b}/control/images/*.tif"))])


def load_tushabe():
    b = find_dir("Tushabe")
    if not b:
        return None
    r = []
    for d in sorted(glob.glob(f"{b}/*")):
        n = os.path.basename(d).lower()
        lab = 0 if "normal" in n else (1 if "sickle" in n else None)
        if lab is not None:
            r += [(p, lab) for p in sorted(glob.glob(f"{d}/**/*", recursive=True))
                  if p.lower().endswith(IMG_EXT)]
    return r


def load_idb():
    b = find_dir("erythrocytesIDB1")
    if not b:
        return None
    r = []
    for folder, lab in [("elongated", 1), ("circular", 0)]:
        r += [(p, lab) for p in sorted(glob.glob(f"{b}/**/{folder}/*", recursive=True))
              if p.lower().endswith(IMG_EXT)]
    return r


CORPORA = [("RedTell", load_redtell), ("Tushabe", load_tushabe),
           ("erythrocytesIDB", load_idb)]


# =====================================================================
# similarity at a given resolution
# =====================================================================
def as_img(x):
    return x if isinstance(x, Image.Image) else Image.open(x)


def _mask(S):
    ms = int(0.15 * S)
    m = np.zeros((S, S), bool)
    for dy in range(-ms, ms + 1):
        for dx in range(-ms, ms + 1):
            m[dy % S, dx % S] = True
    return m


MASK = {S1: _mask(S1), S2: _mask(S2)}
HANN = {S: np.outer(np.hanning(S), np.hanning(S)).astype(np.float32) for S in (S1, S2)}


def prep(x, S):
    im = as_img(x).convert("L")
    w, h = im.size
    im = im.crop((int(w * (1 - CENTRE) / 2), int(h * (1 - CENTRE) / 2),
                  int(w * (1 + CENTRE) / 2), int(h * (1 + CENTRE) / 2)))
    a = np.asarray(im.resize((S, S), Image.BILINEAR), np.float32)
    a = a - gaussian_filter(a, S / 8)
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        for m in (r, np.fliplr(r)):
            m = (m - m.mean()) / (m.std() + 1e-9) * HANN[S]
            out.append(m / (np.sqrt((m ** 2).sum()) + 1e-9))
    return np.fft.rfft2(np.stack(out)).astype(np.complex64)      # (8, S, S//2+1)


def spectra(items, S):
    return np.stack([prep(x, S) for x in items])


def sim_rows(Flib, Fq, S):
    out = np.empty((len(Fq), len(Flib)), np.float32)
    for i in range(len(Fq)):
        xc = np.fft.irfft2(Flib * np.conj(Fq[i, 0])[None, None], s=(S, S))
        out[i] = xc[:, :, MASK[S]].max(axis=(1, 2))
    return out


def pair_sim(Fa, Fb, S):
    xc = np.fft.irfft2(Fb * np.conj(Fa[0])[None], s=(S, S))
    return float(xc[:, MASK[S]].max())


# ---- realistic copies for calibration
def _jpeg(im, q):
    b = io.BytesIO()
    im.convert("RGB").save(b, "JPEG", quality=q)
    b.seek(0)
    return Image.open(b).convert("RGB")


def _crop(im, f):
    return im.crop((int(im.width * (1 - f) / 2), int(im.height * (1 - f) / 2),
                    int(im.width * (1 + f) / 2), int(im.height * (1 + f) / 2)))


TRANSFORMS = {
    "exact": lambda im: im,
    "flip+rotate": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True),
    "JPEG q=70": lambda im: _jpeg(im, 70),
    "resize 80%": lambda im: im.resize((int(im.width * .8), int(im.height * .8)), Image.BILINEAR),
    "brightness +12%": lambda im: ImageEnhance.Brightness(im).enhance(1.12),
    "crop 95%": lambda im: _crop(im, .95),
    "shift 8%": lambda im: im.crop((int(im.width * .08), 0, im.width, im.height)),
    "crop 90%": lambda im: _crop(im, .90),
}
REALISTIC = ["flip+rotate", "JPEG q=70", "resize 80%", "crop 95%", "shift 8%"]


# =====================================================================
# G  verified, non-chaining grouping
# =====================================================================
def verified_groups(paths, F1=None, F2=None, extra=None):
    """Return groups, thresholds and diagnostics. `extra` lets planted
    copies be appended to an existing library without recomputing it."""
    items = list(paths) + (list(extra) if extra is not None else [])
    if F1 is None:
        F1 = spectra(items, S1)
        F2 = spectra(items, S2)
    n = len(items)

    # stage 1: candidate neighbours at low resolution
    M1 = sim_rows(F1, F1, S1)
    np.fill_diagonal(M1, -1)
    cand = set()
    for i in range(n):
        for j in np.argsort(-M1[i])[:TOPK_CAND]:
            if M1[i, j] > 0.45:
                cand.add((min(i, j), max(i, j)))

    # calibration at the verification resolution
    pick = RNG.choice(len(paths), min(40, len(paths)), replace=False)
    cal = {}
    for t, fn in TRANSFORMS.items():
        cal[t] = [pair_sim(F2[i], prep(fn(as_img(items[i]).convert("RGB")), S2), S2)
                  for i in pick]
    thr = float(np.percentile(np.concatenate([cal[t] for t in REALISTIC]), 5))

    # stage 2: verify candidates
    vs = {}
    for i, j in cand:
        vs[(i, j)] = pair_sim(F2[i], F2[j], S2)

    # mutual nearest neighbours among verified pairs
    nbr = collections.defaultdict(list)
    for (i, j), s in vs.items():
        if s >= thr:
            nbr[i].append((s, j))
            nbr[j].append((s, i))
    top = {i: {j for _, j in sorted(v, reverse=True)[:TOPK_MUTUAL]} for i, v in nbr.items()}
    parent = list(range(n))

    def f(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    kept = 0
    for (i, j), s in vs.items():
        if s >= thr and j in top.get(i, ()) and i in top.get(j, ()):
            a, b = f(i), f(j)
            if a != b:
                parent[b] = a
            kept += 1
    g = np.array([f(i) for i in range(n)])
    diag = {"candidates": len(cand), "verified_above": sum(s >= thr for s in vs.values()),
            "mutual_edges": kept, "threshold_128": thr,
            "recall": {t: float(np.mean(np.array(v) >= thr)) for t, v in cal.items()}}
    return g, diag, F1, F2, vs


def contact_sheet(items, vs, thr, name, k=8):
    above = sorted([(abs(s - thr), p) for p, s in vs.items() if s >= thr])[:k]
    below = sorted([(abs(s - thr), p) for p, s in vs.items() if s < thr])[:k]
    fig, ax = plt.subplots(4, k, figsize=(2 * k, 8.4))
    for blk, sel in enumerate([above, below]):
        for c_, (_, (a, b)) in enumerate(sel):
            for r_, pi in enumerate((a, b)):
                ax[blk * 2 + r_, c_].imshow(as_img(items[pi]).convert("RGB"))
                ax[blk * 2 + r_, c_].set_xticks([]); ax[blk * 2 + r_, c_].set_yticks([])
            ax[blk * 2, c_].set_title(f"{vs[(a, b)]:.3f}", fontsize=9,
                                      color="#1F6F3D" if blk == 0 else "#A33A2A")
        for c_ in range(len(sel), k):
            for r_ in range(2):
                ax[blk * 2 + r_, c_].axis("off")
    ax[0, 0].set_ylabel("LINKED", fontsize=10)
    ax[2, 0].set_ylabel("REJECTED", fontsize=10)
    fig.suptitle(f"{name}: verified pairs nearest the 128-px threshold {thr:.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/contact_verified_{name}.png", dpi=110)
    plt.close(fig)


# =====================================================================
# features and evaluation
# =====================================================================
def feat_gray(x):
    a = np.asarray(as_img(x).convert("L").resize((96, 96), Image.BILINEAR), np.float32)
    a = (a - a.min()) / (np.ptp(a) + 1e-9)
    return np.concatenate([
        np.histogram(a, 32, (0, 1), density=True)[0],
        np.histogram(local_binary_pattern((a * 255).astype(np.uint8), 8, 1, "uniform"),
                     10, (0, 10), density=True)[0],
        hog(a, orientations=8, pixels_per_cell=(24, 24), cells_per_block=(1, 1),
            feature_vector=True),
        [a.mean(), a.std(), np.percentile(a, 10), np.percentile(a, 90)]])


def feat_color(x):
    a = np.asarray(as_img(x).convert("RGB").resize((24, 24), Image.BILINEAR), np.float32) / 255
    return np.concatenate([np.concatenate([np.histogram(a[..., c], 16, (0, 1), density=True)[0]
                                           for c in range(3)]),
                           a.mean(2).ravel(), a.reshape(-1, 3).mean(0), a.reshape(-1, 3).std(0)])


def learner(kind, seed=0):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=-1)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


def naive(X, y, kind, reps=REPEATS):
    return float(np.mean([roc_auc_score(y, cross_val_predict(
        learner(kind, r), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=r),
        method="predict_proba")[:, 1]) for r in range(reps)]))


def held(X, y, g, kind, reps=REPEATS):
    m = min(len(np.unique(g[y == c])) for c in (0, 1))
    if m < 3:
        return np.nan, np.nan, np.nan
    k = int(min(5, m))
    v = []
    for r in range(reps):
        try:
            sp = list(StratifiedGroupKFold(k, shuffle=True, random_state=r).split(X, y, g))
        except Exception:
            continue
        if any(len(np.unique(y[te])) < 2 for _, te in sp):
            continue
        pr = np.zeros(len(y))
        for tr, te in sp:
            pr[te] = learner(kind, r).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        v.append(roc_auc_score(y, pr))
    if not v:
        return np.nan, np.nan, np.nan
    return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def conf_index(trace, y, g):
    t = np.asarray(trace, float).reshape(-1, 1)
    if not np.isfinite(t).all() or np.ptp(t) == 0:
        return np.nan
    maj = max(y.mean(), 1 - y.mean()) * 100
    k = int(min(5, len(np.unique(g))))
    pred = cross_val_predict(DecisionTreeClassifier(max_depth=1, random_state=0),
                             t, y, groups=g, cv=GroupKFold(k))
    return (accuracy_score(y, pred) * 100 - maj) / (100 - maj)


# =====================================================================
# G + I
# =====================================================================
C = {}
rows = []
say("=" * 78)
say("G + I   VERIFIED GROUPING AND INFLATION")
say("=" * 78)
for name, loader in CORPORA:
    got = loader()
    say(f"\n{'-' * 78}\n{name}")
    if not got:
        say("   not found -- skipped")
        continue
    paths = [p for p, _ in got]
    y = np.array([l for _, l in got])
    say(f"   {len(paths)} images (pos {y.sum()}, neg {(1 - y).sum()}); computing spectra ...")
    g, dg, F1, F2, vs = verified_groups(paths)
    nsrc = len(np.unique(g))
    mixed = sum(1 for s in np.unique(g) if len(set(y[g == s])) > 1)
    say(f"   candidates {dg['candidates']}, verified above {dg['threshold_128']:.3f}: "
        f"{dg['verified_above']}, mutual links kept: {dg['mutual_edges']}")
    say("   recall at 128 px: " + "  ".join(f"{t} {100 * v:.0f}%" for t, v in dg["recall"].items()))
    say(f"   sources {nsrc} behind {len(paths)} (expansion {len(paths) / nsrc:.2f}x);  "
        f"pos {len(np.unique(g[y == 1]))}, neg {len(np.unique(g[y == 0]))};  mixed-label {mixed}")
    contact_sheet(paths, vs, dg["threshold_128"], name)
    say("   extracting features ...")
    X = {"grey": np.stack([feat_gray(p) for p in paths]),
         "colour": np.stack([feat_color(p) for p in paths])}
    C[name] = dict(paths=paths, y=y, g=g, F1=F1, F2=F2, X=X, thr=dg["threshold_128"])
    rec = {"corpus": name, "images": len(paths), "sources": nsrc,
           "expansion": round(len(paths) / nsrc, 2), "mixed_label": mixed,
           "links": dg["mutual_edges"], "thr128": round(dg["threshold_128"], 3)}
    for fk in ("grey", "colour"):
        for lk in ("rf", "lr"):
            n_ = naive(X[fk], y, lk)
            h_ = held(X[fk], y, g, lk)
            rec[f"infl_{fk}_{lk}"] = round(n_ - h_[0], 3) if h_[0] == h_[0] else np.nan
            rec[f"naive_{fk}_{lk}"], rec[f"held_{fk}_{lk}"] = round(n_, 3), round(h_[0], 3)
            say(f"   {fk:6s} {lk}: naive {n_:.3f}  held {h_[0]:.3f} "
                f"[{h_[1]:.3f}, {h_[2]:.3f}]  inflation {n_ - h_[0]:+.3f}"
                if h_[0] == h_[0] else f"   {fk:6s} {lk}: naive {n_:.3f}  held not estimable")
    rows.append(rec)
    pd.DataFrame(rows).to_csv(f"{OUT}/inflation_verified.csv", index=False)
    checkpoint(f"{name} grouping + inflation")


# =====================================================================
# V2  estimator recovery -- copies appended to the library, not recomputed
# =====================================================================
if "Tushabe" in C:
    say("\n" + "=" * 78)
    say("V2  ESTIMATOR RECOVERY -- hard vs random planted duplicates (Tushabe)")
    say("=" * 78)
    c = C["Tushabe"]
    paths, y, g0, Xb = c["paths"], c["y"], c["g"], c["X"]["grey"]
    pr = cross_val_predict(learner("rf"), Xb, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                           method="predict_proba")[:, 1]
    pos = np.where(y == 1)[0]
    hard = pos[np.argsort(pr[pos])]
    v2 = []
    for mode in ("hard", "random"):
        for rate in [0.0, 0.10, 0.20, 0.30]:
            k = int(round(rate * len(pos)))
            pick = (hard[:k] if mode == "hard" else RNG.choice(pos, k, replace=False)) \
                if k else np.array([], int)
            cps = []
            for i in pick:
                im = as_img(paths[i]).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)
                im = im.rotate(int(RNG.choice([0, 90, 180, 270])), expand=True)
                cps.append(_crop(_jpeg(im, 70), .95))
            X = np.vstack([Xb] + ([np.stack([feat_gray(im) for im in cps])] if cps else []))
            yy = np.r_[y, np.ones(len(cps), int)]
            g_true = np.r_[g0, g0[pick]]
            # recovered: each copy linked to its best verified match in the library
            g_rec = g_true.copy()
            if cps:
                Fc2 = spectra(cps, S2)
                hit = 0
                for n_, (i, cf) in enumerate(zip(pick, Fc2)):
                    s = pair_sim(c["F2"][i], cf, S2)
                    if s >= c["thr"]:
                        hit += 1
                    else:
                        g_rec[len(paths) + n_] = 10 ** 6 + n_     # missed: its own source
                rec_rate = hit / len(cps)
            else:
                rec_rate = np.nan
            a_n = naive(X, yy, "rf", reps=6)
            a_t = held(X, yy, g_true, "rf", reps=6)[0]
            a_r = held(X, yy, g_rec, "rf", reps=6)[0]
            v2.append({"mode": mode, "rate": rate, "copies": len(cps), "copy_recall": rec_rate,
                       "naive": a_n, "held_true": a_t, "held_recovered": a_r,
                       "gap_true": a_n - a_t, "gap_recovered": a_n - a_r})
            say(f"   {mode:6s} {int(rate * 100):2d}%  naive {a_n:.3f}  held(true) {a_t:.3f}  "
                f"held(recovered) {a_r:.3f}  gap(true) {a_n - a_t:+.3f}  "
                f"gap(rec) {a_n - a_r:+.3f}  copies found "
                f"{'-' if rec_rate != rec_rate else f'{100 * rec_rate:.0f}%'}")
            pd.DataFrame(v2).to_csv(f"{OUT}/v2_estimator_recovery.csv", index=False)
    checkpoint("V2")


# =====================================================================
# V3  permutation-calibrated confounding index
# =====================================================================
say("\n" + "=" * 78)
say("V3  CONFOUNDING INDEX vs A SIMULATED SECOND ARCHIVE, PERMUTATION NULL")
say("=" * 78)


def perm_p(trace, y, g, n=200):
    obs = conf_index(trace, y, g)
    if obs != obs:
        return obs, np.nan
    r = np.random.default_rng(5)
    null = np.array([conf_index(trace, r.permutation(y), g) for _ in range(n)])
    return obs, float((np.sum(null >= obs) + 1) / (n + 1))


v3 = []
for name in ("RedTell", "Tushabe"):
    if name not in C:
        continue
    paths, y = C[name]["paths"], C[name]["y"]
    bs = [os.path.getsize(p) for p in paths]
    bd = [np.prod(as_img(p).size) for p in paths]
    say(f"\n   {name}: {len(set(bs))} distinct file sizes, {len(set(bd))} distinct dimensions")
    pos = np.where(y == 1)[0]
    g0 = np.arange(len(paths))
    for frac in [0.0, 0.05, 0.10, 0.25, 0.50, 1.0]:
        pick = set(RNG.choice(pos, int(round(frac * len(pos))), replace=False).tolist())
        sz, dm = list(bs), list(bd)
        for i in pick:
            im = as_img(paths[i]).convert("RGB")
            im = im.resize((int(im.width * .85), int(im.height * .85)), Image.BILINEAR)
            bb = io.BytesIO()
            im.save(bb, "JPEG", quality=60)
            sz[i], dm[i] = bb.tell(), im.width * im.height
        cs, ps = perm_p(sz, y, g0)
        cd, pdm = perm_p(dm, y, g0)
        best = np.nanmin([ps, pdm])
        v3.append({"corpus": name, "fraction": frac, "CI_size": cs, "p_size": ps,
                   "CI_dims": cd, "p_dims": pdm})
        say(f"      {int(frac * 100):3d}%  size {cs:.3f} (p={ps:.3f})  dims {cd:.3f} "
            f"(p={pdm:.3f})  -> {'DETECTED' if best < .01 else 'not detected'}"
            if cs == cs else f"      {int(frac * 100):3d}%  no variation in the trace")
        pd.DataFrame(v3).to_csv(f"{OUT}/v3_calibration.csv", index=False)
checkpoint("V3")


# =====================================================================
# E  leave-one-phone-out on Tushabe (external + subgroup)
# =====================================================================
if "Tushabe" in C:
    say("\n" + "=" * 78)
    say("E   LEAVE-ONE-PHONE-OUT -- image dimensions as a proxy for the phone")
    say("=" * 78)
    c = C["Tushabe"]
    paths, y, X = c["paths"], c["y"], c["X"]["grey"]
    dims = np.array([f"{a}x{b}" for a, b in (as_img(p).size for p in paths)])
    tab = pd.crosstab(dims, y).rename(columns={0: "normal", 1: "sickle"})
    say(tab.to_string())
    groups = [d for d in tab.index if tab.loc[d].min() >= 5]
    e_rows = []
    if len(groups) >= 2:
        for d in groups:
            te = dims == d
            tr = ~te
            if len(np.unique(y[tr])) < 2:
                continue
            m = learner("rf").fit(X[tr], y[tr])
            a = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
            e_rows.append({"held_out_phone": d, "n_test": int(te.sum()), "auc": a})
            say(f"   train on others, test on {d:>11s}  (n={te.sum():3d})  AUC {a:.3f}")
        ov = naive(X, y, "rf", reps=6)
        say(f"   pooled random-split AUC for comparison: {ov:.3f}")
        pd.DataFrame(e_rows).to_csv(f"{OUT}/e_leave_one_phone_out.csv", index=False)
    else:
        say("   fewer than two resolution groups contain both classes -- the phone is not")
        say("   recoverable from image dimensions, so leave-one-phone-out cannot be run.")
        say("   Report this: the release does not record which device took each image.")
    tab.to_csv(f"{OUT}/e_dimension_by_class.csv")
    checkpoint("E")


# =====================================================================
# O  bootstrap optimism correction vs source-held estimate
# =====================================================================
say("\n" + "=" * 78)
say("O   STANDARD OPTIMISM CORRECTION vs SOURCE-HELD")
say("=" * 78)
say("   Harrell's bootstrap resamples images. If images are not independent it")
say("   should UNDER-estimate optimism. The cluster bootstrap resamples sources.")
say("   Model: 10 principal components + logistic regression, so predictors are")
say("   few relative to images -- bootstrap correction fails for p > n regardless")
say("   of dependence, and that failure must not be mistaken for this one.")
from sklearn.decomposition import PCA


def small_model():
    return make_pipeline(StandardScaler(), PCA(n_components=10, random_state=0),
                         LogisticRegression(max_iter=3000))
o_rows = []
B = 60 if FAST else 150
for name in ("RedTell", "Tushabe"):
    if name not in C:
        continue
    c = C[name]
    X, y, g = c["X"]["grey"], c["y"], c["g"]
    m = small_model().fit(X, y)
    apparent = roc_auc_score(y, m.predict_proba(X)[:, 1])
    src = np.unique(g)
    by_src = {s: np.where(g == s)[0] for s in src}
    res = {}
    for mode in ("image", "cluster"):
        opt = []
        r = np.random.default_rng(21)
        for _ in range(B):
            if mode == "image":
                idx = r.integers(0, len(y), len(y))
            else:
                idx = np.concatenate([by_src[s] for s in r.choice(src, len(src))])
            if len(np.unique(y[idx])) < 2:
                continue
            mb = small_model().fit(X[idx], y[idx])
            opt.append(roc_auc_score(y[idx], mb.predict_proba(X[idx])[:, 1]) -
                       roc_auc_score(y, mb.predict_proba(X)[:, 1]))
        res[mode] = apparent - float(np.mean(opt))
    def _cv(groups):
        vals = []
        for r in range(REPEATS):
            if groups is None:
                cv = StratifiedKFold(5, shuffle=True, random_state=r)
                pr = cross_val_predict(small_model(), X, y, cv=cv, method="predict_proba")[:, 1]
            else:
                mm = min(len(np.unique(groups[y == c_])) for c_ in (0, 1))
                if mm < 3:
                    return np.nan
                sp = list(StratifiedGroupKFold(int(min(5, mm)), shuffle=True,
                                               random_state=r).split(X, y, groups))
                if any(len(np.unique(y[te])) < 2 for _, te in sp):
                    continue
                pr = np.zeros(len(y))
                for tr, te in sp:
                    pr[te] = small_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            vals.append(roc_auc_score(y, pr))
        return float(np.mean(vals)) if vals else np.nan
    h_ = _cv(g)
    n_ = _cv(None)
    o_rows.append({"corpus": name, "apparent": apparent, "naive_cv": n_,
                   "harrell_image_bootstrap": res["image"],
                   "cluster_bootstrap": res["cluster"], "source_held_cv": h_})
    say(f"\n   {name} (10 principal components + logistic regression)")
    say(f"      apparent (train = test)          {apparent:.3f}")
    say(f"      naive cross-validation           {n_:.3f}")
    say(f"      Harrell bootstrap, images        {res['image']:.3f}")
    say(f"      bootstrap, sources resampled     {res['cluster']:.3f}")
    say(f"      source-held cross-validation     {h_:.3f}")
    pd.DataFrame(o_rows).to_csv(f"{OUT}/o_optimism.csv", index=False)
say("""
   Reading. If the image bootstrap sits close to naive CV and above the
   source-held estimate, the field's standard optimism correction misses the
   dependence O5 measures. If the cluster bootstrap moves toward source-held,
   resampling sources rather than images is the correction to recommend.
""")
checkpoint("O")

say("\n" + "=" * 78)
say("SUMMARY -- inflation with verified grouping")
say("=" * 78)
say(pd.DataFrame(rows).to_string(index=False))
LOG.close()
try:
    from IPython.display import FileLink, display
    display(FileLink(os.path.relpath(OUT + ".zip", "/kaggle/working")))
except Exception:
    pass
