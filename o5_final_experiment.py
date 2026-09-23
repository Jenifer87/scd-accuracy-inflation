"""
OBJECTIVE 5 -- FINAL CONSOLIDATED EXPERIMENT

One run, every result, with the improved source grouping.

  A  GROUPING       dihedral + shift-tolerant cross-correlation (+/-15%),
                    threshold CALIBRATED per corpus by planting known copies,
                    contact sheets saved for visual confirmation
  B  INFLATION      naive vs source-held AUC, 20 fold assignments,
                    two learners x two feature sets, at the calibrated
                    threshold and across a range of thresholds
  C  TUSHABE PROBE  capture-order tests T1 (lag similarity) and T2 (blocks
                    k = 2..6)
  D  VALIDATION     V1 recall by transformation, V2 estimator recovery,
                    V3 threshold calibration for the protocol

Everything is written to /kaggle/working/o5_final/ and zipped at the end.
Expect 25-40 minutes. Progress is printed throughout.

Attach: Tushabe / BioMorphNet, erythrocytesIDB, RedTell, erythroSight.
"""

import os
import re
import glob
import shutil
import zipfile
import hashlib
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
ROOT = os.environ.get("O5_ROOT", "/kaggle/input")
OUT = os.environ.get("O5_OUT", "/kaggle/working/o5_final")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
IMG_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
RNG = np.random.default_rng(0)

S = 64                       # similarity resolution
CENTRE = 0.60                # central crop, removes eyepiece vignette
MAXSHIFT = int(0.15 * S)     # shift search, +/-15%
REPEATS = 20
SWEEP = [0.65, 0.75, 0.85, 0.95]
LOG = open(f"{OUT}/log.txt", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


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
    r = [(p, 1) for p in sorted(glob.glob(f"{b}/scd/images/*.tif"))]
    r += [(p, 0) for p in sorted(glob.glob(f"{b}/control/images/*.tif"))]
    return r, "SCD vs control"


def load_tushabe():
    b = find_dir("Tushabe")
    if not b:
        return None
    r = []
    for d in sorted(glob.glob(f"{b}/*")):
        n = os.path.basename(d).lower()
        lab = 0 if "normal" in n else (1 if "sickle" in n else None)
        if lab is None:
            continue
        r += [(p, lab) for p in sorted(glob.glob(f"{d}/**/*", recursive=True))
              if p.lower().endswith(IMG_EXT)]
    return r, "sickle present vs absent"


def load_idb():
    b = find_dir("erythrocytesIDB1")
    if not b:
        return None
    r = []
    for folder, lab in [("elongated", 1), ("circular", 0)]:
        r += [(p, lab) for p in sorted(glob.glob(f"{b}/**/{folder}/*", recursive=True))
              if p.lower().endswith(IMG_EXT)]
    return r, "elongated vs circular (shape task, all donors SCD)"


CORPORA = [("RedTell", load_redtell), ("Tushabe", load_tushabe),
           ("erythrocytesIDB", load_idb)]


# =====================================================================
# A  similarity: dihedral x shift-tolerant normalised cross-correlation
# =====================================================================
HANN = np.outer(np.hanning(S), np.hanning(S)).astype(np.float32)
_mask = np.zeros((S, S), bool)
for dy in range(-MAXSHIFT, MAXSHIFT + 1):
    for dx in range(-MAXSHIFT, MAXSHIFT + 1):
        _mask[dy % S, dx % S] = True


def as_img(x):
    return x if isinstance(x, Image.Image) else Image.open(x)


def prep(x):
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
            m = (m - m.mean()) / (m.std() + 1e-9) * HANN
            out.append(m / (np.sqrt((m ** 2).sum()) + 1e-9))
    return np.stack(out).astype(np.float32)


def spectra(V):
    return np.fft.rfft2(V).astype(np.complex64)


def sim_rows(F, Fq):
    """Similarity of query stack Fq (m,8,..) against library F (n,8,..).
    Returns (m, n): best correlation over 8 transforms and shifts."""
    out = np.empty((len(Fq), len(F)), np.float32)
    for i in range(len(Fq)):
        xc = np.fft.irfft2(F * np.conj(Fq[i, 0])[None, None], s=(S, S))
        out[i] = xc[:, :, _mask].max(axis=(1, 2))
    return out


def sim_matrix(V):
    F = spectra(V)
    M = sim_rows(F, F)
    np.fill_diagonal(M, -1)
    return M, F


def groups_at(M, thr):
    n = len(M)
    parent = list(range(n))

    def f(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in zip(*np.where(np.triu(M >= thr, 1))):
        a, b = f(i), f(j)
        if a != b:
            parent[b] = a
    return np.array([f(i) for i in range(n)])


# ---- realistic copy transformations
def _jpeg(im, q):
    import io
    b = io.BytesIO()
    im.convert("RGB").save(b, "JPEG", quality=q)
    b.seek(0)
    return Image.open(b).convert("RGB")


def _crop(im, f):
    return im.crop((int(im.width * (1 - f) / 2), int(im.height * (1 - f) / 2),
                    int(im.width * (1 + f) / 2), int(im.height * (1 + f) / 2)))


TRANSFORMS = {
    "exact":            lambda im: im,
    "flip+rotate":      lambda im: im.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True),
    "JPEG q=70":        lambda im: _jpeg(im, 70),
    "JPEG q=40":        lambda im: _jpeg(im, 40),
    "resize 80%":       lambda im: im.resize((int(im.width * .8), int(im.height * .8)), Image.BILINEAR),
    "brightness +12%":  lambda im: ImageEnhance.Brightness(im).enhance(1.12),
    "noise sd 4":       lambda im: Image.fromarray(np.clip(np.asarray(im.convert("RGB"), np.float32)
                           + np.random.default_rng(3).normal(0, 4, (im.height, im.width, 3)),
                           0, 255).astype(np.uint8)),
    "crop 95%":         lambda im: _crop(im, .95),
    "crop 90%":         lambda im: _crop(im, .90),
    "shift 8%":         lambda im: im.crop((int(im.width * .08), 0, im.width, im.height)),
}
REALISTIC = ["flip+rotate", "JPEG q=70", "resize 80%", "crop 95%", "shift 8%"]


def calibrate(paths, F, n_plant=40):
    """Plant copies under realistic transformations; threshold is the 5th
    percentile of copy-to-original similarity, so 95% of such copies are
    caught. Also reports recall per transformation (V1)."""
    pick = RNG.choice(len(paths), min(n_plant, len(paths)), replace=False)
    per_t = {}
    for t, fn in TRANSFORMS.items():
        Vc = np.stack([prep(fn(as_img(paths[i]).convert("RGB"))) for i in pick])
        s = sim_rows(F, spectra(Vc))                 # copy vs library
        per_t[t] = s[np.arange(len(pick)), pick]     # similarity to its original
    thr = float(np.percentile(np.concatenate([per_t[t] for t in REALISTIC]), 5))
    return thr, per_t


def contact_sheet(paths, M, thr, name, k=8):
    """Pairs just above and just below threshold, for visual confirmation."""
    iu = np.triu_indices(len(M), 1)
    v = M[iu]
    above = np.argsort(np.abs(v - thr) + (v < thr) * 9)[:k]
    below = np.argsort(np.abs(v - thr) + (v >= thr) * 9)[:k]
    fig, ax = plt.subplots(4, k, figsize=(2 * k, 8.4))
    for col, sel in enumerate([above, below]):
        for j, idx in enumerate(sel):
            a, b = iu[0][idx], iu[1][idx]
            for r_, pi in enumerate((a, b)):
                axx = ax[col * 2 + r_, j]
                axx.imshow(as_img(paths[pi]).convert("RGB"))
                axx.set_xticks([]); axx.set_yticks([])
            ax[col * 2, j].set_title(f"{v[idx]:.3f}", fontsize=9,
                                     color="#1F6F3D" if col == 0 else "#A33A2A")
    ax[0, 0].set_ylabel("MERGED\n(above)", fontsize=10)
    ax[2, 0].set_ylabel("KEPT APART\n(below)", fontsize=10)
    fig.suptitle(f"{name}: pairs nearest the threshold {thr:.3f} -- check by eye",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/contact_{name}.png", dpi=110)
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
    """Second, independent feature set: colour statistics and a coarse
    thumbnail. Carries stain information the grey features do not."""
    im = as_img(x).convert("RGB").resize((24, 24), Image.BILINEAR)
    a = np.asarray(im, np.float32) / 255.0
    hs = np.concatenate([np.histogram(a[..., c], 16, (0, 1), density=True)[0]
                         for c in range(3)])
    g = a.mean(axis=2)
    return np.concatenate([hs, g.ravel(), a.reshape(-1, 3).mean(0), a.reshape(-1, 3).std(0)])


FEATS = {"grey": feat_gray, "colour": feat_color}


def learner(kind, seed):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=-1)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


def naive(X, y, kind, reps=REPEATS):
    v = [roc_auc_score(y, cross_val_predict(
            learner(kind, r), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=r),
            method="predict_proba")[:, 1]) for r in range(reps)]
    return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def held(X, y, g, kind, reps=REPEATS):
    if min(len(np.unique(g[y == c])) for c in (0, 1)) < 3:
        return np.nan, np.nan, np.nan
    k = int(min(5, min(len(np.unique(g[y == c])) for c in (0, 1))))
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
    if k < 2:
        return np.nan
    pred = cross_val_predict(DecisionTreeClassifier(max_depth=1, random_state=0),
                             t, y, groups=g, cv=GroupKFold(k))
    return round((accuracy_score(y, pred) * 100 - maj) / (100 - maj), 3)


# =====================================================================
# A + B  per image corpus
# =====================================================================
MAIN, SWEEP_ROWS, V1_ROWS, CACHE = [], [], [], {}
say("=" * 78)
say("A + B   GROUPING, CALIBRATION AND INFLATION")
say("=" * 78)

for name, loader in CORPORA:
    got = loader()
    say(f"\n{'-' * 78}\n{name}")
    if not got:
        say("   not found -- skipped")
        continue
    rows, task = got
    paths = [r[0] for r in rows]
    y = np.array([r[1] for r in rows])
    say(f"   task {task};  {len(paths)} images (pos {y.sum()}, neg {(1 - y).sum()})")

    say("   computing shift-tolerant similarity ...")
    V = np.stack([prep(p) for p in paths])
    M, F = sim_matrix(V)
    nn = M.max(axis=1)
    say("   nearest-neighbour similarity  " +
        "  ".join(f"p{q}={np.percentile(nn, q):.3f}" for q in (10, 50, 90, 99)))

    thr, per_t = calibrate(paths, F)
    say(f"   calibrated threshold {thr:.3f}  (catches 95% of realistic planted copies)")
    for t, s in per_t.items():
        rec = float((s >= thr).mean())
        V1_ROWS.append({"corpus": name, "transform": t, "recall": rec,
                        "median_sim": float(np.median(s))})
    say("   recall by transformation: " +
        "  ".join(f"{t} {100 * float((s >= thr).mean()):.0f}%" for t, s in per_t.items()))

    g = groups_at(M, thr)
    nsrc = len(np.unique(g))
    mixed = sum(1 for s_ in np.unique(g) if len(set(y[g == s_])) > 1)
    say(f"   sources {nsrc} behind {len(paths)} images (expansion {len(paths) / nsrc:.2f}x);"
        f"  pos {len(np.unique(g[y == 1]))}, neg {len(np.unique(g[y == 0]))};"
        f"  sources spanning both labels: {mixed}")
    contact_sheet(paths, M, thr, name)

    dup = {}
    for lab, tag in [(1, "pos"), (0, "neg")]:
        h = collections.Counter(hashlib.md5(open(p, "rb").read()).hexdigest()
                                for p, l in zip(paths, y) if l == lab)
        dup[tag] = sum(v - 1 for v in h.values() if v > 1)
    ords = np.array([float(m.group(1)) if (m := re.search(r"(\d+)$",
                     os.path.splitext(os.path.basename(p))[0])) else np.nan for p in paths])
    ci = {"size": conf_index([os.path.getsize(p) for p in paths], y, g),
          "dims": conf_index([np.prod(as_img(p).size) for p in paths], y, g),
          "ordinal": conf_index(ords, y, g) if np.isfinite(ords).mean() > .9 else np.nan}
    say(f"   exact duplicates pos {dup['pos']} neg {dup['neg']};  confounding index {ci}")

    say("   extracting features ...")
    X = {k: np.stack([f(p) for p in paths]) for k, f in FEATS.items()}
    CACHE[name] = dict(paths=paths, y=y, M=M, X=X, thr=thr)

    rec = {"corpus": name, "task": task, "images": len(paths), "threshold": round(thr, 3),
           "sources": nsrc, "expansion": round(len(paths) / nsrc, 2),
           "mixed_label_sources": mixed, "dup_pos": dup["pos"], "dup_neg": dup["neg"],
           "CI_max": np.nanmax([v for v in ci.values() if v == v]) if any(v == v for v in ci.values()) else np.nan}
    for fk in FEATS:
        for lk in ("rf", "lr"):
            n_ = naive(X[fk], y, lk)
            h_ = held(X[fk], y, g, lk)
            tag = f"{fk}_{lk}"
            rec[f"naive_{tag}"] = round(n_[0], 3)
            rec[f"held_{tag}"] = round(h_[0], 3)
            rec[f"held_{tag}_lo"] = round(h_[1], 3)
            rec[f"held_{tag}_hi"] = round(h_[2], 3)
            rec[f"inflation_{tag}"] = round(n_[0] - h_[0], 3) if h_[0] == h_[0] else np.nan
            say(f"   {fk:6s} {lk}: naive {n_[0]:.3f}  held {h_[0]:.3f} "
                f"[{h_[1]:.3f}, {h_[2]:.3f}]  inflation "
                f"{(n_[0] - h_[0]):+.3f}" if h_[0] == h_[0] else
                f"   {fk:6s} {lk}: naive {n_[0]:.3f}  held not estimable")
    rec["ceiling"] = bool(rec["naive_grey_rf"] >= .99 and rec.get("held_grey_rf", 0) >= .99)
    MAIN.append(rec)

    say("   threshold sensitivity (grey features, random forest):")
    for t_ in sorted(set(SWEEP + [round(thr, 3)])):
        gt = groups_at(M, t_)
        h_ = held(X["grey"], y, gt, "rf", reps=10)
        n_ = rec["naive_grey_rf"]
        SWEEP_ROWS.append({"corpus": name, "threshold": t_, "sources": len(np.unique(gt)),
                           "held": h_[0], "inflation": n_ - h_[0] if h_[0] == h_[0] else np.nan})
        say(f"      thr {t_:.3f}: sources {len(np.unique(gt)):4d}   held "
            f"{h_[0]:.3f}   inflation {(n_ - h_[0]):+.3f}" if h_[0] == h_[0] else
            f"      thr {t_:.3f}: sources {len(np.unique(gt)):4d}   held not estimable")

# ---- erythroSight: real participants
say(f"\n{'-' * 78}\nerythroSight")
es = sorted(glob.glob("/kaggle/**/erythrosight_tiles.csv", recursive=True))
T = pd.read_csv(es[0]) if es else None
if T is None:
    mroot = find_dir("Morphology_CSV")
    if mroot:
        say("   building tile table ...")
        USE = ["Area", "Major", "Eccentricity", "Circularity", "AR", "Solidity",
               "CurlW", "Mean", "StdDev"]
        rr = []
        for f in sorted(glob.glob(f"{mroot}/**/*.csv", recursive=True)):
            rel = os.path.relpath(f, mroot).split(os.sep)
            try:
                d = pd.read_csv(f, usecols=USE)
            except Exception:
                continue
            q = {"group": rel[0], "participant": rel[1].strip(), "n_cells": len(d)}
            q.update({f"{c}_med": float(d[c].median()) for c in USE})
            rr.append(q)
        T = pd.DataFrame(rr)
if T is not None and len(T):
    T["participant"] = T.participant.astype(str).str.strip()
    sub = T[T.group.isin(["SCD", "Normal"])]
    ys = (sub.group == "SCD").astype(int).values
    Xs = sub[[c for c in sub.columns if c.endswith("_med")] + ["n_cells"]].fillna(0).values
    gp = sub.participant.values
    rec = {"corpus": "erythroSight", "task": "SCD vs normal (tiles)", "images": len(sub),
           "sources": len(np.unique(gp)), "expansion": round(len(sub) / len(np.unique(gp)), 2)}
    for lk in ("rf", "lr"):
        n_ = naive(Xs, ys, lk)
        h_ = held(Xs, ys, gp, lk)
        rec[f"naive_grey_{lk}"] = round(n_[0], 3)
        rec[f"held_grey_{lk}"] = round(h_[0], 3)
        rec[f"held_grey_{lk}_lo"] = round(h_[1], 3)
        rec[f"held_grey_{lk}_hi"] = round(h_[2], 3)
        rec[f"inflation_grey_{lk}"] = round(n_[0] - h_[0], 3)
        say(f"   {lk}: naive {n_[0]:.3f}  participant-held {h_[0]:.3f} "
            f"[{h_[1]:.3f}, {h_[2]:.3f}]  inflation {n_[0] - h_[0]:+.3f}")
    rec["grouping"] = "real participant identifiers"
    MAIN.append(rec)

pd.DataFrame(MAIN).to_csv(f"{OUT}/main_inflation.csv", index=False)
pd.DataFrame(SWEEP_ROWS).to_csv(f"{OUT}/threshold_sensitivity.csv", index=False)
pd.DataFrame(V1_ROWS).to_csv(f"{OUT}/v1_recall_by_transform.csv", index=False)


# =====================================================================
# C  Tushabe capture-order probe
# =====================================================================
if "Tushabe" in CACHE:
    say("\n" + "=" * 78)
    say("C   TUSHABE CAPTURE-ORDER PROBE")
    say("=" * 78)
    c = CACHE["Tushabe"]
    paths, y, M, Xg = c["paths"], c["y"], c["M"], c["X"]["grey"]
    ordn = np.array([int(re.search(r"(\d+)", os.path.splitext(os.path.basename(p))[0]).group(1))
                     for p in paths])
    t1 = []
    for lab, tag in [(1, "sickle"), (0, "normal")]:
        idx = np.where(y == lab)[0]
        idx = idx[np.argsort(ordn[idx])]
        n = len(idx)
        rand = np.median([M[idx[a], idx[b]] for a, b in
                          (RNG.choice(n, 2, replace=False) for _ in range(4000))])
        say(f"  {tag}: random-pair median {rand:.3f}")
        for L in [1, 2, 3, 4, 5, 8, 12, 20]:
            s = np.array([M[idx[a], idx[a + L]] for a in range(n - L)])
            t1.append({"class": tag, "lag": L, "median": float(np.median(s)),
                       "random": float(rand), "frac_above_thr": float((s >= c["thr"]).mean())})
            say(f"     lag {L:2d}  median {np.median(s):.3f}  ({np.median(s) - rand:+.3f})"
                f"   pairs above threshold {100 * (s >= c['thr']).mean():.1f}%")
    pd.DataFrame(t1).to_csv(f"{OUT}/tushabe_T1_lag.csv", index=False)

    say("\n  T2  consecutive vs random blocks")
    order = {lab: np.where(y == lab)[0][np.argsort(ordn[y == lab])] for lab in (0, 1)}

    def blk(k, od):
        g = np.empty(len(y), dtype=object)
        for lab in (0, 1):
            for pos, i in enumerate(od[lab]):
                g[i] = f"{lab}_{pos // k}"
        return g
    t2 = []
    for k in [2, 3, 4, 5, 6]:
        cons = held(Xg, y, blk(k, order), "rf", reps=15)[0]
        rnd = np.nanmean([held(Xg, y, blk(k, {l: np.random.default_rng(900 + r).permutation(v)
                                              for l, v in order.items()}), "rf", reps=1)[0]
                          for r in range(15)])
        t2.append({"k": k, "consecutive": cons, "random": rnd, "difference": rnd - cons})
        say(f"     k={k}: consecutive {cons:.3f}   random {rnd:.3f}   difference {rnd - cons:+.3f}")
    pd.DataFrame(t2).to_csv(f"{OUT}/tushabe_T2_blocks.csv", index=False)


# =====================================================================
# D  validation
# =====================================================================
import io as _io


def perm_p(trace, y, g, n_perm=300):
    """Confounding index with a permutation null: shuffle labels, recompute,
    report how often the null reaches the observed value. Adaptive to how
    heterogeneous the corpus already is, unlike a fixed cut-off."""
    obs = conf_index(trace, y, g)
    if obs != obs:
        return obs, np.nan
    r = np.random.default_rng(11)
    null = [conf_index(trace, r.permutation(y), g) for _ in range(n_perm)]
    null = np.array([v for v in null if v == v])
    return obs, float((np.sum(null >= obs) + 1) / (len(null) + 1))


def oof_prob(X, y):
    return cross_val_predict(learner("rf", 0), X, y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=0),
                             method="predict_proba")[:, 1]


def planted_copy(p):
    im = as_img(p).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)
    im = im.rotate(int(RNG.choice([0, 90, 180, 270])), expand=True)
    return _crop(_jpeg(im, 70), .95)


say("\n" + "=" * 78)
say("D   VALIDATION")
say("=" * 78)

# ---------------------------------------------------------------- V2
if "Tushabe" in CACHE:
    c = CACHE["Tushabe"]
    paths, y, thr, Xb = c["paths"], c["y"], c["thr"], c["X"]["grey"]
    base_g = groups_at(c["M"], thr)
    Vlib = np.stack([prep(p) for p in paths])
    pr = oof_prob(Xb, y)
    pos = np.where(y == 1)[0]
    hard_order = pos[np.argsort(pr[pos])]          # lowest score = hardest positives

    say("  V2  planted duplicates in the sickle class -- HARD cases vs RANDOM cases")
    say("      hard = the positives a clean model scores lowest; random = any positives")
    v2 = []
    for mode in ("hard", "random"):
        for rate in [0.0, 0.05, 0.10, 0.20, 0.30]:
            k = int(round(rate * len(pos)))
            pick = (hard_order[:k] if mode == "hard"
                    else RNG.choice(pos, k, replace=False)) if k else np.array([], int)
            cps = [planted_copy(paths[i]) for i in pick]
            X = np.vstack([Xb] + ([np.stack([feat_gray(im) for im in cps])] if cps else []))
            yy = np.r_[y, np.ones(len(cps), int)]
            g_true = np.r_[base_g, base_g[pick]]
            if cps:
                Mall, _ = sim_matrix(np.concatenate([Vlib, np.stack([prep(im) for im in cps])]))
                g_rec = groups_at(Mall, thr)
            else:
                g_rec = base_g
            a_n = naive(X, yy, "rf", reps=10)[0]
            a_t = held(X, yy, g_true, "rf", reps=10)[0]
            a_r = held(X, yy, g_rec, "rf", reps=10)[0]
            v2.append({"mode": mode, "rate": rate, "copies": len(cps), "naive": a_n,
                       "held_true": a_t, "held_recovered": a_r,
                       "gap_true": a_n - a_t, "gap_recovered": a_n - a_r})
            say(f"     {mode:6s} {int(rate * 100):2d}%  naive {a_n:.3f}  held(true) {a_t:.3f}  "
                f"held(recovered) {a_r:.3f}   gap(true) {a_n - a_t:+.3f}  "
                f"gap(recovered) {a_n - a_r:+.3f}")
    pd.DataFrame(v2).to_csv(f"{OUT}/v2_estimator_recovery.csv", index=False)
    say("""
      Reading. Duplicating random positives in an easy corpus should change
      little -- a leaked copy of an image already classified correctly adds
      nothing. Duplicating the hard positives should inflate the naive AUC and
      open a gap that the source-held split removes. If gap(recovered) tracks
      gap(true), the grouping finds these copies and its bound is tight.
""")

# ---------------------------------------------------------------- V3
say("  V3  confounding index against a simulated second archive, with a")
say("      permutation null -- on a homogeneous corpus and a heterogeneous one")
v3 = []
for cname in ("RedTell", "Tushabe"):
    if cname not in CACHE:
        continue
    c = CACHE[cname]
    paths, y = c["paths"], c["y"]
    base_sizes = [os.path.getsize(p) for p in paths]
    base_dims = [np.prod(as_img(p).size) for p in paths]
    say(f"\n    {cname}: file sizes {len(set(base_sizes))} distinct, "
        f"dimensions {len(set(base_dims))} distinct")
    g0 = np.arange(len(paths))
    pos = np.where(y == 1)[0]
    for frac in [0.0, 0.05, 0.10, 0.25, 0.50, 1.0]:
        pick = set(RNG.choice(pos, int(round(frac * len(pos))), replace=False).tolist())
        sizes, dims = list(base_sizes), list(base_dims)
        for i in pick:
            im = as_img(paths[i]).convert("RGB")
            im = im.resize((int(im.width * .85), int(im.height * .85)), Image.BILINEAR)
            bb = _io.BytesIO()
            im.save(bb, "JPEG", quality=60)
            sizes[i], dims[i] = bb.tell(), im.width * im.height
        cs, ps = perm_p(sizes, y, g0)
        cd, pd_ = perm_p(dims, y, g0)
        best = min([p for p in (ps, pd_) if p == p], default=np.nan)
        verdict = "DETECTED" if best == best and best < 0.01 else "not detected"
        v3.append({"corpus": cname, "fraction": frac, "CI_size": cs, "p_size": ps,
                   "CI_dims": cd, "p_dims": pd_, "verdict": verdict})
        say(f"      {int(frac * 100):3d}% re-encoded   size {cs} (p={ps:.3f})   "
            f"dims {cd} (p={pd_:.3f})   -> {verdict}")
pd.DataFrame(v3).to_csv(f"{OUT}/v3_threshold_calibration.csv", index=False)
say("""
      Reading. On a corpus whose files are uniform, even a small re-encoded
      batch stands out. On one already varied -- several phones, several
      resolutions -- the same batch disappears into the spread. A fixed index
      cut-off cannot serve both, which is why the protocol flags against each
      corpus's own permutation null rather than a universal number.
""")


# =====================================================================
# summary + bundle
# =====================================================================
say("\n" + "=" * 78)
say("MAIN TABLE")
say("=" * 78)
R = pd.DataFrame(MAIN)
cols = [c for c in ["corpus", "images", "threshold", "sources", "expansion",
                    "mixed_label_sources", "dup_pos", "dup_neg", "CI_max",
                    "naive_grey_rf", "held_grey_rf", "inflation_grey_rf",
                    "inflation_grey_lr", "inflation_colour_rf", "inflation_colour_lr"]
        if c in R.columns]
say(R[cols].to_string(index=False))
say(f"\nContact sheets saved as contact_<corpus>.png -- open them and confirm that")
say("pairs marked MERGED are genuinely the same field and pairs KEPT APART are not.")
LOG.close()

zp = OUT + ".zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for p in glob.glob(f"{OUT}/*"):
        z.write(p, os.path.basename(p))
print(f"\nbundle: {zp}")
try:
    from IPython.display import FileLink, display
    display(FileLink(os.path.relpath(zp, "/kaggle/working")))
except Exception:
    pass
