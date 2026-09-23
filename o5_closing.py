"""
OBJECTIVE 5 -- closing experiments

  R1  BALANCED CONFOUNDING INDEX
      Paper 1's index uses accuracy against the majority class, so a trace
      that marks a subset of the MAJORITY class cannot raise it. Balanced
      accuracy weights both classes equally; the index then equals the
      fraction of a class the trace identifies, on any class balance.

  R2  THE TUSHABE RESOLUTION CONFOUND
      All 147 normal images are 1000 px wide. 136 of 422 sickle images are
      at native phone resolution, and no normal image is. Does detection
      accuracy depend on those 136 images?
        a  accuracy on all images vs on the 1000-px subset only
        b  can the features tell native from 1000-px sickle images apart?
        c  are native-resolution sickle images easier than 1000-px ones?

  R3  ANNOTATION SHEET
      Tushabe pairs sampled across the similarity range, numbered, with a
      CSV template. Label each SAME FIELD or DIFFERENT by eye; the scoring
      cell at the end then gives true precision and recall.

Output: /kaggle/working/o5_closing/ , zipped after each section. ~15 min.
"""

import os
import io
import glob
import zipfile
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.feature import local_binary_pattern, hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = os.environ.get("O5_ROOT", "/kaggle/input")
OUT = os.environ.get("O5_OUT", "/kaggle/working/o5_closing")
os.makedirs(OUT, exist_ok=True)
IMG_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
RNG = np.random.default_rng(0)
REPEATS = 10
LOG = open(f"{OUT}/log.txt", "a")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def checkpoint(tag):
    with zipfile.ZipFile(OUT + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in glob.glob(f"{OUT}/*"):
            z.write(p, os.path.basename(p))
    say(f"   [saved after {tag}]")


def find_dir(name):
    for depth in range(1, 8):
        for p in glob.glob(ROOT + "/*" * depth):
            if os.path.isdir(p) and os.path.basename(p).lower() == name.lower():
                return p
    return None


def load(name):
    if name == "RedTell":
        hits = glob.glob(ROOT + "/**/scd/images", recursive=True)
        if not hits:
            return None
        b = os.path.dirname(os.path.dirname(hits[0]))
        return ([(p, 1) for p in sorted(glob.glob(f"{b}/scd/images/*.tif"))] +
                [(p, 0) for p in sorted(glob.glob(f"{b}/control/images/*.tif"))])
    if name == "Tushabe":
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
    if name == "erythrocytesIDB":
        b = find_dir("erythrocytesIDB1")
        if not b:
            return None
        r = []
        for folder, lab in [("elongated", 1), ("circular", 0)]:
            r += [(p, lab) for p in sorted(glob.glob(f"{b}/**/{folder}/*", recursive=True))
                  if p.lower().endswith(IMG_EXT)]
        return r


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


def rf(seed=0):
    return RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=-1)


def oof(X, y, seed):
    return cross_val_predict(rf(seed), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                             method="predict_proba")[:, 1]


# =====================================================================
# R1  balanced confounding index
# =====================================================================
say("=" * 78)
say("R1  CONFOUNDING INDEX -- ACCURACY vs BALANCED ACCURACY")
say("=" * 78)
say("   accuracy index  = (A - majority) / (100 - majority)")
say("   balanced index  = (balanced accuracy - 0.5) / 0.5")
say("   The balanced form cannot be hidden by class imbalance.\n")


def indices(trace, y):
    t = np.asarray(trace, float).reshape(-1, 1)
    if not np.isfinite(t).all() or np.ptp(t) == 0:
        return np.nan, np.nan, np.nan
    pred = cross_val_predict(DecisionTreeClassifier(max_depth=1, random_state=0,
                                                    class_weight="balanced"),
                             t, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    pred_u = cross_val_predict(DecisionTreeClassifier(max_depth=1, random_state=0),
                               t, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    maj = max(y.mean(), 1 - y.mean())
    acc = accuracy_score(y, pred_u)
    ba = balanced_accuracy_score(y, pred)
    r = np.random.default_rng(3)
    null = []
    for _ in range(200):
        yp = r.permutation(y)
        pp = cross_val_predict(DecisionTreeClassifier(max_depth=1, random_state=0,
                                                      class_weight="balanced"),
                               t, yp, cv=StratifiedKFold(5, shuffle=True, random_state=0))
        null.append(balanced_accuracy_score(yp, pp))
    p = (np.sum(np.array(null) >= ba) + 1) / 201
    return (acc - maj) / (1 - maj), (ba - .5) / .5, p


r1 = []
for name in ("RedTell", "Tushabe", "erythrocytesIDB"):
    got = load(name)
    if not got:
        continue
    paths = [p for p, _ in got]
    y = np.array([l for _, l in got])
    sizes = [os.path.getsize(p) for p in paths]
    w, h = zip(*(Image.open(p).size for p in paths))
    traces = {"file size": sizes, "width": w, "height": h,
              "pixels": np.array(w) * np.array(h)}
    stem = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if all(s.isdigit() for s in stem):
        traces["filename ordinal"] = [int(s) for s in stem]
    say(f"   {name}  (majority class {100 * max(y.mean(), 1 - y.mean()):.1f}%)")
    for tn, tv in traces.items():
        a_idx, b_idx, p = indices(tv, y)
        r1.append({"corpus": name, "trace": tn, "accuracy_index": a_idx,
                   "balanced_index": b_idx, "p_balanced": p})
        if a_idx == a_idx:
            flag = "  <-- missed by the accuracy index" if (b_idx >= .2 and p < .01 and a_idx < .1) else ""
            say(f"      {tn:17s} accuracy index {a_idx:+.3f}   balanced index {b_idx:+.3f} "
                f"(p={p:.3f}){flag}")
        else:
            say(f"      {tn:17s} constant -- carries no information")
    say("")
pd.DataFrame(r1).to_csv(f"{OUT}/r1_balanced_index.csv", index=False)
checkpoint("R1")


# =====================================================================
# R2  Tushabe resolution confound
# =====================================================================
got = load("Tushabe")
if got:
    say("=" * 78)
    say("R2  DOES TUSHABE'S ACCURACY DEPEND ON THE NATIVE-RESOLUTION IMAGES?")
    say("=" * 78)
    paths = [p for p, _ in got]
    y = np.array([l for _, l in got])
    width = np.array([Image.open(p).size[0] for p in paths])
    native = width > 1100
    say(f"   native resolution (> 1100 px wide): sickle {int(native[y == 1].sum())}, "
        f"normal {int(native[y == 0].sum())}")
    say(f"   1000 px wide:                        sickle {int((~native & (y == 1)).sum())}, "
        f"normal {int((~native & (y == 0)).sum())}\n")
    say("   extracting features ...")
    X = np.stack([feat(p) for p in paths])

    # a -- all images vs 1000-px subset only
    a_all, a_sub = [], []
    sub = ~native
    for r in range(REPEATS):
        a_all.append(roc_auc_score(y, oof(X, y, r)))
        a_sub.append(roc_auc_score(y[sub], oof(X[sub], y[sub], r)))
    say(f"   a  AUC, all images              {np.mean(a_all):.3f}  "
        f"[{np.percentile(a_all, 2.5):.3f}, {np.percentile(a_all, 97.5):.3f}]")
    say(f"      AUC, 1000-px images only     {np.mean(a_sub):.3f}  "
        f"[{np.percentile(a_sub, 2.5):.3f}, {np.percentile(a_sub, 97.5):.3f}]")
    say(f"      difference                   {np.mean(a_all) - np.mean(a_sub):+.3f}")

    # b -- can the features see resolution at all?
    ys = native[y == 1].astype(int)
    Xs = X[y == 1]
    b_auc = np.mean([roc_auc_score(ys, oof(Xs, ys, r)) for r in range(REPEATS)])
    say(f"\n   b  native vs 1000-px, within the SICKLE class only: AUC {b_auc:.3f}")
    say("      (0.5 = the features cannot see resolution; near 1 = they can, so")
    say("       resolution is available to the classifier as a shortcut)")

    # c -- are native sickle images easier?
    p_all = np.mean([oof(X, y, r) for r in range(REPEATS)], axis=0)
    nat_idx = np.where(native & (y == 1))[0]
    std_idx = np.where(~native & (y == 1))[0]
    neg = np.where(y == 0)[0]
    auc_nat = roc_auc_score(np.r_[np.ones(len(nat_idx)), np.zeros(len(neg))],
                            np.r_[p_all[nat_idx], p_all[neg]])
    auc_std = roc_auc_score(np.r_[np.ones(len(std_idx)), np.zeros(len(neg))],
                            np.r_[p_all[std_idx], p_all[neg]])
    say(f"\n   c  AUC against all normals, native-resolution sickle   {auc_nat:.3f}")
    say(f"      AUC against all normals, 1000-px sickle             {auc_std:.3f}")
    say(f"      difference                                           {auc_nat - auc_std:+.3f}")
    pd.DataFrame([{"auc_all": np.mean(a_all), "auc_1000px_only": np.mean(a_sub),
                   "auc_native_vs_1000_within_sickle": b_auc,
                   "auc_native_sickle_vs_normal": auc_nat,
                   "auc_1000px_sickle_vs_normal": auc_std}]).to_csv(
        f"{OUT}/r2_resolution_confound.csv", index=False)
    say("""
   Reading. If (b) is high the features can see resolution; if (c) shows
   native-resolution sickle images scoring markedly easier than 1000-px ones,
   and (a) drops when they are removed, part of the reported accuracy is a
   resolution shortcut rather than morphology. If (b) is near 0.5 the
   shortcut is not available to these features -- but a CNN trained at full
   resolution may still find it, and that is worth stating for any network
   reported on this corpus.
""")
    checkpoint("R2")


# =====================================================================
# R3  annotation sheet for human ground truth
# =====================================================================
if got:
    say("=" * 78)
    say("R3  ANNOTATION SHEET -- label each pair SAME FIELD or DIFFERENT by eye")
    say("=" * 78)
    S = 128
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
            r_ = np.rot90(a, k)
            for m in (r_, np.fliplr(r_)):
                m = (m - m.mean()) / (m.std() + 1e-9) * H
                out.append(m / (np.sqrt((m ** 2).sum()) + 1e-9))
        return np.fft.rfft2(np.stack(out)).astype(np.complex64)

    say("   computing similarities at 128 px (a few minutes) ...")
    F = np.stack([spec(p) for p in paths])
    n = len(paths)
    best = np.full((n, n), -1, np.float32)
    for i in range(n):
        xc = np.fft.irfft2(F * np.conj(F[i, 0])[None, None], s=(S, S))
        best[i] = xc[:, :, mask].max(axis=(1, 2))
    np.fill_diagonal(best, -1)
    iu = np.triu_indices(n, 1)
    v = best[iu]
    bins = [(0.90, 1.01), (0.75, 0.90), (0.65, 0.75), (0.55, 0.65), (0.45, 0.55), (0.30, 0.45)]
    picks = []
    for lo, hi in bins:
        idx = np.where((v >= lo) & (v < hi))[0]
        if len(idx):
            picks += [(iu[0][k], iu[1][k], float(v[k]))
                      for k in RNG.choice(idx, min(10, len(idx)), replace=False)]
    RNG.shuffle(picks)                        # so the order does not reveal the score
    rows = []
    per_sheet = 10
    for s0 in range(0, len(picks), per_sheet):
        chunk = picks[s0:s0 + per_sheet]
        fig, ax = plt.subplots(len(chunk), 2, figsize=(7, 3.4 * len(chunk)))
        ax = np.atleast_2d(ax)
        for r_, (i, j, s) in enumerate(chunk):
            pid = s0 + r_ + 1
            for c_, k in enumerate((i, j)):
                ax[r_, c_].imshow(Image.open(paths[k]).convert("RGB"))
                ax[r_, c_].set_xticks([]); ax[r_, c_].set_yticks([])
            ax[r_, 0].set_ylabel(f"PAIR {pid}", fontsize=13, fontweight="bold")
            rows.append({"pair": pid, "file_a": os.path.basename(paths[i]),
                         "file_b": os.path.basename(paths[j]),
                         "class_a": int(y[i]), "class_b": int(y[j]),
                         "similarity": round(s, 4), "label": ""})
        fig.tight_layout()
        fig.savefig(f"{OUT}/annotate_sheet_{s0 // per_sheet + 1:02d}.png", dpi=90)
        plt.close(fig)
    A = pd.DataFrame(rows)
    A[["pair", "file_a", "file_b", "label"]].to_csv(f"{OUT}/annotation_template.csv", index=False)
    A.to_csv(f"{OUT}/annotation_key_DO_NOT_OPEN_BEFORE_LABELLING.csv", index=False)
    say(f"   wrote {len(A)} pairs across {int(np.ceil(len(A) / per_sheet))} sheets")
    say("""
   HOW TO LABEL
     1  Open annotate_sheet_01.png onward. Each row is one numbered pair.
     2  In annotation_template.csv, write SAME if both images show the same
        field -- the same cells in the same arrangement, even if shifted,
        rotated or differently exposed -- or DIFF otherwise.
     3  Do not open the key file until you have finished; the similarity
        scores are hidden so they cannot influence the labels.
     4  Upload the filled template and the key, and the scoring gives true
        precision and recall of automated repeat detection at every threshold.
""")
    checkpoint("R3")

LOG.close()
try:
    from IPython.display import FileLink, display
    display(FileLink(os.path.relpath(OUT + ".zip", "/kaggle/working")))
except Exception:
    pass
