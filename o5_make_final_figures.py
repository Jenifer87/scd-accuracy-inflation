"""
Objective 5 -- final figures, from the completed experiments.
600 dpi PNG + vector PDF + SVG.

Main text
  O5_F1  inflation across the four corpora
  O5_F2  two kinds of dependence
  O5_F3  human reference study: probability of a true repeat by similarity,
         and precision / recall against threshold
  O5_F4  Tushabe: inflation is stable across grouping thresholds
  O5_F5  a trace can be present and unused: resolution and annotation boxes
  O5_F6  accuracy-based vs balanced confounding index
  O5_F7  bootstrap optimism correction vs source-held estimate
  O5_F8  the acceptance protocol, revised
Supplementary
  O5_S1  confounding index against a simulated second archive
  O5_S2  grouping recall by transformation, and its failure on isolated cells
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.environ.get("FIGDIR", "./figs_o5_final")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.3, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42, "ps.fonttype": 42})
RED, BLUE, GREY, AMBER = "#C1443C", "#3E6DA3", "#8A8A8A", "#D9873C"
TEAL, PURPLE, INK, LIGHT = "#2F8F7F", "#7A5AA6", "#1F2933", "#EEF2F5"
GRID = dict(color="#DDDDDD", lw=0.6, zorder=0)


def save(fig, name):
    for e in ("png", "pdf", "svg"):
        fig.savefig(f"{OUT}/{name}.{e}")
    plt.close(fig)
    print("wrote", name)


def tag(ax, s, x=-0.14, y=1.07):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def box(ax, x, y, w, h, t, fc, fs=7.2, tc="white", bold=True):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.025",
                                facecolor=fc, edgecolor="none", zorder=3))
    ax.text(x, y, t, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", zorder=4, linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, c=INK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                                 color=c, lw=1.2, zorder=2, shrinkA=2, shrinkB=3))


# ------------------------------------------------------------------ F1
def f1():
    C = [("RedTell", "12 exact duplicate fields", 0.824, 0.730, 0.681, 0.792, RED, "+0.094"),
         ("Tushabe", "repeated fields, human-anchored", 0.905, 0.880, 0.874, 0.889, RED, "+0.025"),
         ("erythroSight", "real participant identifiers", 1.000, 0.908, 0.816, 0.997, RED, "+0.092"),
         ("erythrocytesIDB", "isolated-cell shape task", 1.000, 1.000, 0.999, 1.000, GREY, "ceiling")]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    y = np.arange(len(C))[::-1]
    for yy, (n, sub, nv, hv, lo, hi, c, lab) in zip(y, C):
        ax.plot([lo, hi], [yy, yy], color=c, lw=2.6, alpha=.45, solid_capstyle="round", zorder=2)
        ax.plot([hv, nv], [yy, yy], color="#BBBBBB", lw=1, ls=":", zorder=1)
        ax.plot(nv, yy, "o", mfc="white", mec=c, mew=1.6, ms=7, zorder=4)
        ax.plot(hv, yy, "o", color=c, ms=7, zorder=4)
        ax.text(1.012, yy, lab if lab == "ceiling" else f"inflation {lab}", va="center",
                fontsize=7.6, color=c, fontweight="bold" if lab != "ceiling" else "normal")
        ax.text(0.655, yy + 0.13, n, fontsize=7.9, fontweight="bold", color=INK)
        ax.text(0.655, yy - 0.10, sub, fontsize=6.4, color="#5A6875", va="top")
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlim(0.65, 1.12); ax.set_ylim(-0.7, len(C) - 0.3)
    ax.set_xticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_xlabel("detection AUC"); ax.grid(axis="x", **GRID)
    ax.plot([], [], "o", mfc="white", mec=INK, mew=1.4, ms=6, label="naive (random split)")
    ax.plot([], [], "o", color=INK, ms=6, label="source- or participant-held")
    ax.plot([], [], color=INK, lw=2.6, alpha=.45, label="range over repeated fold assignments")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, -0.36), ncol=3)
    ax.set_title("Holding sources out lowers every estimate that is not at ceiling", pad=8)
    save(fig, "O5_F1_inflation")


# ------------------------------------------------------------------ F2
def f2():
    fig, ax = plt.subplots(figsize=(7.2, 3.7)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(.5, .955, "Two kinds of dependence, and only one can be seen in the pixels",
            ha="center", fontsize=9.2, fontweight="bold", color=INK)
    for x, head, c, items in [
        (.25, "IMAGE-LEVEL", TEAL, ["the same field saved twice,\nor photographed again",
                                    "found by dihedral, shift-tolerant\ncorrelation, checked by eye",
                                    "RedTell +0.094   Tushabe +0.025\n(duplicates)      (repeated fields)"]),
        (.75, "PARTICIPANT-LEVEL", RED, ["different fields from\nthe same person",
                                          "different cells in view:\npixels cannot link them",
                                          "measurable only with identifiers\nerythroSight +0.092"])]:
        box(ax, x, .83, .42, .075, head, c, fs=8.4)
        for i, t in enumerate(items):
            yy = .66 - i * .175
            box(ax, x, yy, .42, .125, t, c if i == 2 else LIGHT, fs=7.1,
                tc="white" if i == 2 else INK, bold=(i == 2))
            if i:
                arrow(ax, x, yy + .105, x, yy + .068, c)
        arrow(ax, x, .79, x, .728, c)
    box(ax, .5, .075, .94, .09, "Most public sickle cell releases distribute no participant identifiers, so "
        "the second kind -- as large as the\nfirst wherever it can be measured -- cannot be checked from "
        "the data as published.", PURPLE, fs=6.8, bold=False)
    save(fig, "O5_F2_dependence_types")


# ------------------------------------------------------------------ F3
def f3():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    a = ax[0]
    bins = ["<0.45", "0.45–\n0.55", "0.55–\n0.65", "0.65–\n0.75", "0.75–\n0.90", "≥0.90"]
    p = [0.00, 0.75, 1.00, 1.00, 1.00, 1.00]
    nn = [9, 8, 8, 9, 8, 10]
    cols = [GREY if v < .5 else (AMBER if v < 1 else TEAL) for v in p]
    a.bar(range(6), p, color=cols, width=.66, zorder=2)
    for i, (v, k) in enumerate(zip(p, nn)):
        a.text(i, v + .03, f"{v:.2f}", ha="center", fontsize=7.2)
        a.text(i, -.13, f"n={k}", ha="center", fontsize=6.3, color="#5A6875")
    a.set_xticks(range(6)); a.set_xticklabels(bins, fontsize=6.8)
    a.set_ylim(-.18, 1.15); a.set_ylabel("proportion judged the same field")
    a.set_xlabel("similarity at 128 px")
    a.grid(axis="y", **GRID); a.set_axisbelow(True)
    a.set_title("Above 0.55, every pair was a true repeat", pad=8); tag(a, "a")
    b = ax[1]
    t = [0.45, 0.55, 0.60, 0.648, 0.70, 0.75, 0.85, 0.95]
    prec = [0.95, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
    rec = [1.00, 0.85, 0.83, 0.66, 0.54, 0.44, 0.34, 0.07]
    b.plot(t, prec, "-o", color=TEAL, lw=1.6, ms=4.5, label="precision")
    b.plot(t, rec, "-s", color=RED, lw=1.6, ms=4.2, label="recall")
    b.axvline(0.55, color=TEAL, lw=1, ls="--"); b.axvline(0.648, color=GREY, lw=1, ls=":")
    b.text(0.555, 0.18, "human-\nanchored\n0.55", fontsize=6.4, color=TEAL)
    b.text(0.653, 0.18, "synthetic-\ncalibrated\n0.648", fontsize=6.4, color=GREY)
    b.set_xlabel("grouping threshold"); b.set_ylabel("against consensus labels")
    b.set_ylim(0, 1.08); b.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62))
    b.grid(axis="y", **GRID); b.set_axisbelow(True)
    b.set_title("Synthetic calibration missed a third of repeats", pad=8); tag(b, "b")
    fig.tight_layout(); save(fig, "O5_F3_human_reference")


# ------------------------------------------------------------------ F4
def f4():
    t = [0.45, 0.50, 0.55, 0.60, 0.648]
    S = {"grey · forest": ([.025, .023, .025, .023, .019], RED, "o"),
         "grey · logistic": ([.051, .046, .045, .048, .046], BLUE, "s"),
         "colour · forest": ([.023, .023, .027, .018, .021], AMBER, "^"),
         "colour · logistic": ([.078, .083, .084, .074, .074], PURPLE, "D")}
    src = [260, 270, 280, 285, 297]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.35, 1]})
    a = ax[0]
    for lab, (v, c, m) in S.items():
        a.plot(t, v, marker=m, color=c, lw=1.5, ms=4.5, label=lab)
    a.axvline(0.55, color=TEAL, lw=1, ls="--")
    a.set_xlabel("grouping threshold"); a.set_ylabel("inflation (naive − field-held AUC)")
    a.set_ylim(0, 0.11); a.legend(frameon=False, fontsize=6.7, loc="upper right", ncol=2)
    a.grid(axis="y", **GRID); a.set_axisbelow(True)
    a.set_title("Tushabe inflation does not depend on the threshold", pad=8); tag(a, "a")
    b = ax[1]
    b.plot(t, src, "-o", color=TEAL, lw=1.6, ms=5)
    b.axhline(569, color=GREY, lw=.9, ls=":")
    b.text(0.452, 555, "569 images", fontsize=6.6, color=GREY, va="top")
    for x_, s_ in zip(t, src):
        b.text(x_, s_ + 8, str(s_), ha="center", fontsize=6.6)
    b.set_ylim(200, 600); b.set_xlabel("grouping threshold"); b.set_ylabel("independent fields")
    b.grid(axis="y", **GRID); b.set_axisbelow(True)
    b.set_title("About two images per field", pad=8); tag(b, "b", x=-0.2)
    fig.tight_layout(); save(fig, "O5_F4_tushabe_threshold_stability")


# ------------------------------------------------------------------ F5
def f5():
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [0.85, 1.5, 0.85]})
    a = ax[0]
    a.bar([0, 1], [0.322, 0.354], color=[AMBER, AMBER], width=.6, zorder=2)
    for i, v in enumerate([0.322, 0.354]):
        a.text(i, v + .012, f"{v:.3f}", ha="center", fontsize=7.4)
    a.set_xticks([0, 1]); a.set_xticklabels(["image\nwidth", "file\nsize"])
    a.set_ylim(0, .45); a.set_ylabel("balanced confounding index")
    a.grid(axis="y", **GRID); a.set_axisbelow(True)
    a.set_title("Present in the files", pad=8); tag(a, "a", x=-0.28)
    b = ax[1]
    labs = ["all\nimages", "1000-px\nimages only", "native\nsickle vs\nnormal", "1000-px\nsickle vs\nnormal"]
    vals = [0.905, 0.944, 0.860, 0.938]
    cols = [GREY, TEAL, AMBER, TEAL]
    b.bar(range(4), vals, color=cols, width=.62, zorder=2)
    for i, v in enumerate(vals):
        b.text(i, v + .005, f"{v:.3f}", ha="center", fontsize=7.2)
    b.set_xticks(range(4)); b.set_xticklabels(labs, fontsize=6.5)
    b.set_ylim(.8, 1.0); b.set_ylabel("detection AUC")
    b.text(1.5, .985, "features can see resolution\n(native vs 1000-px, within sickle: 0.866)",
           ha="center", va="top", fontsize=6.4, color="#5A6875", style="italic")
    b.grid(axis="y", **GRID); b.set_axisbelow(True)
    b.set_title("Visible, yet native images are harder", pad=8); tag(b, "b", x=-0.2)
    c = ax[2]
    c.bar([0, 1], [0.905, 0.963], color=[TEAL, RED], width=.6, zorder=2)
    for i, v in enumerate([0.905, 0.963]):
        c.text(i, v + .004, f"{v:.3f}", ha="center", fontsize=7.4)
    c.text(.5, .988, "+0.058", ha="center", color=RED, fontsize=8, fontweight="bold")
    c.set_xticks([0, 1]); c.set_xticklabels(["clean\ncopy", "boxed\ncopy"])
    c.set_ylim(.85, 1.0)
    c.grid(axis="y", **GRID); c.set_axisbelow(True)
    c.set_title("Drawn boxes are used", pad=8); tag(c, "c", x=-0.3)
    fig.tight_layout(); save(fig, "O5_F5_availability_vs_use")


# ------------------------------------------------------------------ F6
def f6():
    rows = [("RedTell  ordinal", 1.000, 1.000, ""),
            ("Tushabe  width", 0.000, 0.322, "resolution"),
            ("Tushabe  file size", 0.000, 0.354, "resolution"),
            ("Tushabe  ordinal", -0.048, 0.642, "per-folder numbering"),
            ("erythrocytesIDB  file size", 0.272, 0.292, "content in compression")]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    y = np.arange(len(rows))[::-1]
    h = .34
    ax.barh(y + h / 2, [r[1] for r in rows], height=h, color=GREY, label="accuracy index (Paper 1)", zorder=2)
    ax.barh(y - h / 2, [r[2] for r in rows], height=h, color=TEAL, label="balanced index", zorder=2)
    for yy, r in zip(y, rows):
        ax.text(max(r[1], r[2]) + .02, yy, r[3], va="center", fontsize=6.6, color="#5A6875", style="italic")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7.4)
    ax.axvline(0, color=INK, lw=.7)
    ax.set_xlim(-.1, 1.35); ax.set_xticks([0, .2, .4, .6, .8, 1.0]); ax.set_xlabel("confounding index")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", **GRID); ax.set_axisbelow(True)
    ax.set_title("With class imbalance, the accuracy index reads zero for traces the balanced index finds", pad=8)
    save(fig, "O5_F6_balanced_index")


# ------------------------------------------------------------------ F7
def f7():
    est = ["apparent", "Harrell\nbootstrap", "bootstrap,\nsources", "naive CV", "source-held\nCV"]
    D = {"RedTell": ([.822, .723, .713, .698, .674], RED),
         "Tushabe": ([.740, .715, .702, .707, .697], BLUE)}
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    x = np.arange(len(est))
    for k, (v, c) in D.items():
        ax.plot(x, v, "-o", color=c, lw=1.6, ms=6, label=k, zorder=3)
        for xx, vv in zip(x, v):
            ax.text(xx, vv + .007, f"{vv:.3f}", ha="center", fontsize=6.6, color=c)
    ax.axvspan(3.5, 4.5, color=TEAL, alpha=.08, zorder=0)
    ax.text(4, .83, "honest\nestimate", ha="center", fontsize=6.8, color=TEAL)
    ax.set_xticks(x); ax.set_xticklabels(est)
    ax.set_ylabel("AUC (10 principal components + logistic)")
    ax.set_ylim(.66, .85); ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", **GRID); ax.set_axisbelow(True)
    ax.set_title("Standard optimism correction stays above the source-held estimate", pad=8)
    save(fig, "O5_F7_optimism")


# ------------------------------------------------------------------ F8
def f8():
    fig, ax = plt.subplots(figsize=(7.2, 5.6)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(.5, .965, "Acceptance test before an imaging tool is deployed in a screening district",
            ha="center", fontsize=9.2, fontweight="bold", color=INK)
    ax.text(.5, .927, "every check is compared with its own corpus's null, not a universal cut-off",
            ha="center", fontsize=7.1, color="#5A6875", style="italic")
    rows = [("1  FILE TRACES", "balanced confounding index on size,\ndimensions, filename order",
             "flag if above its permutation\nnull (p < 0.01)", BLUE),
            ("2  REPEATS", "shift-tolerant correlation; threshold\nset from ~60 pairs labelled by eye",
             "group repeats before\nany evaluation", TEAL),
            ("3  SOURCE-HELD GAP", "naive vs source-held AUC,\nrepeated fold assignments",
             "report the held-out figure\nwhenever the gap is non-zero", AMBER),
            ("4  USE OF A TRACE", "stratify by any flagged trace:\ndoes accuracy depend on it?",
             "present-but-unused is reported;\nused traces are removed", PURPLE),
            ("5  THREE-GROUP", "direction across disease,\nsecond disorder, control",
             "feature assigned to\nacquisition → rejected", GREY),
            ("6  LABEL CONFLICTS", "repeated sources spanning\nboth classes",
             "inspect by eye; resolve\nbefore reporting", RED)]
    for i, (h, w, r, c) in enumerate(rows):
        yy = .835 - i * .118
        box(ax, .16, yy, .27, .085, h, c, fs=7.1)
        box(ax, .48, yy, .33, .085, w, LIGHT, fs=6.6, tc=INK, bold=False)
        box(ax, .82, yy, .32, .085, r, "#FBEFEF", fs=6.5, tc=RED)
        arrow(ax, .295, yy, .314, yy); arrow(ax, .647, yy, .66, yy)
        if i < len(rows) - 1:
            arrow(ax, .16, yy - .043, .16, yy - .075)
    box(ax, .5, .075, .92, .085, "PASS on all six → deploy, and repeat the checks periodically\n"
        "ANY UNRESOLVED FAIL → do not report a detection figure from this site", TEAL, fs=7.1)
    save(fig, "O5_F8_acceptance_protocol")


# ------------------------------------------------------------------ S1
def s1():
    fr = [0, 5, 10, 25, 50, 100]
    red = [np.nan, 0.000, 0.083, 0.250, 0.500, 1.000]
    red_det = [False, False, False, True, True, True]
    tus = [0.000, 0.000, 0.000, 0.000, 0.000, 0.068]
    tus_det = [False, False, False, False, False, True]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(fr, fr and [f / 100 for f in fr], color=GREY, lw=.8, ls=":", label="index = fraction")
    ax.plot(fr, red, "-o", color=RED, lw=1.6, ms=4.5, label="RedTell (uniform files)")
    ax.plot(fr, tus, "-s", color=BLUE, lw=1.6, ms=4.2, label="Tushabe (four phones)")
    for f, v, d in zip(fr, red, red_det):
        if d:
            ax.plot(f, v, "o", mfc="none", mec=INK, ms=9)
    for f, v, d in zip(fr, tus, tus_det):
        if d:
            ax.plot(f, v, "s", mfc="none", mec=INK, ms=9)
    ax.plot([], [], "o", mfc="none", mec=INK, ms=8, label="detected, permutation p < 0.01")
    ax.set_xlabel("share of one class re-encoded as a second archive (%)")
    ax.set_ylabel("confounding index"); ax.set_ylim(-.05, 1.08)
    ax.legend(frameon=False, fontsize=6.6, loc="upper left")
    ax.grid(axis="y", **GRID); ax.set_axisbelow(True)
    ax.set_title("A varied corpus hides an archive trace", pad=8)
    save(fig, "O5_S1_archive_calibration")


# ------------------------------------------------------------------ S2
def s2():
    tr = ["exact", "flip+\nrotate", "JPEG\nq=70", "resize\n80%", "bright-\nness", "crop\n95%", "shift\n8%", "crop\n90%"]
    R = {"RedTell": ([100, 100, 100, 100, 98, 92, 82, 10], RED),
         "Tushabe": ([100, 100, 100, 100, 100, 88, 88, 28], BLUE),
         "erythrocytesIDB": ([100, 100, 100, 95, 100, 92, 88, 75], GREY)}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.6, 1]})
    a = ax[0]
    w = .26
    for k, (name, (v, c)) in enumerate(R.items()):
        a.bar(np.arange(8) + (k - 1) * w, v, width=w, color=c, label=name, zorder=2)
    a.set_xticks(range(8)); a.set_xticklabels(tr, fontsize=6.6)
    a.set_ylabel("planted copies recovered (%)"); a.set_ylim(0, 112)
    a.legend(frameon=False, fontsize=6.6, ncol=3, loc="lower left")
    a.grid(axis="y", **GRID); a.set_axisbelow(True)
    a.set_title("Recall by transformation at 128 px", pad=8); tag(a, "a", x=-0.09)
    b = ax[1]
    b.bar([0, 1], [413, 338], color=[TEAL, GREY], width=.6, zorder=2)
    for i, v in enumerate([413, 338]):
        b.text(i, v + 6, str(v), ha="center", fontsize=7.4)
    b.set_xticks([0, 1]); b.set_xticklabels(["distinct cells\n(by construction)", "sources after\ngrouping"])
    b.set_ylim(0, 470); b.set_ylabel("count")
    b.grid(axis="y", **GRID); b.set_axisbelow(True)
    b.set_title("Isolated cells merge falsely", pad=8); tag(b, "b", x=-0.22)
    fig.tight_layout(); save(fig, "O5_S2_grouping_recall")


if __name__ == "__main__":
    for fn in (f1, f2, f3, f4, f5, f6, f7, f8, s1, s2):
        fn()
