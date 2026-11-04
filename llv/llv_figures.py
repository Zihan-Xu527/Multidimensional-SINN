#!/usr/bin/env python3
"""Stage 7 of spec section 13: the figure and table builder.

Reads ONLY saved evaluation products. It never regenerates model output, never
reclassifies a run, never reselects a checkpoint and never recomputes a summary
statistic -- every number it prints is read from a file and every input file's
checksum is verified against the checksum that file was published with.

Produces:

  main text        five-row endpoint table (LaTeX + text)
                   four-panel SHORT-LAG correlation figure, k = 0..399 on
                   G_out, PAIR-COUNT normalised at draw time, no envelope
                   (fig_main_correlation)
                   two-panel illustrative setup figure, lattice + one KMC
                   density trajectory  (fig_llv_setup)
  Supplementary    long-lag correlation figure, k = 400..600, with the
                   transferred validation-side envelope, data unchanged
                   (figS_llv_longlag_correlation)
                   five-panel sliding-window figure (21 overlapping windows)
                   joint-density figure, N_vis = 12000, representative run
                   full run-level tables (validation and final)
                   10^6-step stability table

Greyscale legibility: the two series are distinguished by line style and
width as well as colour, so identity never rests on colour alone. The two
bivariate density panels share one colour scale, since panels on different
scales cannot be compared by eye.

ALL style parameters live in STYLE below. The figures are drawn from saved
data, so restyling costs one rerun of this script and no model computation --
if sections 3 and 4 have an established palette, substitute it there.

usage:
    python3 llv_figures.py --final-dir final_llv --curves-dir curves_llv \
        --density-dir density_llv --stability-dir stability_llv \
        --class-dir class_formal --calib-dir calib_llv \
        --select-dir select_formal --out figures_llv
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

import llv_spec as S

# --------------------------------------------------------------------- style --
# Section 4's established style. Full-width figures are 6.02 in.
#
# Two encodings are in use, by figure:
#   * the main-text short-lag correlation figure follows Figure 11's two-ink
#     convention -- black solid target, red dashed SINN -- so hue there
#     carries the SOURCE and the panel titles carry the quantity;
#   * every other figure here carries the QUANTITY in hue (which family,
#     which component, which correlation function) and the SOURCE in line
#     style (Target solid, SINN dashed).
# Both survive greyscale printing, because Target and SINN always differ in
# line style whichever encoding applies.
WIDTH = 6.02

# Figure 11's two inks, reused by the main-text short-lag correlation figure:
# black solid for the target ensemble, red dashed for the SINN ensemble.
SERIES_INK = {"target": "#000000", "sinn": "#c0392b"}

# The Supplementary long-lag correlation figure keeps the per-quantity hues:
# there colour identifies the correlation function and Target and SINN are
# distinguished by line style alone.
CURVE_COLORS = {
    "ACF_A": "#1b4f72",
    "ACF_B": "#a93226",
    "CCF_AB": "#1e8449",
    "CCF_BA": "#6c3483",
}
# The five statistical families. The first four are the Section 4 dark
# variants above; PROJ's colour is the same family of Flat-UI dark steps and
# is PROVISIONAL -- it was not supplied with the other four and should be
# checked against Section 4 before publication.
FAMILY_COLORS = {
    "ACF": "#1b4f72",
    "CCF": "#a93226",
    "ACF2": "#1e8449",
    "PDF": "#6c3483",
    "PROJ": "#b9770e",
}
PROVISIONAL_COLORS = ("PROJ",)
# the two coarse-grained densities, for the marginal panels
RHO_COLORS = {"A": "#1b4f72", "B": "#a93226"}
# lattice site states in the illustrative setup figure. A and B reuse the two
# density hues, so the lattice panel and the trajectory panel agree about
# which species is which; the vacancy is unsaturated so it reads as background.
SITE_COLORS = {"O": "#f2f2f2", "A": "#1b4f72", "B": "#a93226"}
# neutral ink for legend proxies, so the legend explains line style and not hue
NEUTRAL = "#333333"

STYLE = {
    "width_full": WIDTH,
    "target_ls": "-",
    "sinn_ls": "--",
    "line_lw": 0.9,
    "band_alpha": 0.16,
    "sinn_band_alpha": 0.18,
    # the reference band stays neutral grey with a dashed edge, so it differs
    # from the family-coloured model band in edge style as well as hue
    "ref_band": {"facecolor": "#9a9a9a", "alpha": 0.22,
                 "edgecolor": "#5a5a5a", "linewidth": 0.5, "linestyle": "--"},
    # standardized display windows, Section 4/5.1 convention; the estimates
    # themselves stay on the frozen [-5, 5] grid
    "marginal_display_sigma": 4.0,
    "joint_display_sigma": 3.5,
    "unit_line": {"color": "#5a5a5a", "ls": ":", "lw": 0.7},
    # one hue, light to dark, monotone in lightness: greyscale-safe, no rainbow
    "density_cmap": "Blues",
    # Section 4 / FPU contour convention, transcribed from
    # scripts/replot_fpu_diagnostics_section4_contours.py: a four-step blue
    # ramp filling the reference density, and four contour levels placed at
    # fractions of the REFERENCE maximum, not at probability mass.
    "contour_ramp": ("#eaf1f8", "#cfe0ef", "#aecde4", "#8ab8d8"),
    "contour_levels": (0.2, 0.4, 0.6, 0.8),
    "contour_ink": "#1a1a1a",
    "contour_lw": 0.8,
    # diverging, centred at zero, symmetric limits; blue negative, red positive
    "diff_cmap": "RdBu_r",
    "zero_contour": {"colors": "#5a5a5a", "linewidths": 0.5},
    # minor ticks: the same 0.6-pt stroke as the majors, at the established
    # short length, and never labelled
    "minor_tick": {"length": 2.0, "width": 0.6},
    "panel_label_xy": (0.03, 0.97),
    "panel_label_size": 8.5,
}

# Figure 12(b) time axis: labelled majors, unlabelled minors between them.
SETUP_XTICKS_MAJOR = (100, 200, 300, 400)
SETUP_XTICKS_MINOR = (50, 150, 250, 350)

# Figure 13 lag axis. The data run to k = 399; the axis runs to 400 so the
# last major tick is drawn.
SHORTLAG_XTICKS_MAJOR = (0, 100, 200, 300, 400)
SHORTLAG_XTICKS_MINOR = (50, 150, 250, 350)
# One vertical range for all four panels. Deliberately not FPU's [-0.2, 1]:
# the LLV correlations oscillate to about -0.75, so the panels need the full
# [-1, 1]. Minors sit at the midpoints of adjacent majors.
SHORTLAG_YLIM = (-1.0, 1.0)
SHORTLAG_YTICKS_MAJOR = (-1.0, -0.5, 0.0, 0.5, 1.0)
SHORTLAG_YTICKS_MINOR = (-0.75, -0.25, 0.25, 0.75)
# C_ii(0) is exactly 1 in exact arithmetic but lands a few ulp either side of
# it, so the clipping guard needs a tolerance. A curve that really leaves the
# axis does so by far more than this.
SHORTLAG_YLIM_TOL = 1e-9

# Joint-density panel (a), the rho_B axis: labelled majors every 0.05 with one
# unlabelled minor at each midpoint. Ticks outside the display window are
# simply not drawn, so this list does not depend on the exact limits.
# figS_joint_density canvas height, inches. Full width is WIDTH as always;
# only the height is set here, and panel (a) is re-laid out from it rather
# than compressed, so the contours keep their shape.
DENSITY_HEIGHT = 2.35

DENSITY_B_YTICKS_MAJOR = (0.30, 0.35, 0.40, 0.45)
DENSITY_B_YTICKS_MINOR = (0.275, 0.325, 0.375, 0.425, 0.475)

# Family-ratio box plot. Every parameter below is transcribed from the
# standalone recipe for the accepted Section 5.1 figure
# (fig_fpu_family_ratios_tau10), so Figures 9 and this one are the same
# object with different data.
#
# Geometry: three slots per family -- SINN box, rejected column, reference
# box -- with the rejected column on the family tick, so the group is
# symmetric about it. BOXW and SLOT are chosen so the gap inside a family
# (2*SLOT - BOXW = 0.26) is smaller than the gap between families
# (1 - 2*SLOT - BOXW = 0.34); otherwise a family's reference box reads as
# belonging to the next family.
BOX_STYLE = {
    "sinn_line": "#a93226", "ref_line": "#1b4f72",
    "face_alpha": 0.30,
    "width": 0.20, "slot": 0.23, "whis": 1.5,
    "box_lw": 0.6, "median_lw": 0.9,
    "flier_ms": 3.0, "flier_mew": 0.7,
    "reject_ms": 5.0, "reject_mew": 0.7,
    "unit_line": {"color": "0.55", "lw": 0.5},
    "height": 2.9,
    "ylim_lo_factor": 0.5, "ylim_hi_factor": 4.0,
}

# Rejected runs: ONE marker, an open square with a cross set inside it, so
# the legend handle and the plotted mark cannot drift apart. Coordinates are
# in markersize units; the square spans the full size, the cross stops short
# of the corners.
# DISPLAY TIME, Section 3.3 convention: the 400-step warm-up is discarded and
# the retained output is renumbered from 1. The two analysed windows are then
#
#     G_in  (scored)       t = 1   .. 400     archive raw [400, 800)
#     G_out (continuation) t = 401 .. 800     archive raw [800, 1200)
#
# This is a LABEL, applied in the main text and the result figures. It never
# touches the warm-up length, the raw slices, the reference blocks or any
# computation, and no raw output is re-sliced to match it. The Supplementary
# passages that describe KMC physical time, and the lag axis k of the
# correlation figures, keep their original numbers.
DISPLAY_TIME = {"in": (1, 400), "out": (401, 800)}
DISPLAY_TIME_NOTE = ("Section 3.3 display convention: the warm-up is "
                     "discarded and the retained output is renumbered from "
                     "1, so the scored window is t = 1-400 and the "
                     "continuation window t = 401-800. Labels only; the "
                     "archive slices, reference blocks and all computed "
                     "quantities are unchanged.")
BOX_PANEL_TEXT = {w: "$t = %d$\u2013$%d$" % DISPLAY_TIME[w]
                  for w in ("in", "out")}


def _crossbox():
    from matplotlib.path import Path
    s, x = 0.5, 0.26
    return Path(
        [(-s, -s), (s, -s), (s, s), (-s, s), (-s, -s),
         (-x, -x), (x, x), (-x, x), (x, -x)],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY,
         Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO])

RCPARAMS = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 0.9,
    "axes.grid": False,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

PANEL_LETTERS = "abcdefgh"


def apply_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(RCPARAMS)
    return plt


def panel_label(ax, i: int, on_image: bool = False):
    """Section 4 panel label. `on_image` adds a light plate so the label stays
    legible over an image panel, where the background colour is data."""
    x, y = STYLE["panel_label_xy"]
    kw = {}
    if on_image:
        kw["bbox"] = {"facecolor": "white", "alpha": 0.78, "edgecolor": "none",
                      "boxstyle": "square,pad=0.15"}
    ax.text(x, y, "(%s)" % PANEL_LETTERS[i], transform=ax.transAxes,
            va="top", ha="left", fontsize=STYLE["panel_label_size"], **kw)


FAMILIES = S.FAMILIES
PANEL_TITLES = {"ACF_A": r"$C_{AA}(k)$", "ACF_B": r"$C_{BB}(k)$",
                "CCF_AB": r"$C_{AB}(k)$", "CCF_BA": r"$C_{BA}(k)$"}


# ------------------------------------------------------------------- loading --
def load_verified(path: str, sha_path: str = None, label: str = "") -> dict:
    """Read a JSON product and verify it against its published checksum."""
    if not os.path.exists(path):
        raise SystemExit("required product %r is missing" % path)
    sha_path = sha_path or (os.path.splitext(path)[0] + ".sha256")
    if not os.path.exists(sha_path):
        raise SystemExit(
            "%s has no published checksum at %r. The builder verifies every "
            "input it draws from and will not continue without one."
            % (label or path, sha_path))
    got = S.sha256_file(path)
    want = open(sha_path).read().strip()
    if want != got:
        raise SystemExit(
            "%s has changed since it was published (%s... now %s...). "
            "The builder reads only saved products and will not draw from "
            "a file that no longer matches its checksum."
            % (label or path, want[:12], got[:12]))
    with open(path) as fh:
        d = json.load(fh)
    return d, got


def archive_previous(out_dir: str, stem: str):
    """Move an existing figure aside before overwriting it, so the version it
    replaces stays available for comparison.

    The copies land in `<out_dir>/superseded/<UTC timestamp>/`. Nothing is
    deleted. Returns the directory, or None when there was nothing to keep.
    """
    import shutil
    existing = [p for p in
                (os.path.join(out_dir, stem + e)
                 for e in (".pdf", ".png", ".caption.txt"))
                if os.path.exists(p)]
    if not existing:
        return None
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest = os.path.join(out_dir, "superseded", stamp)
    os.makedirs(dest, exist_ok=True)
    for p in existing:
        shutil.copy2(p, os.path.join(dest, os.path.basename(p)))
    return dest


def verify_artifact(doc: dict, npz_path: str, label: str) -> str:
    """Check a binary artifact against the checksum its JSON records."""
    want = (doc.get("artifact") or {}).get("sha256")
    if not want:
        raise SystemExit("%s records no artifact checksum; refusing to draw "
                         "from an unverified binary" % label)
    if not os.path.exists(npz_path):
        raise SystemExit("%s is missing" % npz_path)
    got = S.sha256_file(npz_path)
    if got != want:
        raise SystemExit(
            "%s does not match the checksum recorded in %s (%s... now %s...)"
            % (npz_path, label, str(want)[:12], got[:12]))
    return got


def q(vals):
    v = np.asarray(vals, np.float64)
    return float(np.median(v)), float(np.quantile(v, 0.25)), \
        float(np.quantile(v, 0.75))


# ------------------------------------------------------------ main-text table --
def endpoint_table(final: dict, out_dir: str) -> dict:
    t = final["endpoint_table"]
    m = int(final["M"])
    rows = []
    for s in FAMILIES:
        e = t[s]
        rows.append({
            "family": s,
            "r_in": (e["r_eval_in"]["median"], e["r_eval_in"]["q1"],
                     e["r_eval_in"]["q3"]),
            "r_out": (e["r_eval_out"]["median"], e["r_eval_out"]["q1"],
                      e["r_eval_out"]["q3"]),
            "delta": (e["Delta"]["median"], e["Delta"]["q1"],
                      e["Delta"]["q3"]),
            "n_plus": int(e["n_plus"]), "M": int(e["M"]),
        })

    txt = os.path.join(out_dir, "table_main_endpoint.txt")
    with open(txt, "w") as fh:
        fh.write("Main text: endpoint table, conditioned on the %d "
                 "validation-successful runs of %d.\n"
                 % (m, len(final["runs"])))
        fh.write("r is a ratio of squared relative discrepancies; Delta is a "
                 "difference of squared quantities on the common scale "
                 "(L_ref^in + L_ref^out)/2. Ensemble size N = 400.\n\n")
        fh.write("%-6s %22s %22s %24s %8s\n"
                 % ("family", "r_in med [Q1,Q3]", "r_out med [Q1,Q3]",
                    "Delta med [Q1,Q3]", "n+/M"))
        for r in rows:
            fh.write("%-6s %7.2f [%5.2f,%5.2f] %7.2f [%5.2f,%5.2f] "
                     "%+8.2f [%+5.2f,%+5.2f] %5d/%d\n"
                     % (r["family"], *r["r_in"], *r["r_out"], *r["delta"],
                        r["n_plus"], r["M"]))
        w = final["run_wise_worst_family_r"]
        fh.write("\nrun-wise worst-family r (not the maxima of the medians "
                 "above): in %.2f [%.2f,%.2f], out %.2f [%.2f,%.2f]\n"
                 % (w["in"]["median"], w["in"]["q1"], w["in"]["q3"],
                    w["out"]["median"], w["out"]["q1"], w["out"]["q3"]))

    tex = os.path.join(out_dir, "table_main_endpoint.tex")
    with open(tex, "w") as fh:
        fh.write("% Main-text endpoint table. Requires booktabs.\n")
        fh.write("% r: ratio of SQUARED relative discrepancies.\n")
        fh.write("% Delta: difference of squared quantities on the common\n")
        fh.write("%   scale (L_ref^{eval,in} + L_ref^{eval,out})/2.\n")
        fh.write("\\begin{tabular}{lccc c}\n\\toprule\n")
        fh.write("Family & $r^{\\rm eval}_{\\rm in}$ & "
                 "$r^{\\rm eval}_{\\rm out}$ & $\\Delta$ & $n_+/M$ \\\\\n")
        fh.write("\\midrule\n")
        for r in rows:
            fh.write("%s & %.2f\\,[%.2f,\\,%.2f] & %.2f\\,[%.2f,\\,%.2f] & "
                     "%+.2f\\,[%+.2f,\\,%+.2f] & %d/%d \\\\\n"
                     % (r["family"].replace("_", "\\_"), *r["r_in"],
                        *r["r_out"], *r["delta"], r["n_plus"], r["M"]))
        fh.write("\\bottomrule\n\\end{tabular}\n")
    return {"txt": txt, "tex": tex, "rows": rows}


# ------------------------------------------------- main-text family-ratio box --
def family_ratio_boxplot(final: dict, out_dir: str, plt) -> dict:
    """Main-text family-ratio box plot, in place of the endpoint table.

    Presentation only. Every number drawn is read from the stored
    final-evaluation product: the per-run ratios r_s^eval, the frozen accepted
    mask, and the nine reference-block losses. Nothing is reclassified,
    recomputed or rescaled -- the ratios are already the current loss ratios
    and are neither squared again nor square-rooted.

    Per family and per segment:

      SINN box      the run-level r_s^eval of the M validation-successful runs
      reference box L_s(R_j, target_w) / L_{s,ref}^{eval,w} over the nine
                    blocks of that segment, so its median is one by
                    construction
      rejected runs one open crossed square per rejected run, at the
                    family centre between the two boxes, no jitter

    Each segment uses its own reference level. The common scale L_bar is NOT
    used here, and the validation-side reference is not borrowed: the two
    panels are separately normalised and their heights are therefore not
    comparable as an error improvement.

    Within each family the order left to right is SINN (accepted runs),
    rejected runs, reference blocks.

    No caption is written for this figure, and no title or other explanatory
    text is drawn inside it beyond the panel labels, the axes and the legend.
    """
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    B = BOX_STYLE
    runs = sorted(final["runs"], key=lambda r: int(r["run"]))
    acc = [r for r in runs if r["accepted"]]
    rej = [r for r in runs if not r["accepted"]]
    if len(acc) != int(final["M"]):
        raise SystemExit("the accepted rows (%d) disagree with the stored M "
                         "(%d)" % (len(acc), int(final["M"])))
    per_block = final["L_ref_eval_per_block"]
    l_ref = final["L_ref_eval"]

    # No title and no in-axes text: the panels carry only (a) and (b), and
    # what each segment is belongs in the manuscript caption.
    # One short in-axes label per panel, at Section 5.1's position and size.
    # Everything else about the panels -- which segment, and the
    # left/middle/right order within each family -- belongs in the
    # manuscript caption, not in the figure.
    segments = (("in", "a", BOX_PANEL_TEXT["in"]),
                ("out", "b", BOX_PANEL_TEXT["out"]))
    table_key = {"in": "r_eval_in", "out": "r_eval_out"}

    # one description per box, so a legend Patch can reuse it verbatim
    def box_style(edge, fill):
        return dict(
            widths=B["width"], patch_artist=True, whis=B["whis"],
            showfliers=True,
            boxprops=dict(facecolor=fill, edgecolor=edge, lw=B["box_lw"]),
            medianprops=dict(color=edge, lw=B["median_lw"]),
            whiskerprops=dict(color=edge, lw=B["box_lw"]),
            capprops=dict(color=edge, lw=B["box_lw"]),
            flierprops=dict(marker="o", ms=B["flier_ms"], mfc="none",
                            mec=edge, mew=B["flier_mew"], ls="none"))

    sinn_fill = to_rgba(B["sinn_line"], B["face_alpha"])
    ref_fill = to_rgba(B["ref_line"], B["face_alpha"])
    sinn_box = box_style(B["sinn_line"], sinn_fill)
    ref_box = box_style(B["ref_line"], ref_fill)
    crossbox = _crossbox()

    fig, axes = plt.subplots(1, 2, figsize=(STYLE["width_full"], B["height"]),
                             sharey=True)
    positions = np.arange(len(FAMILIES), dtype=np.float64)
    span, checks = [], []

    for ax, (w, letter, panel_text) in zip(axes, segments):
        blocks = per_block[w]
        if len(blocks) != S.N_REF_BLOCKS:
            raise SystemExit("segment %r carries %d reference blocks, "
                             "expected %d" % (w, len(blocks), S.N_REF_BLOCKS))
        for i, s in enumerate(FAMILIES):
            sinn = [float(r["r_eval_squared"][w][s]) for r in acc]
            ref = [float(blocks[j][s]) / float(l_ref[w][s])
                   for j in range(S.N_REF_BLOCKS)]
            bp = ax.boxplot([sinn], positions=[positions[i] - B["slot"]],
                            **sinn_box)
            ax.boxplot([ref], positions=[positions[i] + B["slot"]], **ref_box)

            # what matplotlib actually drew, checked against the stored table
            ys = bp["boxes"][0].get_path().vertices[:, 1]
            checks.append({
                "segment": w, "family": s,
                "drawn": {"median": float(bp["medians"][0].get_ydata()[0]),
                          "q1": float(ys.min()), "q3": float(ys.max())},
                "stored": {k: float(final["endpoint_table"][s][table_key[w]][k])
                           for k in ("median", "q1", "q3")}})

            # rejected runs: their own x at the family centre, between the
            # SINN box on the left and the reference box on the right, no
            # jitter, drawn on top
            for r in rej:
                v = float(r["r_eval_squared"][w][s])
                ax.plot([positions[i]], [v], marker=crossbox, ls="none",
                        ms=B["reject_ms"], mfc="none", mec=B["sinn_line"],
                        mew=B["reject_mew"], zorder=6)
                span.append(v)
            span.extend(sinn)
            span.extend(ref)

        ax.axhline(1.0, zorder=0, **B["unit_line"])
        ax.set_yscale("log")
        ax.set_xticks(positions)
        ax.set_xticklabels(FAMILIES)
        ax.set_xlim(-0.6, len(FAMILIES) - 0.4)
        ax.tick_params(which="both", top=False, right=False)
        panel_label(ax, 0 if letter == "a" else 1)
        # Section 5.1's placement and size, verbatim
        ax.text(0.5, 0.965, panel_text, transform=ax.transAxes, va="top",
                ha="center", fontsize=STYLE["panel_label_size"])
        if w == "in":
            ax.set_ylabel(r"$r_s^{\mathrm{eval}}$")

    bad = [c for c in checks
           if any(abs(c["drawn"][k] - c["stored"][k]) > 1e-9
                  for k in ("median", "q1", "q3"))]
    if bad:
        raise SystemExit(
            "the drawn boxes disagree with the stored endpoint table for %s; "
            "refusing to publish a figure that does not match the numbers "
            "already frozen"
            % ", ".join("%s/%s" % (c["segment"], c["family"]) for c in bad))

    lo, hi = float(min(span)), float(max(span))
    axes[0].set_ylim(lo * B["ylim_lo_factor"], hi * B["ylim_hi_factor"])

    def circle(c):
        return Line2D([], [], ls="none", marker="o", ms=B["flier_ms"],
                      mfc="none", mec=c, mew=B["flier_mew"])

    # Three legend columns of two rows. matplotlib fills a legend
    # column-major, so the empty string holds the blank cell under
    # "rejected run"; each series sits above its own outlier marker.
    handles = [
        Patch(facecolor=sinn_fill, edgecolor=B["sinn_line"], lw=B["box_lw"]),
        circle(B["sinn_line"]),
        Line2D([], [], ls="none", marker=crossbox, ms=B["reject_ms"],
               mfc="none", mec=B["sinn_line"], mew=B["reject_mew"]),
        Line2D([], [], ls="none", marker="none"),
        Patch(facecolor=ref_fill, edgecolor=B["ref_line"], lw=B["box_lw"]),
        circle(B["ref_line"]),
    ]
    # The two SINN entries name the validation outcome rather than the
    # outcome's consequence. "validation-unsuccessful" is nearly twice the
    # width of the label it replaces, so the column spacing is tightened to
    # keep the row inside the 6.02-inch canvas; the font size is NOT reduced.
    labels = ["validation-successful", "outlier (1.5 IQR)",
              "validation-unsuccessful", "", "reference blocks",
              "outlier (1.5 IQR)"]
    fig.legend(handles, labels, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), labelspacing=0.35,
               columnspacing=1.3, handletextpad=0.5)
    fig.tight_layout(pad=0.4, rect=[0, 0, 1, 0.88])

    paths = {}
    for ext in ("pdf", "png"):
        pth = os.path.join(out_dir, "fig_llv_family_ratios." + ext)
        fig.savefig(pth)
        paths[ext] = pth
    plt.close(fig)

    return {"paths": paths, "caption": None,
            "caption_written": False,
            "M": len(acc), "n_runs": len(runs),
            "rejected_runs": [int(r["run"]) for r in rej],
            "families": list(FAMILIES),
            "segments": {"in": [401, 800], "out": [801, 1200]},
            "panel_meaning": {"a": "scored segment, t = 401-800",
                              "b": "continuation segment, t = 801-1200"},
            "titles_drawn": False,
            "panel_text": dict(BOX_PANEL_TEXT),
            "panel_text_note": "drawn in-axes at (0.5, 0.965), 8.5 pt, the "
                               "Section 5.1 placement. " + DISPLAY_TIME_NOTE,
            "accepted_count_in_figure": False,
            "text_inside_figure": "panel labels, the two short panel labels "
                                  "above, axis labels and legend only",
            "reference": "L_s(R_j, target_w) / L_ref_eval[w][s] over the nine "
                         "blocks of that segment; median one by construction",
            "per_segment_denominator": True,
            "common_scale_L_bar_used": False,
            "validation_reference_borrowed": False,
            "within_family_order": "left SINN (accepted runs), centre "
                                   "rejected runs, right reference blocks",
            "rejected_markers": "at the family centre, between the two "
                                "boxes, no jitter",
            "accepted_points_overlaid": False,
            "yscale": "log", "unit_reference_line": 1.0,
            "ylim": [float(lo * B["ylim_lo_factor"]),
                     float(hi * B["ylim_hi_factor"])],
            "style_source": "standalone recipe for the accepted Section 5.1 "
                            "figure fig_fpu_family_ratios_tau10",
            "box_vs_table_check": checks}


# ------------------------------------------------ main-text short-lag corr fig --
def shortlag_figure(npz_path: str, doc: dict, out_dir: str, plt) -> dict:
    """The main-text figure: pair-count-normalised correlations, no envelope.

    Two inks, following Figure 11: black solid is the target ensemble and red
    dashed is the SINN ensemble. Hue therefore carries the SOURCE here, and
    the quantity is carried by the panel titles, not by colour.

    PAIR-COUNT NORMALISATION. The saved curves were computed with divisor
    N*T at every lag, so they carry the finite-window taper (1 - k/T). The
    displayed curves divide by the pair count instead:

        C^pc_ij(k) = T / (T - k) * C_ij(k),    k = 0 .. T-1

    This is exact arithmetic on the saved full-precision arrays, not a
    re-estimate: the numerator and the pooled means and standard deviations
    are untouched, only the divisor changes from N*T to N*(T - k). Both
    ensembles are converted identically. Nothing upstream is affected -- the
    ACF and CCF family losses, the evaluation tables, the success
    classification and the representative run all remain as stored, and the
    saved .npz is not rewritten.

    The conversion amplifies the far tail: at k = T-1 only one pair per
    trajectory contributes, so the factor is T and the sampling noise of the
    estimate is amplified with the signal. That tail is kept, not cropped.

    No band is drawn: the validation-calibrated envelope belongs to the
    Supplementary long-lag figure, which is already pair-count normalised and
    must NOT be converted again.
    """
    from matplotlib.lines import Line2D

    z = np.load(npz_path, allow_pickle=True)
    lags = np.asarray(z["lags"], np.int64)
    names = [str(x) for x in z["curve_names"]]
    c_t, c_s = SERIES_INK["target"], SERIES_INK["sinn"]

    # ---- pair-count conversion, per lag, on the full-precision arrays ----
    n_t = int(lags.shape[0])
    if not np.array_equal(lags, np.arange(n_t)):
        raise SystemExit("the saved lag axis is not 0..T-1; the pair-count "
                         "conversion factor T/(T-k) assumes it is")
    factor = n_t / (n_t - lags.astype(np.float64))      # k = 0 -> 1, k = T-1 -> T
    tgt = np.asarray(z["curve_target"], np.float64) * factor
    sinn = np.asarray(z["curve_sinn"], np.float64) * factor
    k_hi = int(lags[-1])

    fig, axes = plt.subplots(2, 2, figsize=(STYLE["width_full"], 3.6),
                             sharex=True)
    y_ticks, extremes = {}, {}
    for i, (ax, nm) in enumerate(zip(axes.ravel(), names)):
        ax.plot(lags, tgt[i], color=c_t, ls=STYLE["target_ls"],
                lw=STYLE["line_lw"])
        ax.plot(lags, sinn[i], color=c_s, ls=STYLE["sinn_ls"],
                lw=STYLE["line_lw"])
        ax.axhline(0.0, color="0.8", lw=0.5, zorder=0)
        ax.set_title(PANEL_TITLES.get(nm, nm), pad=3)
        # the axis runs to 400 so the requested major tick is drawn; the data
        # end at k = 399
        ax.set_xlim(0, n_t)
        ax.set_xticks(SHORTLAG_XTICKS_MAJOR)
        ax.set_xticks(SHORTLAG_XTICKS_MINOR, minor=True)
        # ONE fixed vertical range for all four panels, so the panels are
        # directly comparable; C_BA in particular used to be drawn on its own
        # much smaller scale. The range is not FPU's: the LLV correlations
        # oscillate well below zero.
        lo = float(min(tgt[i].min(), sinn[i].min()))
        hi = float(max(tgt[i].max(), sinn[i].max()))
        extremes[nm] = (lo, hi)
        ax.set_ylim(*SHORTLAG_YLIM)
        ax.set_yticks(SHORTLAG_YTICKS_MAJOR)
        ax.set_yticks(SHORTLAG_YTICKS_MINOR, minor=True)
        ax.tick_params(which="minor", labelbottom=False, labelleft=False,
                       **STYLE["minor_tick"])
        panel_label(ax, i)
        if i >= 2:
            ax.set_xlabel(r"lag index $k$")
        if i % 2 == 0:
            ax.set_ylabel("correlation")
        y_ticks[nm] = {"ylim": [float(v) for v in ax.get_ylim()]}
    # A fixed range must not hide data. The pair-count estimate is not a
    # bounded correlation coefficient at large lags, so an overshoot past
    # +/-1 is possible in principle; if it ever happens, stop rather than
    # publish a clipped curve.
    #
    # The tolerance is there because C_ii(0) is exactly 1 in exact
    # arithmetic -- the estimator divides by the same pooled variance it
    # sums -- but the FFT path and the variance accumulate differently, so
    # it lands at 1 +/- a few ulp. That is not a curve leaving the axis.
    tol = SHORTLAG_YLIM_TOL
    out = {nm: v for nm, v in extremes.items()
           if v[0] < SHORTLAG_YLIM[0] - tol or v[1] > SHORTLAG_YLIM[1] + tol}
    if out:
        raise SystemExit(
            "the fixed vertical range %s would clip %s; refusing to publish "
            "a figure that hides part of a curve"
            % (list(SHORTLAG_YLIM),
               ", ".join("%s (%.12g, %.12g)" % (nm, v[0], v[1])
                         for nm, v in sorted(out.items()))))

    # one shared legend, Target and SINN only, in the inks the curves use
    proxies = [Line2D([], [], color=c_t, ls=STYLE["target_ls"],
                      lw=STYLE["line_lw"]),
               Line2D([], [], color=c_s, ls=STYLE["sinn_ls"],
                      lw=STYLE["line_lw"])]
    # lower-case "target" here; "SINN" stays capitalised as an acronym
    fig.legend(proxies, ["target", "SINN"], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.045))
    fig.tight_layout()
    superseded = archive_previous(out_dir, "fig_main_correlation")
    paths = {}
    for ext in ("pdf", "png"):
        pth = os.path.join(out_dir, "fig_main_correlation." + ext)
        fig.savefig(pth)
        paths[ext] = pth
    plt.close(fig)

    cap = ("Pair-count-normalized auto- and cross-correlations for the "
           "representative LLV run, computed over the %d-step continuation "
           "segment, $t = %d$--$%d$. Black solid lines denote the KMC "
           "target, and red dashed lines denote SINN. $N = %d$."
           % (n_t, DISPLAY_TIME["out"][0], DISPLAY_TIME["out"][1],
              doc["protocol"]["ensemble_size_N"]))
    with open(os.path.join(out_dir, "fig_main_correlation.caption.txt"),
              "w") as fh:
        fh.write("DRAFT -- short factual note only; the final caption is "
                 "written separately.\n\n" + cap + "\n")
    return {"paths": paths, "caption": cap, "k_max": k_hi,
            "envelope_drawn": False,
            "ylim": list(SHORTLAG_YLIM),
            "ylim_shared_by_all_panels": True,
            "yticks_major": list(SHORTLAG_YTICKS_MAJOR),
            "yticks_minor": list(SHORTLAG_YTICKS_MINOR),
            "panel_extremes": {nm: [float(v[0]), float(v[1])]
                               for nm, v in extremes.items()},
            "clipped": False,
            "normalization": {
                "displayed": "pair-count, divisor N*(T - k)",
                "stored": "window, divisor N*T",
                "conversion": "C_pc(k) = T/(T-k) * C(k), applied per lag to "
                              "the saved full-precision arrays",
                "T": n_t,
                "max_factor": float(factor[-1]),
                "applied_to": ["target", "sinn"],
                "means_and_sd_unchanged": True,
                "saved_npz_rewritten": False,
                "upstream_unaffected": "family losses, evaluation tables, "
                                       "success classification and the "
                                       "representative run are as stored",
                "tail_cropped": False,
            },
            "colour_semantics": "two inks, as in Figure 11: black solid is "
                                "the target ensemble and red dashed the "
                                "SINN ensemble. Hue carries the source; the "
                                "quantity is carried by the panel titles",
            "series_ink": dict(SERIES_INK),
            "superseded_copy": superseded,
            "panel_ylim": {nm: rec["ylim"] for nm, rec in y_ticks.items()},
            "stored_L_out_not_plotted": {
                s: float(doc["cross_check_vs_stage_4"]["L_here"][s])
                for s in ("ACF", "CCF")},
            "stored_L_out_note": "recorded for provenance only; the family "
                                 "losses were computed from the window-"
                                 "normalised arrays, not from the pair-count "
                                 "curves drawn here"}


# -------------------------------------------- Supplementary long-lag corr fig --
def correlation_figure(curves_npz: str, curves_doc: dict, out_dir: str,
                       plt) -> dict:
    z = np.load(curves_npz, allow_pickle=True)
    lags = z["lags"]
    tgt, sinn = z["curve_target"], z["curve_sinn"]
    lo, hi = z["band_lower"], z["band_upper"]
    eps = float(z["eps_lag"])
    names = [str(x) for x in z["curve_names"]]

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    missing = [nm for nm in names if nm not in CURVE_COLORS]
    if missing:
        raise SystemExit("no colour is defined for the curve(s) %s; hue "
                         "identifies the correlation function, so the figure "
                         "cannot be drawn without one" % ", ".join(missing))

    fig, axes = plt.subplots(2, 2, figsize=(STYLE["width_full"], 3.6),
                             sharex=True)
    for i, (ax, nm) in enumerate(zip(axes.ravel(), names)):
        c = CURVE_COLORS[nm]
        # the envelope is the same hue, lightened: it belongs to this panel's
        # quantity, not to either series
        ax.fill_between(lags, lo[i], hi[i], facecolor=c,
                        alpha=STYLE["band_alpha"], linewidth=0.0)
        ax.plot(lags, tgt[i], color=c, ls=STYLE["target_ls"],
                lw=STYLE["line_lw"])
        ax.plot(lags, sinn[i], color=c, ls=STYLE["sinn_ls"],
                lw=STYLE["line_lw"])
        ax.set_title(PANEL_TITLES.get(nm, nm), pad=3)
        ax.set_xlim(S.LONGLAG_K_LO, S.LONGLAG_K_HI)
        panel_label(ax, i)
        if i >= 2:
            ax.set_xlabel(r"lag index $k$")
        if i % 2 == 0:
            ax.set_ylabel("correlation")
    # The legend is built from neutral proxy artists: it explains solid versus
    # dashed versus shaded band, and must not suggest that a hue means Target
    # or SINN.
    proxies = [
        Line2D([], [], color=NEUTRAL, ls=STYLE["target_ls"],
               lw=STYLE["line_lw"]),
        Line2D([], [], color=NEUTRAL, ls=STYLE["sinn_ls"],
               lw=STYLE["line_lw"]),
        Patch(facecolor=NEUTRAL, alpha=STYLE["band_alpha"], linewidth=0.0),
    ]
    labels = ["Target", "SINN", r"Target $\pm\,\epsilon_{\rm lag}$"]
    fig.legend(proxies, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.045))
    fig.tight_layout()
    paths = {}
    for ext in ("pdf", "png"):
        pth = os.path.join(out_dir, "figS_llv_longlag_correlation." + ext)
        fig.savefig(pth)
        paths[ext] = pth
    plt.close(fig)

    cap = ("Long-lag correlation functions of the representative run against "
           "the held-out KMC target, pair-count normalised over the combined "
           "retained interval. Band: Target $\\pm\\,\\epsilon_{\\rm lag}$, the "
           "transferred validation-side envelope, "
           "$\\epsilon_{\\rm lag} = %.6f$. The axis starts at $k = 400$, so "
           "every lag shown lies beyond the training-time unroll and none "
           "entered the loss. Colour identifies the correlation function; "
           "solid is the target and dashed the model. "
           "$M_{\\rm SINN}/\\epsilon_{\\rm lag} = %.3f$, so the SINN curve "
           "stays inside the simultaneous envelope at every lag shown. It "
           "nevertheless retains small oscillatory residuals at these lags, "
           "which lie outside the scored training range: passing the "
           "simultaneous finite-sample envelope is a statement about the "
           "whole band and is not pointwise reconstruction of the long-lag "
           "decay. $N = %d$."
           % (eps, curves_doc["M_SINN_over_eps_lag"],
              curves_doc["protocol"]["ensemble_size_N"]))
    with open(os.path.join(out_dir,
                           "figS_llv_longlag_correlation.caption.txt"),
              "w") as fh:
        fh.write("DRAFT -- short factual note only; the final caption is "
                 "written separately.\n\n" + cap + "\n")
    return {"paths": paths, "caption": cap, "eps_lag": eps}


# ------------------------------------------------- Supplementary sliding fig --
def sliding_figure(final: dict, out_dir: str, plt) -> dict:
    sl = final.get("sliding_windows")
    if not sl:
        return {"skipped": "the final-evaluation product carries no "
                           "sliding-window data"}
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    starts = [int(a) for a in sl["starts"]]
    acc = [r for r in final["runs"] if r["accepted"]]
    missing = [s for s in FAMILIES if s not in FAMILY_COLORS]
    if missing:
        raise SystemExit("no family colour is defined for %s; hue identifies "
                         "the family in this figure" % ", ".join(missing))

    fig, axes = plt.subplots(1, 5, figsize=(STYLE["width_full"], 1.8),
                             sharex=True)
    for i, (ax, s_name) in enumerate(zip(axes, FAMILIES)):
        c = FAMILY_COLORS[s_name]
        med, q1, q3, bl, bh = [], [], [], [], []
        for a in starts:
            vals = [r["sliding_r_squared"][str(a)][s_name] for r in acc]
            m, a1, a3 = q(vals)
            med.append(m); q1.append(a1); q3.append(a3)
            _, b1, b3 = q(sl["block_ratios"][str(a)][s_name])
            bl.append(b1); bh.append(b3)
        # reference blocks: neutral grey with a dashed edge, so they differ
        # from the family-coloured model band in edge style as well as hue
        ax.fill_between(starts, bl, bh, **STYLE["ref_band"])
        # the model band and line take the family's colour: hue means family
        ax.fill_between(starts, q1, q3, facecolor=c,
                        alpha=STYLE["sinn_band_alpha"], linewidth=0.0)
        ax.plot(starts, med, color=c, ls="-", lw=STYLE["line_lw"])
        ax.axhline(1.0, **STYLE["unit_line"])
        ax.set_title(s_name, pad=3)
        ax.set_xlabel(r"window start $a$")
        ax.set_xlim(min(starts), max(starts))
        panel_label(ax, i)
    axes[0].set_ylabel(r"$r_{s,a}$")
    rb = STYLE["ref_band"]
    proxies = [
        Line2D([], [], color=NEUTRAL, ls="-", lw=STYLE["line_lw"]),
        Patch(facecolor=NEUTRAL, alpha=STYLE["sinn_band_alpha"],
              linewidth=0.0),
        Patch(facecolor=rb["facecolor"], alpha=rb["alpha"],
              edgecolor=rb["edgecolor"], linewidth=rb["linewidth"],
              linestyle=rb["linestyle"]),
    ]
    labels = ["SINN, median over runs", "SINN, IQR over runs",
              "Reference blocks, IQR"]
    fig.legend(proxies, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    paths = {}
    for ext in ("pdf", "png"):
        pth = os.path.join(out_dir, "figS_sliding_window." + ext)
        fig.savefig(pth)
        paths[ext] = pth
    plt.close(fig)
    cap = ("Sliding-window family ratios over the 21 length-400 windows with "
           "zero-based starts $a = 400, 420, \\ldots, 800$. One panel per "
           "family, each in that family's colour: line, median over the %d "
           "validation-successful runs; coloured band, their interquartile "
           "range; grey dashed-edged band, interquartile range of the nine "
           "reference-block ratios, whose median is one. Adjacent windows "
           "overlap and are strongly dependent; the figure is descriptive "
           "and does not add 21 tests. $N = 400$." % len(acc))
    with open(os.path.join(out_dir, "figS_sliding_window.caption.txt"),
              "w") as fh:
        fh.write("DRAFT -- short factual note only; the final caption is "
                 "written separately.\n\n" + cap + "\n")
    return {"paths": paths, "caption": cap, "n_windows": len(starts)}


# ------------------------------------------------- Supplementary density fig --
def density_figure(npz_path: str, doc: dict, out_dir: str, plt) -> dict:
    from matplotlib.lines import Line2D

    z = np.load(npz_path)
    ga, gb = z["grid_rho_A"], z["grid_rho_B"]
    f_t, f_s = z["f2d_rho_Target"], z["f2d_rho_SINN"]
    mu = np.asarray(z["mu"], np.float64)
    sd = np.asarray(z["s"], np.float64)

    # Display windows, Section 4/5.1 convention. The estimates themselves were
    # computed on the frozen standardized grid [GRID_LO, GRID_HI] and are not
    # recomputed here; only the displayed range is restricted, and it is the
    # physical image of a standardized window, so both panels show the same
    # number of standard deviations.
    def phys(sigma, i):
        return (float(-sigma * sd[i] + mu[i]), float(sigma * sd[i] + mu[i]))

    sj = float(STYLE["joint_display_sigma"])
    sm = float(STYLE["marginal_display_sigma"])
    xlim, ylim = phys(sj, 0), phys(sj, 1)

    def inwin(grid, lim):
        g = np.asarray(grid, np.float64)
        return (g >= lim[0]) & (g <= lim[1])
    mx, my = inwin(ga, xlim), inwin(gb, ylim)
    win = np.ix_(mx, my)

    # The single reference scale for both joint panels: the maximum of the KMC
    # target density INSIDE the displayed window. The contour levels and the
    # normalisation of the signed difference are both fractions of this one
    # number, so (a) and (b) are on one scale and no level is computed from the
    # SINN density.
    f_kmc_max = float(f_t[win].max())
    levels = [f * f_kmc_max for f in STYLE["contour_levels"]]
    # the filled band above the top contour is closed at the target's own
    # maximum, as in the FPU figure
    fill_levels = levels + [float(max(f_t[win].max(), f_t.max()))]

    # Explicit axes rectangles, not a gridspec. Panel (a) must keep equal
    # aspect, so its box width is DERIVED from the axes height and the data
    # aspect of the display window; panel (b) then takes the rest of the
    # width. Shortening the figure therefore re-lays it out instead of
    # squashing it, and the contours stay the shape they are.
    height = DENSITY_HEIGHT
    bot, top = 0.185, 0.975
    h = top - bot
    aspect = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    w_a = (h * height / aspect) / STYLE["width_full"]
    left_a, gap, right_edge = 0.082, 0.095, 0.995
    fig = plt.figure(figsize=(STYLE["width_full"], height))
    left_b = left_a + w_a + gap
    X, Y = np.meshgrid(np.asarray(ga, np.float64),
                       np.asarray(gb, np.float64), indexing="ij")

    # (a) the two densities at the SAME absolute levels, target filled
    ax = fig.add_axes([left_a, bot, w_a, h])
    # Panel (a) uses the two inks of Figure 11 and Figure 13: hue carries the
    # SOURCE here, black solid target against red dashed SINN. The fill ramp
    # and the four absolute levels are unchanged.
    ax.contourf(X, Y, f_t, levels=fill_levels, colors=STYLE["contour_ramp"])
    ax.contour(X, Y, f_t, levels=levels, colors=SERIES_INK["target"],
               linewidths=STYLE["contour_lw"])
    ax.contour(X, Y, f_s, levels=levels, colors=SERIES_INK["sinn"],
               linewidths=STYLE["contour_lw"], linestyles="dashed")
    ax.set_xlabel(r"$\rho_A$")
    ax.set_ylabel(r"$\rho_B$")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # rho_B at 0.05 steps, with one unlabelled minor at each midpoint; the
    # default locator put a labelled tick every 0.025, which is too dense for
    # this panel's height
    ax.set_yticks(DENSITY_B_YTICKS_MAJOR)
    ax.set_yticks(DENSITY_B_YTICKS_MINOR, minor=True)
    ax.tick_params(axis="y", which="minor", labelleft=False,
                   **STYLE["minor_tick"])
    ax.set_aspect("equal", adjustable="box")
    ax.legend([Line2D([], [], color=SERIES_INK["target"], ls="-",
                      lw=STYLE["contour_lw"]),
               Line2D([], [], color=SERIES_INK["sinn"], ls="--",
                      lw=STYLE["contour_lw"])],
              ["target", "SINN"], loc="upper right", handlelength=2.4)
    panel_label(ax, 0)

    # (b) both marginals on ONE common physical-density axis: no twin axis, no
    # broken axis and no per-component rescaling, so the two curves are
    # directly comparable in both position and height. Hue identifies the
    # component, line style the source.
    ax = fig.add_axes([left_b, bot, right_edge - left_b, h])
    marg_lims = {c: phys(sm, i) for c, i in (("A", 0), ("B", 1))}
    union = (min(l[0] for l in marg_lims.values()),
             max(l[1] for l in marg_lims.values()))
    top = 0.0
    for comp, grid, i in (("A", ga, 0), ("B", gb, 1)):
        c = RHO_COLORS[comp]
        lim = marg_lims[comp]
        mt = np.asarray(z["marg_rho_%s_Target" % comp], np.float64)
        ms = np.asarray(z["marg_rho_%s_SINN" % comp], np.float64)
        # each curve is drawn over its own frozen display window; the AXIS is
        # the union of the two, which is what makes them comparable
        k = inwin(grid, lim)
        g = np.asarray(grid, np.float64)[k]
        ax.plot(g, mt[k], color=c, ls=STYLE["target_ls"], lw=STYLE["line_lw"])
        ax.plot(g, ms[k], color=c, ls=STYLE["sinn_ls"], lw=STYLE["line_lw"])
        top = max(top, float(mt[k].max()), float(ms[k].max()))
    ax.set_xlabel(r"population density $\rho$")
    ax.set_ylabel(r"$f_\rho$")
    ax.set_xlim(*union)
    # headroom for the single two-row legend, which sits above both peaks
    ax.set_ylim(0.0, 1.34 * top)
    # ONE legend, two rows by two columns. matplotlib fills a multi-column
    # legend COLUMN BY COLUMN, so the handle order below is
    #   column 0 (solid, target): rho_A, rho_B
    #   column 1 (dashed, SINN) : rho_A, rho_B
    # which renders as
    #   rho_A (target) | rho_A (SINN)
    #   rho_B (target) | rho_B (SINN)
    handles, labels = [], []
    for ls, src in ((STYLE["target_ls"], "target"), (STYLE["sinn_ls"], "SINN")):
        for comp in ("A", "B"):
            handles.append(Line2D([], [], color=RHO_COLORS[comp], ls=ls,
                                  lw=STYLE["line_lw"]))
            labels.append(r"$\rho_%s$ (%s)" % (comp, src))
    ax.legend(handles, labels, ncol=2, loc="upper right",
              bbox_to_anchor=(1.0, 1.0), handlelength=1.7,
              columnspacing=1.0, labelspacing=0.35, handletextpad=0.45,
              borderaxespad=0.0)
    panel_label(ax, 1)

    paths = {}
    for ext in ("pdf", "png"):
        pth = os.path.join(out_dir, "figS_joint_density." + ext)
        fig.savefig(pth)
        paths[ext] = pth
    plt.close(fig)
    cap = ("Joint and marginal densities of the coarse-grained densities in "
           "the continuation window $G_{\\rm out}$, $t = %d$--$%d$, for the "
           "representative run and the held-out KMC target."
           % DISPLAY_TIME["out"] + "  Frozen bandwidths $h_2$ for the joint "
           "densities and $h$ for the marginals, computed in standardised "
           "coordinates and transformed to physical coordinates with the "
           "Jacobian $f_\\rho = f_x/(s_A s_B)$ and $1/s_i$. (a) Filled blue "
           "ramp, the KMC target density; black solid lines, target "
           "contours; red dashed lines, SINN contours at the same absolute "
           "levels "
           "$0.2, 0.4, 0.6, 0.8$ times $\\max f_{\\rm KMC}$ over the "
           "displayed window, so no level is derived from the SINN density; "
           "equal aspect on the physical $(\\rho_A,\\rho_B)$ axes. (b) The two "
           "marginal densities on one common physical-density axis and one "
           "common vertical scale, without twin or broken axes and without "
           "per-component rescaling: colour identifies the component, solid "
           "the target and dashed the SINN. The densities were estimated on "
           "the full frozen standardised grid; the figure shows the frozen "
           "display windows, the physical image of standardised "
           "$[-%.1f,%.1f]^2$ in (a) and of standardised $[-%.1f,%.1f]$ per "
           "component in (b), whose axis is the union of the two. "
           "$N_{\\rm vis} = %d$; this ensemble enters no numerical summary "
           "and no decision."
           % (sj, sj, sm, sm, doc["ensemble"]["N_vis"]))
    with open(os.path.join(out_dir, "figS_joint_density.caption.txt"),
              "w") as fh:
        fh.write("DRAFT -- short factual note only; the final caption is "
                 "written separately.\n\n" + cap + "\n")
    return {"paths": paths, "caption": cap,
            "f_kmc_max_in_window": f_kmc_max,
            "contour_level_fractions": list(STYLE["contour_levels"]),
            "contour_levels_absolute": levels,
            "contour_levels_from": "the KMC target maximum inside the "
                                   "displayed window; no SINN level is "
                                   "computed",
            "layout": "1x2 full width; the signed-difference panel was "
                      "removed",
            "joint_xlim": list(xlim), "joint_ylim": list(ylim),
            "joint_display_sigma": sj, "marginal_display_sigma": sm,
            "marginal_xlim": {c: list(l) for c, l in marg_lims.items()},
            "marginal_axis": {
                "common_axis": list(union),
                "from": "the union of the two frozen per-component display "
                        "windows",
                "shared_vertical_scale": True,
                "twin_axis": False, "broken_axis": False,
                "per_component_rescaling": False},
            "estimation_grid_standardized": [S.GRID_LO, S.GRID_HI],
            "equal_aspect": True}


# ------------------------------------------------------ main-text setup figure --
def setup_figure(npz_path: str, doc: dict, out_dir: str, plt) -> dict:
    """Illustrative two-panel introduction to the model system.

    No SINN quantity appears here and none may be added: the two panels are
    one KMC lattice configuration and one fixed KMC density trajectory, and
    they come from different runs. Nothing in this figure is a pathwise
    comparison with anything.
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    z = np.load(npz_path)
    rho = np.asarray(z["rho_trajectory"], np.float64)
    t = np.asarray(z["t_physical"], np.int64)
    has_lat = "lattice" in getattr(z, "files", [])
    if not has_lat:
        return {"skipped": "the setup export carries no lattice frame; "
                           "rerun llv_setup_data.py with --frames"}
    lat = np.asarray(z["lattice"])

    cmap = ListedColormap([SITE_COLORS["O"], SITE_COLORS["A"],
                           SITE_COLORS["B"]])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    # Explicit axes rectangles rather than a gridspec: the lattice panel must
    # be square, and under tight_layout a square panel and a free-aspect one
    # do not come out the same height.
    height = 2.6
    bot, top = 0.195, 0.885
    fig = plt.figure(figsize=(STYLE["width_full"], height))
    h = top - bot
    w_sq = h * height / STYLE["width_full"]          # square, in figure units
    left_a, gap, right = 0.035, 0.085, 0.985
    ax_a = fig.add_axes([left_a, bot, w_sq, h])
    left_b = left_a + w_sq + gap
    ax_b = fig.add_axes([left_b, bot, right - left_b, h])
    axes = (ax_a, ax_b)

    # (a) one lattice configuration
    ax = axes[0]
    ax.imshow(lat, cmap=cmap, norm=norm, interpolation="nearest",
              origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    side = int(lat.shape[0])
    ax.set_title(r"KMC lattice, $%d \times %d$" % (side, side), pad=3)
    ax.legend([Patch(facecolor=SITE_COLORS[k], edgecolor="0.5",
                     linewidth=0.4) for k in ("O", "A", "B")],
              [r"$O$", r"$A$", r"$B$"], loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, -0.02), handlelength=1.0,
              handletextpad=0.45, columnspacing=1.2, borderaxespad=0.0)
    panel_label(ax, 0, on_image=True)

    # (b) one fixed KMC density trajectory
    ax = axes[1]
    # The trajectory is the scored window, so it is displayed on the same
    # renumbered axis as panel (a) of the family-ratio figure: t = 1..400.
    # Only the labels move; the samples are the archive slice as exported.
    t = np.arange(DISPLAY_TIME["in"][0],
                  DISPLAY_TIME["in"][0] + t.shape[0], dtype=np.int64)
    for comp, col in (("A", 0), ("B", 1)):
        ax.plot(t, rho[:, col], color=RHO_COLORS[comp], ls="-",
                lw=STYLE["line_lw"])
    ax.set_xlim(int(t[0]), int(t[-1]))
    # labelled majors on the hundreds, unlabelled minors on the fifties; the
    # minors use the short-tick style and carry no labels
    ax.set_xticks(SETUP_XTICKS_MAJOR)
    ax.set_xticks(SETUP_XTICKS_MINOR, minor=True)
    ax.tick_params(axis="x", which="minor", labelbottom=False,
                   **STYLE["minor_tick"])
    lo = float(rho.min())
    hi = float(rho.max())
    span = hi - lo
    # headroom for the legend and the panel label, which would otherwise sit
    # on the upper trace
    ax.set_ylim(lo - 0.08 * span, hi + 0.30 * span)
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel("population density")
    ax.set_title("One KMC trajectory", pad=3)
    ax.legend([Line2D([], [], color=RHO_COLORS["A"], ls="-",
                      lw=STYLE["line_lw"]),
               Line2D([], [], color=RHO_COLORS["B"], ls="-",
                      lw=STYLE["line_lw"])],
              [r"$\rho_A$", r"$\rho_B$"], loc="upper right", ncol=2,
              handlelength=1.6, columnspacing=1.2)
    panel_label(ax, 1)

    paths = {}
    for ext in ("pdf", "png"):
        pth = os.path.join(out_dir, "fig_llv_setup." + ext)
        fig.savefig(pth)
        paths[ext] = pth
    plt.close(fig)

    lat_doc = doc["lattice"]
    tr = doc["trajectory"]
    cap = ("The lattice Lotka--Volterra system. (a) One representative "
           "post-transient KMC configuration of the $%d \\times %d$ lattice: "
           "vacant sites $O$, prey $A$ and predator $B$. (b) One fixed KMC "
           "trajectory of the coarse-grained population densities "
           "$\\rho_A(t)$ and $\\rho_B(t)$ over the retained window "
           "$t = %d, \\ldots, %d$, trajectory %d of the fitting segment, in "
           "physical density coordinates. The two panels come from different "
           "KMC runs and are illustrations of the same stationary regime, "
           "not two views of one realisation. The figure is illustrative: no "
           "SINN output appears in it and no pathwise comparison with the "
           "model is implied."
           % (lat_doc["side"], lat_doc["side"], DISPLAY_TIME["in"][0],
              DISPLAY_TIME["in"][1], tr["index"]))
    with open(os.path.join(out_dir, "fig_llv_setup.caption.txt"), "w") as fh:
        fh.write("DRAFT -- short factual note only; the final caption is "
                 "written separately.\n\n" + cap + "\n")
    return {"paths": paths, "caption": cap,
            "lattice_side": lat_doc["side"],
            "lattice_frame_index": lat_doc["frame_index"],
            "lattice_occupancy": lat_doc["occupancy"],
            "trajectory_index": tr["index"],
            "archive_physical_time": tr["physical_time"],
            "displayed_time": list(DISPLAY_TIME["in"]),
            "display_time_note": DISPLAY_TIME_NOTE,
            "contains_sinn_output": False,
            "symbol": "rho", "y_axis_label": "population density"}


# ---------------------------------------------- Supplementary run-level tables --
def run_tables(final: dict, class_dir: str, out_dir: str) -> dict:
    cls = {}
    for p in sorted(glob.glob(os.path.join(class_dir,
                                           "classification_run_*.json"))):
        if not re.search(r"classification_run_\d+\.json$", p):
            continue
        with open(p) as fh:
            d = json.load(fh)
        cls[int(d["run"])] = d
    rows = sorted(final["runs"], key=lambda r: int(r["run"]))

    txt = os.path.join(out_dir, "tableS_run_level.txt")
    with open(txt, "w") as fh:
        fh.write("Supplementary: all %d runs, validation and final. "
                 "Unsuccessful runs are marked.\n" % len(rows))
        fh.write("All r are ratios of SQUARED relative discrepancies; the "
                 "unsquared ratio is sqrt(r). Ensemble size N = 400.\n\n")
        fh.write("%4s %4s %7s %11s %9s %7s %9s %9s %10s %11s\n"
                 % ("run", "ok", "ckpt it", "Q_val", "r_val", "worst",
                    "r_in", "r_out", "M/eps_lag", "invalid out"))
        for r in rows:
            n = int(r["run"])
            c = cls.get(n, {})
            fh.write("%4d %4s %7d %11.4e %9.2f %7s %9.2f %9.2f %10.3f "
                     "%11.3e\n"
                     % (n, "Y" if r["accepted"] else "N",
                        int(r["checkpoint_iteration"]),
                        float(r["Q_val_at_checkpoint"]),
                        float(c.get("r_val", float("nan"))),
                        c.get("worst_family", "?"),
                        float(r["r_worst_squared"]["in"]),
                        float(r["r_worst_squared"]["out"]),
                        float(r["M_SINN_over_eps_lag"]),
                        float(r["admissibility"]["out"]["invalid_rate"])))

        fh.write("\nPer-family final losses and ratios, and Delta, by run\n")
        for r in rows:
            fh.write("\nrun %d (%s)\n" % (int(r["run"]),
                                          "successful" if r["accepted"]
                                          else "not successful"))
            fh.write("  %-6s %12s %12s %9s %9s %10s\n"
                     % ("family", "L_in", "L_out", "r_in", "r_out", "Delta"))
            for s in FAMILIES:
                fh.write("  %-6s %12.4e %12.4e %9.2f %9.2f %+10.3f\n"
                         % (s, r["L_eval"]["in"][s], r["L_eval"]["out"][s],
                            r["r_eval_squared"]["in"][s],
                            r["r_eval_squared"]["out"][s], r["Delta"][s]))

        fh.write("\nReference errors used as denominators\n")
        fh.write("  %-6s %14s %14s %14s\n"
                 % ("family", "L_ref^eval,in", "L_ref^eval,out", "common Lbar"))
        for s in FAMILIES:
            fh.write("  %-6s %14.4e %14.4e %14.4e\n"
                     % (s, final["L_ref_eval"]["in"][s],
                        final["L_ref_eval"]["out"][s],
                        final["L_bar_ref_eval"][s]))

    tex = os.path.join(out_dir, "tableS_run_level.tex")
    with open(tex, "w") as fh:
        fh.write("% Supplementary run-level table. Requires booktabs.\n")
        fh.write("\\begin{tabular}{rcrrrlrrrr}\n\\toprule\n")
        fh.write("Run & Succ. & Iter. & $Q_{\\rm val}$ & $r^{\\rm val}$ & "
                 "Worst & $r_{\\rm in}$ & $r_{\\rm out}$ & "
                 "$M_{\\rm SINN}/\\epsilon_{\\rm lag}$ & Invalid \\\\\n")
        fh.write("\\midrule\n")
        for r in rows:
            n = int(r["run"])
            c = cls.get(n, {})
            fh.write("%d & %s & %d & %.3e & %.2f & %s & %.2f & %.2f & %.3f & "
                     "%.1e \\\\\n"
                     % (n, "\\checkmark" if r["accepted"] else "--",
                        int(r["checkpoint_iteration"]),
                        float(r["Q_val_at_checkpoint"]),
                        float(c.get("r_val", float("nan"))),
                        str(c.get("worst_family", "?")).replace("_", "\\_"),
                        float(r["r_worst_squared"]["in"]),
                        float(r["r_worst_squared"]["out"]),
                        float(r["M_SINN_over_eps_lag"]),
                        float(r["admissibility"]["out"]["invalid_rate"])))
        fh.write("\\bottomrule\n\\end{tabular}\n")
    return {"txt": txt, "tex": tex, "n_runs": len(rows),
            "n_classified": len(cls)}


# ------------------------------------------------ Supplementary stability tab --
def stability_table(stab: dict, out_dir: str) -> dict:
    w = stab["retained_windows"]
    on = stab["online_accumulation"]
    tex = os.path.join(out_dir, "tableS_stability.tex")
    with open(tex, "w") as fh:
        fh.write("% Supplementary 10^6-step stability table. Requires "
                 "booktabs.\n")
        fh.write("% The five losses are squared relative discrepancies "
                 "against fixed\n%   T_ev,in statistics: no reference "
                 "denominator, no threshold, and no\n%   interpretation as "
                 "10^6-step target accuracy.\n")
        fh.write("\\begin{tabular}{rrrrr rrrrr}\n\\toprule\n")
        fh.write("Step & $\\bar x_A$ & $\\bar x_B$ & $\\hat\\sigma_A$ & "
                 "$\\hat\\sigma_B$ & $\\mathcal{L}_{\\rm ACF}$ & "
                 "$\\mathcal{L}_{\\rm CCF}$ & $\\mathcal{L}_{\\rm ACF2}$ & "
                 "$\\mathcal{L}_{\\rm PDF}$ & "
                 "$\\mathcal{L}_{\\rm PROJ}$ \\\\\n\\midrule\n")
        for r in w:
            L = r["family_loss_vs_T_ev_in"]
            fh.write("$%d{\\times}10^{5}$ & %+.5f & %+.5f & %.5f & %.5f & "
                     "%.3e & %.3e & %.3e & %.3e & %.3e \\\\\n"
                     % (int(r["checkpoint_step"] // 100000),
                        r["pooled_mean_x"]["A"], r["pooled_mean_x"]["B"],
                        r["pooled_sd_x"]["A"], r["pooled_sd_x"]["B"],
                        L["ACF"], L["CCF"], L["ACF2"], L["PDF"], L["PROJ"]))
        fh.write("\\bottomrule\n\\end{tabular}\n")
    note = os.path.join(out_dir, "tableS_stability.caption.txt")
    with open(note, "w") as fh:
        fh.write(
            "Supplementary. Stability of the validation-selected "
            "representative run (seed %d, checkpoint iteration %d) over "
            "10^6 post-warm-up recurrent steps with N = 50 trajectories, no "
            "state reset, and no clipping, projection or rounding. Each row "
            "summarises the 10^4 steps preceding the listed step, split into "
            "25 contiguous 400-step blocks (1250 analysis blocks of length "
            "T = 400). The five losses are squared relative discrepancies "
            "against fixed T_ev,in statistics; they have no reference "
            "denominator, no threshold, and no interpretation as pathwise or "
            "10^6-step target accuracy. Accumulated online over all 10^6 "
            "evaluated steps and %d state-time samples: %d non-finite "
            "states, %d finite constraint violations, invalid rate %.3e, "
            "maximum exceedance %.6e; no first non-finite or first violation "
            "step exists.\n"
            % (stab["protocol"]["stability_stream_seed"],
               stab["representative_checkpoint_iteration"],
               on["n_state_time_samples"], on["n_nonfinite"],
               on["n_finite_violating"], on["invalid_rate"],
               on["max_exceedance"]))
    return {"tex": tex, "caption": note, "n_windows": len(w)}


# ---------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-dir", default="final_llv")
    ap.add_argument("--shortlag-dir", default="shortlag_llv")
    ap.add_argument("--setup-dir", default="setup_llv")
    ap.add_argument("--curves-dir", default="curves_llv")
    ap.add_argument("--density-dir", default="density_llv")
    ap.add_argument("--stability-dir", default="stability_llv")
    ap.add_argument("--class-dir", default="class_formal")
    ap.add_argument("--calib-dir", default="calib_llv")
    ap.add_argument("--select-dir", default="select_formal")
    ap.add_argument("--out", default="figures_llv")
    args = ap.parse_args()

    try:
        plt = apply_style()
    except ImportError as exc:
        raise SystemExit("matplotlib is required to draw the figures: %s"
                         % exc)

    os.makedirs(args.out, exist_ok=True)
    print("LLV figure and table builder, spec section 13 stage 7")
    print("reads only saved products; regenerates nothing\n")

    final, final_sha = load_verified(
        os.path.join(args.final_dir, "final_evaluation.json"),
        label="final_evaluation.json")
    short, short_sha = load_verified(
        os.path.join(args.shortlag_dir, "shortlag_curves.json"),
        label="shortlag_curves.json")
    curves, curves_sha = load_verified(
        os.path.join(args.curves_dir, "longlag_curves.json"),
        label="longlag_curves.json")
    dens, dens_sha = load_verified(
        os.path.join(args.density_dir, "joint_density.json"),
        label="joint_density.json")
    stab, stab_sha = load_verified(
        os.path.join(args.stability_dir, "stability.json"),
        label="stability.json")
    calib, calib_sha = load_verified(
        os.path.join(args.calib_dir, "calibration.json"),
        label="calibration.json")
    sel, sel_sha = load_verified(
        os.path.join(args.select_dir, "representative.json"),
        sha_path=os.path.join(args.select_dir, "representative.sha256"),
        label="representative.json")
    setup_json = os.path.join(args.setup_dir, "setup_data.json")
    setup, setup_sha = ((None, None) if not os.path.exists(setup_json)
                        else load_verified(setup_json,
                                           label="setup_data.json"))
    print("all %d input products match their published checksums"
          % (7 + (setup is not None)))

    # the products must agree about which experiment this is
    for name, d in (("shortlag", short), ("curves", curves), ("density", dens),
                    ("stability", stab)):
        if int(d["representative_run"]) != int(sel["representative_run"]):
            raise SystemExit("%s names a different representative run" % name)
        if d["representative_selection_sha256"] != sel_sha:
            raise SystemExit("%s was built against a different frozen "
                             "selection" % name)
    checked = [("final", final), ("shortlag", short), ("curves", curves),
               ("density", dens), ("stability", stab), ("calibration", calib)]
    if setup is not None:
        checked.append(("setup", setup))
    for name, d in checked:
        if d["archive_sha256"] != final["archive_sha256"]:
            raise SystemExit("%s was built against a different archive" % name)
    # the short-lag export is only usable if it reproduced the stored losses
    if not short["cross_check_vs_stage_4"].get("identical"):
        raise SystemExit("the short-lag export does not record an exact "
                         "match against the stored G_out ACF/CCF losses")
    rep_rows = [r for r in final["runs"]
                if int(r["run"]) == int(sel["representative_run"])]
    if len(rep_rows) != 1:
        raise SystemExit("stage 4 has %d records for the representative run"
                         % len(rep_rows))
    for s in ("ACF", "CCF"):
        if float(short["cross_check_vs_stage_4"]["L_here"][s]) != \
                float(rep_rows[0]["L_eval"]["out"][s]):
            raise SystemExit("the short-lag export disagrees with the stored "
                             "%s G_out loss for the representative run" % s)
    print("the short-lag export reproduces the stored G_out ACF and CCF "
          "losses exactly (checked on the window-normalised arrays; the "
          "main-text figure displays them pair-count normalised)")
    if abs(float(curves["band"]["eps_lag"])
           - float(calib["long_lag_envelope"]["eps_lag"])) != 0.0:
        raise SystemExit("the curve export and the calibration disagree "
                         "about eps_lag")
    print("representative run %d, checkpoint iteration %d; M = %d of %d"
          % (sel["representative_run"],
             sel["representative_checkpoint_iteration"], final["M"],
             len(final["runs"])))
    print("eps_lag = %.6f, consistent across calibration and the curves\n"
          % float(curves["band"]["eps_lag"]))

    arts = {}
    arts["table_main_endpoint"] = endpoint_table(final, args.out)
    print("endpoint table                -> %s, %s"
          % (os.path.basename(arts["table_main_endpoint"]["txt"]),
             os.path.basename(arts["table_main_endpoint"]["tex"])))

    arts["fig_llv_family_ratios"] = family_ratio_boxplot(final, args.out, plt)
    _b = arts["fig_llv_family_ratios"]
    print("family-ratio box plot         -> fig_llv_family_ratios.pdf/.png "
          "(M = %d of %d, rejected %s; no caption file)"
          % (_b["M"], _b["n_runs"],
             ", ".join(str(r) for r in _b["rejected_runs"])))
    print("  every box matches the stored endpoint table's median/Q1/Q3")

    short_npz = os.path.join(args.shortlag_dir, "shortlag_curves.npz")
    short_npz_sha = verify_artifact(short, short_npz, "shortlag_curves.json")
    curves_npz = os.path.join(args.curves_dir, "longlag_curves.npz")
    curves_npz_sha = verify_artifact(curves, curves_npz,
                                     "longlag_curves.json")
    dens_npz = os.path.join(args.density_dir, "joint_density.npz")
    dens_npz_sha = verify_artifact(dens, dens_npz, "joint_density.json")
    setup_npz = setup_npz_sha = None
    if setup is not None:
        setup_npz = os.path.join(args.setup_dir, "setup_data.npz")
        setup_npz_sha = verify_artifact(setup, setup_npz, "setup_data.json")
    print("every binary artifact matches the checksum its JSON records\n")

    arts["fig_main_correlation"] = shortlag_figure(
        short_npz, short, args.out, plt)
    _s = arts["fig_main_correlation"]
    print("main-text correlation figure  -> fig_main_correlation.pdf/.png "
          "(k = 0..%d, pair-count normalised, no envelope)" % _s["k_max"])
    print("  shared vertical range %s; panel extrema %s"
          % (_s["ylim"],
             ", ".join("%s %+.4f..%+.4f" % (nm, v[0], v[1])
                       for nm, v in _s["panel_extremes"].items())))
    if _s.get("superseded_copy"):
        print("  the version it replaces was kept at %s"
              % _s["superseded_copy"])

    arts["figS_llv_longlag_correlation"] = correlation_figure(
        curves_npz, curves, args.out, plt)
    print("long-lag figure (Supplementary) -> "
          "figS_llv_longlag_correlation.pdf/.png")

    if setup is None:
        arts["fig_llv_setup"] = {
            "skipped": "no setup_data.json in %r; run llv_setup_data.py"
                       % args.setup_dir}
        print("setup figure                  -> SKIPPED: %s"
              % arts["fig_llv_setup"]["skipped"])
    else:
        arts["fig_llv_setup"] = setup_figure(setup_npz, setup, args.out, plt)
        if "skipped" in arts["fig_llv_setup"]:
            print("setup figure                  -> SKIPPED: %s"
                  % arts["fig_llv_setup"]["skipped"])
        else:
            print("setup figure                  -> fig_llv_setup.pdf/.png "
                  "(lattice frame %d, trajectory %d, t = %d..%d)"
                  % (arts["fig_llv_setup"]["lattice_frame_index"],
                     arts["fig_llv_setup"]["trajectory_index"],
                     *arts["fig_llv_setup"]["displayed_time"]))

    arts["figS_sliding_window"] = sliding_figure(final, args.out, plt)
    if "skipped" in arts["figS_sliding_window"]:
        print("sliding-window figure         -> SKIPPED: %s"
              % arts["figS_sliding_window"]["skipped"])
    else:
        print("sliding-window figure         -> figS_sliding_window.pdf/.png "
              "(%d windows)" % arts["figS_sliding_window"]["n_windows"])

    arts["figS_joint_density"] = density_figure(dens_npz, dens, args.out, plt)
    _d = arts["figS_joint_density"]
    print("joint-density figure          -> figS_joint_density.pdf/.png "
          "(1x2; contours at %s x max f_KMC = %.4g; marginal axis %.4f..%.4f)"
          % ("/".join("%.1f" % f for f in _d["contour_level_fractions"]),
             _d["f_kmc_max_in_window"], *_d["marginal_axis"]["common_axis"]))

    arts["tableS_run_level"] = run_tables(final, args.class_dir, args.out)
    print("run-level tables              -> tableS_run_level.txt/.tex "
          "(%d runs, %d classifications)"
          % (arts["tableS_run_level"]["n_runs"],
             arts["tableS_run_level"]["n_classified"]))

    arts["tableS_stability"] = stability_table(stab, args.out)
    print("stability table               -> tableS_stability.tex "
          "(%d windows)" % arts["tableS_stability"]["n_windows"])

    manifest = {
        "stage": "figure and table builder",
        "regenerated_model_output": False,
        "regeneration_note": "this builder reads saved products only; the "
                             "short-lag curves were produced upstream by "
                             "llv_shortlag.py, which reproduced one "
                             "deterministic rollout of the frozen "
                             "representative checkpoint and asserted that it "
                             "returns the stored G_out ACF and CCF losses "
                             "exactly",
        "retrained_anything": False,
        "rebuilt_archive": False,
        "changed_accepted_mask": False,
        "recomputed_any_endpoint_summary": False,
        "reclassified_any_run": False,
        "reselected_checkpoint": False,
        "recomputed_any_summary_statistic": False,
        "inputs": {k: v for k, v in (
            ("final_evaluation.json", final_sha),
            ("shortlag_curves.json", short_sha),
            ("longlag_curves.json", curves_sha),
            ("joint_density.json", dens_sha),
            ("stability.json", stab_sha),
            ("calibration.json", calib_sha),
            ("representative.json", sel_sha),
            ("setup_data.json", setup_sha),
            ("shortlag_curves.npz", short_npz_sha),
            ("longlag_curves.npz", curves_npz_sha),
            ("joint_density.npz", dens_npz_sha),
            ("setup_data.npz", setup_npz_sha),
        ) if v is not None},
        "figure_roles": {
            "fig_main_correlation": "main text; lags 0..%d on G_out, "
                                    "pair-count normalised for display, "
                                    "no envelope" % (S.K_LAGS - 1),
            "fig_llv_setup": "main text; illustrative, no SINN output",
            "fig_llv_family_ratios": "main text; replaces the endpoint table; "
                                     "per-segment reference levels, so the "
                                     "two panels are not comparable as an "
                                     "error improvement; no caption file is "
                                     "written and no explanatory text is "
                                     "drawn inside the figure",
            "table_main_endpoint": "retained as the numerical record behind "
                                   "fig_llv_family_ratios; the box plot is "
                                   "what the main text shows",
            "figS_llv_longlag_correlation": "Supplementary; lags %d..%d with "
                                            "the validation-calibrated "
                                            "envelope, data unchanged"
                                            % (S.LONGLAG_K_LO,
                                               S.LONGLAG_K_HI),
            "figS_sliding_window": "Supplementary",
            "figS_joint_density": "Supplementary",
            "tableS_run_level": "Supplementary",
            "tableS_stability": "Supplementary",
        },
        "shortlag_cross_check": short["cross_check_vs_stage_4"],
        "representative_run": int(sel["representative_run"]),
        "representative_checkpoint_iteration":
            int(sel["representative_checkpoint_iteration"]),
        "M": int(final["M"]),
        "n_runs": len(final["runs"]),
        "eps_lag": float(curves["band"]["eps_lag"]),
        "notation": {
            "ensemble_size_symbol": "N",
            "N_final": S.B_GEN, "N_stability": 50, "N_vis": S.B_VIS,
            "note": "the symbol B is not used for an ensemble size in any "
                    "figure, table or caption produced here",
        },
        "units": "every r is a ratio of squared relative discrepancies; the "
                 "unsquared ratio is sqrt(r); Delta is a difference of "
                 "squared quantities on the common scale",
        "style": {"STYLE": STYLE, "rcParams": RCPARAMS,
                  "source": "Section 4 established style",
                  "full_width_inches": WIDTH,
                  "colour_semantics":
                      "two conventions, by figure. The main-text short-lag "
                      "correlation figure follows Figure 11: black solid is "
                      "the target ensemble, red dashed the SINN ensemble, so "
                      "hue carries the source and the quantity is carried by "
                      "the panel titles; its legend proxies use the same two "
                      "inks. Everywhere else hue carries the quantity and "
                      "line style the source -- in the Supplementary "
                      "long-lag correlation figure hue is the correlation "
                      "function, in the sliding-window figure hue is the "
                      "family, in the density marginals hue is the component "
                      "-- with Target solid and SINN dashed, and those "
                      "shared legends are drawn from neutral proxy artists "
                      "so no legend entry depends on hue. Target and SINN "
                      "differ in line style under both conventions.",
                  "series_ink_shortlag": dict(SERIES_INK),
                  "curve_colors_longlag": CURVE_COLORS,
                  "family_colors": FAMILY_COLORS,
                  "rho_colors": RHO_COLORS,
                  "neutral": NEUTRAL,
                  "provisional_colors": list(PROVISIONAL_COLORS),
                  "provisional_note":
                      "the colours listed in provisional_colors were not "
                      "supplied with the Section 4 palette and must be "
                      "checked against Section 4 before publication"},
        "greyscale": "Target and SINN differ in line style, the two "
                     "sliding-window bands differ in edge style as well as "
                     "hue, and the density panels use one light-to-dark hue "
                     "and share one colour scale and one coordinate range",
        "captions_are_drafts": "the .caption.txt files are short factual "
                               "notes, not final manuscript text",
        "artifacts": {k: (v.get("paths") or
                          {kk: v[kk] for kk in ("txt", "tex", "caption")
                           if kk in v}) for k, v in arts.items()},
        # the full record each builder step returned, so the checks it made
        # (box-versus-table, clipping, the pair-count conversion, the display
        # renumbering) are in the manifest and not only on the terminal
        "artifact_records": {k: {kk: vv for kk, vv in v.items()
                                 if kk not in ("paths", "caption")}
                             for k, v in arts.items()},
        "captions": {k: v.get("caption") for k, v in arts.items()
                     if v.get("caption")},
        "source_sha256": {n: S.sha256_file(n) for n in
                          ("llv_spec.py", "llv_figures.py")
                          if os.path.exists(n)},
    }
    p = os.path.join(args.out, "figures_manifest.json")
    with open(p, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    digest = S.sha256_file(p)
    with open(os.path.join(args.out, "figures_manifest.sha256"), "w") as fh:
        fh.write(digest + "\n")
    print("\nmanifest -> %s (sha %s...)" % (p, digest[:12]))
    print("\nSTYLE is a single dictionary at the top of this file. The "
          "figures are drawn from saved data, so restyling is one rerun of "
          "this script and no model computation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
