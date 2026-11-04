#!/usr/bin/env python
"""Section 5.1 main figure: family-level evaluation ratios, tau10 convention.

Everything plotted is read from the saved full-precision per-run file written
by build_section51_tau10.py.  Nothing is recomputed and no run is reclassified:
the accepted mask is the one the validation pass produced.

Encoding, 2026-10-10 revision.  The boxes now carry colour, because the two
series they separate -- SINN against reference block -- are the only thing the
marks have to distinguish, and the five families are already named on the
horizontal axis.  Red is the SINN colour used for the dashed curves and
contours in Figs. 10 and 11; blue is the target colour there.

    red box    accepted SINN runs, 1.5-IQR whiskers, outliers as open circles
    blue box   the nine reference blocks, same conventions
    red square rejected run, an open crossed square, outside the box
               statistics and in a column of its own

Each family occupies three slots: the SINN box, then the rejected column,
then the reference box.  Giving the rejected runs their own slot keeps them
clear of the boxes' outlier circles, which sit on the box centre lines.
Individual accepted runs and individual reference blocks are not drawn, and
nothing is jittered.

    python fig_family_ratios.py --root <refresh root>
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style51 import (FAMILIES, REF_LINE, SINN_LINE, WIDTH,  # noqa: E402
                     panel, style)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

# Section 4 already has a paired two-colour boxplot figure -- the threshold
# attainment panel, Fig. 5(b) -- and this one now matches it exactly: the same
# two hues, saturated for every line, and the same 0.30 face transparency.
FACE_ALPHA = 0.30
SINN_FILL = to_rgba(SINN_LINE, FACE_ALPHA)
REF_FILL = to_rgba(REF_LINE, FACE_ALPHA)

BOX_LW, MED_LW = 0.6, 0.9

# Three slots per family, SLOT apart: SINN box, rejected column, reference
# box.  The rejected column lands on the family tick, so the painted group is
# symmetric about it.  SLOT and BOXW are chosen so that the gap inside a
# family (2*SLOT - BOXW = 0.26) stays smaller than the gap between families
# (1 - 2*SLOT - BOXW = 0.34); otherwise a family's reference box reads as
# belonging to the next family's SINN box.
BOXW, SLOT = 0.20, 0.23
SINN_OFF = -SLOT
REJ_OFF = 0.0
REF_OFF = SLOT
FLIER_MS, REJ_MS = 3.0, 5.0

# One marker, not two: an open square with a cross set inside it.  Drawing it
# as a single Path means the legend handle and the plotted mark cannot drift
# apart.  Coordinates are in marker-size units, so the square spans the full
# markersize and the cross stops short of the corners.
_S, _X = 0.5, 0.26
CROSSBOX = Path(
    [(-_S, -_S), (_S, -_S), (_S, _S), (-_S, _S), (-_S, -_S),
     (-_X, -_X), (_X, _X), (-_X, _X), (_X, -_X)],
    [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY,
     Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO])


def box_style(edge, fill):
    """Every property of one box, so the legend patch can reuse it verbatim."""
    return dict(
        widths=BOXW, patch_artist=True, whis=1.5, showfliers=True,
        boxprops=dict(facecolor=fill, edgecolor=edge, lw=BOX_LW),
        medianprops=dict(color=edge, lw=MED_LW),
        whiskerprops=dict(color=edge, lw=BOX_LW),
        capprops=dict(color=edge, lw=BOX_LW),
        flierprops=dict(marker="o", ms=FLIER_MS, mfc="none", mec=edge,
                        mew=0.7, ls="none"))


def rejected_marks(ax, x, y):
    """Open crossed square, in the rejected column of its family."""
    ax.plot([x], [y], marker=CROSSBOX, ls="none", ms=REJ_MS, mfc="none",
            mec=SINN_LINE, mew=0.7, zorder=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    style()

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9), sharey=True)
    rows, meta, span = [], {}, []
    pos = np.arange(len(FAMILIES))
    sinn = box_style(SINN_LINE, SINN_FILL)
    ref = box_style(REF_LINE, REF_FILL)

    for ax, d, tag in zip(axes, (2, 3), "ab"):
        z = np.load(os.path.join(a.root, "data",
                                 "finaleval_d%d_w75_tau10.npz" % d))
        assert list(z["keys"]) == list(FAMILIES)
        r, rblk = z["r_eval"], z["r_blocks"]
        ok, seed = z["accepted"].astype(bool), z["seed"]
        meta["d%d" % d] = dict(n_runs=int(len(seed)), n_accepted=int(ok.sum()),
                               n_blocks=int(rblk.shape[0]))

        for j, _f in enumerate(FAMILIES):
            ax.boxplot([r[ok, j]], positions=[pos[j] + SINN_OFF], **sinn)
            ax.boxplot([rblk[:, j]], positions=[pos[j] + REF_OFF], **ref)
            for i in np.nonzero(~ok)[0]:
                rejected_marks(ax, pos[j] + REJ_OFF, r[i, j])

        ax.axhline(1.0, color="0.55", lw=0.5, zorder=0)
        ax.set_yscale("log")
        ax.set_xticks(pos)
        ax.set_xticklabels(FAMILIES)
        ax.set_xlim(-0.6, len(FAMILIES) - 0.4)
        ax.tick_params(which="both", top=False, right=False)
        span.append(np.concatenate([r.ravel(), rblk.ravel()]))
        # Panel letter and panel count are one label now, at the panel
        # letter's usual corner; the wording matches the text, which calls
        # the two groups validation-successful and validation-unsuccessful.
        # in_layout=False keeps it out of tight_layout's margin calculation:
        # the label is longer than the old one and would otherwise push the
        # two panels narrower, changing the figure width and hence the scale
        # at which LaTeX includes it.  It sits well inside the axes.
        ax.text(0.03, 0.97,
                r"(%s) $d = %d$,  %d/%d validation-successful"
                % (tag, d, ok.sum(), len(ok)), transform=ax.transAxes,
                va="top", ha="left", fontsize=8.5, in_layout=False)
        if d == 2:
            ax.set_ylabel(r"$r_s^{\mathrm{eval}}$")
            # Three legend columns, filled top-to-bottom: each series sits
            # above its own outlier marker, and the rejected column stands
            # alone between them.  matplotlib fills a legend column-major,
            # so the blank entry holds the empty cell under "rejected run".
            circle = lambda c: Line2D([], [], ls="none", marker="o",
                                      ms=FLIER_MS, mfc="none", mec=c, mew=0.7)
            handles = [
                Patch(facecolor=SINN_FILL, edgecolor=SINN_LINE, lw=BOX_LW),
                circle(SINN_LINE),
                Line2D([], [], ls="none", marker=CROSSBOX, ms=REJ_MS,
                       mfc="none", mec=SINN_LINE, mew=0.7),
                Line2D([], [], ls="none", marker="none"),
                Patch(facecolor=REF_FILL, edgecolor=REF_LINE, lw=BOX_LW),
                circle(REF_LINE)]
            labels = ["validation-successful", r"outlier (1.5 IQR)",
                      "validation-unsuccessful", "",
                      "reference blocks", r"outlier (1.5 IQR)"]

        for i, s in enumerate(seed):
            for j, f in enumerate(FAMILIES):
                rows.append(dict(panel="(%s)" % tag, target_dim=d,
                                 series="SINN", id="s%d" % s,
                                 accepted=int(ok[i]), family=f,
                                 r_eval=float(r[i, j])))
        for i in range(rblk.shape[0]):
            for j, f in enumerate(FAMILIES):
                rows.append(dict(panel="(%s)" % tag, target_dim=d,
                                 series="reference", id="R%d" % (i + 1),
                                 accepted="", family=f,
                                 r_eval=float(rblk[i, j])))

    # One shared range across both panels: sharey means the last set wins,
    # so the limit is applied once, after every observation is known.
    allv = np.concatenate(span)
    axes[0].set_ylim(allv.min() / 2.0, allv.max() * 4.0)

    fig.legend(handles, labels, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), labelspacing=0.35,
               columnspacing=2.2, handletextpad=0.7)
    fig.tight_layout(pad=0.4, rect=[0, 0, 1, 0.88])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.root, "figs",
                                 "fig_fpu_family_ratios_tau10." + ext))
    print("wrote fig_fpu_family_ratios_tau10.pdf / .png")

    with open(os.path.join(a.root, "data",
                           "fig_fpu_family_ratios_tau10_values.csv"), "w",
              newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    cap = (
        r"\caption{Final-evaluation family-loss ratios for the selected "
        r"75-unit architecture: (a) $\boldsymbol{X}^{\mathrm{2D}}$ and "
        r"(b) $\boldsymbol{X}^{\mathrm{3D}}$. Red boxes summarize the runs "
        r"accepted on $\mathcal V$ (%d/%d and %d/%d) and blue boxes the nine "
        r"reference blocks; both show medians, interquartile ranges and "
        r"$1.5$-IQR whiskers, with points beyond the whiskers drawn as open "
        r"circles. Rejected runs enter no box statistic and are drawn as "
        r"crossed squares in a separate column to the left of each pair of "
        r"boxes. All estimates "
        r"compare 400-trajectory ensembles with $|\mathcal T|=12{,}000$; the "
        r"ordinate is logarithmic and the horizontal line marks the reference "
        r"level $r_s^{\mathrm{eval}}=1$.}"
        % (meta["d2"]["n_accepted"], meta["d2"]["n_runs"],
           meta["d3"]["n_accepted"], meta["d3"]["n_runs"]))
    with open(os.path.join(a.root, "tables",
                           "fig_fpu_family_ratios_tau10_caption.tex"), "w") as fh:
        fh.write(cap + "\n")
        fh.write("%% %d words in the caption body.\n"
                 % len(cap.split("\\caption{")[1].split()))
    with open(os.path.join(a.root, "metadata",
                           "fig_fpu_family_ratios_tau10.json"), "w") as fh:
        json.dump(dict(family_order=list(FAMILIES), panels=meta,
                       quantity="r_s^eval = L_s^eval / L_s,ref^eval",
                       y_scale="log", n_gen=400, n_blocks=9, n_test=12000,
                       boxes="validation-accepted runs only",
                       encoding=dict(
                           scheme="two-colour, identical to Section 4 "
                                  "Fig. 5(b) (threshold attainment)",
                           rationale="colour separates SINN from reference "
                                     "block, the only distinction the marks "
                                     "must carry; the five families are "
                                     "already named on the horizontal axis",
                           sinn_box=dict(facecolor="%s at alpha %.2f"
                                         % (SINN_LINE, FACE_ALPHA),
                                         edgecolor=SINN_LINE,
                                         median=SINN_LINE, lw=BOX_LW),
                           reference_box=dict(facecolor="%s at alpha %.2f"
                                              % (REF_LINE, FACE_ALPHA),
                                              edgecolor=REF_LINE,
                                              median=REF_LINE, lw=BOX_LW),
                           outliers="open circles, ms %.1f, series colour"
                                    % FLIER_MS,
                           rejected_run="open crossed square, single Path "
                                        "marker, ms %.1f, %s, in its own "
                                        "column at offset %+.2f"
                                        % (REJ_MS, SINN_LINE, REJ_OFF),
                           slot_order="SINN box, rejected column, "
                                      "reference box",
                           scatter="none; individual accepted runs and "
                                   "reference blocks are not drawn",
                           jitter="none",
                           legend_layout="3 columns x 2 rows: SINN box over "
                                         "its outlier circle, rejected run "
                                         "alone, reference box over its "
                                         "outlier circle",
                           legend_fills_match_boxes=True),
                       whiskers="1.5 IQR, standard matplotlib fliers shown",
                       box_offsets=dict(rejected=REJ_OFF, sinn=SINN_OFF,
                                        reference=REF_OFF, box_width=BOXW,
                                        slot_spacing=SLOT),
                       revision="2026-10-15: slot order is SINN box, "
                                "rejected column, reference box, with the "
                                "rejected column on the family tick; box "
                                "width 0.22, slots 0.28 apart. Legend "
                                "regrouped into three columns of two. Data, "
                                "accepted mask, colours and axis limits "
                                "unchanged"),
                  fh, indent=1)


if __name__ == "__main__":
    main()
