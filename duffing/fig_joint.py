#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The n_f figure: the target joint density with the SINN's overlaid, one panel per
number of projection frames.

WHAT THIS FIGURE IS EVIDENCE FOR. The bivariate density enters no configuration's
training objective. PROJ does -- the projected-PDF loss optimises it directly --
so "PROJ improves as n_f rises" is a statement about the objective. The joint
density is the quantity the projected constraints are supposed to be a proxy
for, and it is what separates the four settings.

CONTOUR LEVELS ARE MASS, NOT DENSITY. The lines enclose 10, 30, 50, 70 and 90
percent of the target's probability mass, computed once from the target and
reused for the SINN, so the two are drawn on identical levels and a visible gap
is a real disagreement rather than a different choice of contour. Quoting raw
density values instead would make the outermost line depend on the grid.

Colours follow the manuscript: the reference is the filled light-blue field with
solid dark lines; the SINN is red dashed. The displayed run of each panel is the
one named in its report_*.json, chosen by the predeclared rule (median validation
aggregate over that configuration's runs) and never by how the contours look.

Usage
-----
    python fig_joint.py --lam 4
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import evaluate as EV

WIDTH = 6.02
REF_LINE = "#1b4f72"
SINN_LINE = "#a93226"
MASS = (0.10, 0.30, 0.50, 0.70, 0.90)


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        # MINOR TICKS. Added so the panels carry the same tick furniture as the
        # rest of the figures. Thinner and shorter than the majors, so they
        # read as subdivisions rather than competing with them.
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "xtick.minor.visible": True, "ytick.minor.visible": True, "lines.linewidth": 0.9,
        "axes.grid": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def mass_levels(p, dz, fractions=MASS):
    """Density values whose super-level sets carry the given probability mass."""
    flat = np.sort(p.ravel())[::-1]
    cum = np.cumsum(flat) * dz * dz
    cum /= cum[-1]
    lv = [float(flat[np.searchsorted(cum, f)]) for f in fractions]
    return sorted(set(lv))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--repdir", default="reports")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--lam", type=float, default=4.0)
    ap.add_argument("--nfs", default="0,1,2,3")
    ap.add_argument("--lim", type=float, default=3.5)
    args = ap.parse_args()

    style()
    os.makedirs(args.outdir, exist_ok=True)
    tag = f"lam{args.lam:g}".replace(".", "p")
    nfs = [int(v) for v in args.nfs.split(",")]

    traj_path = os.path.join(args.datadir, f"traj_{tag}.npy")
    ref = EV.load_reference(args.datadir, tag, traj_path)
    z = np.asarray(ref["z_grid"])
    dz = float(z[1] - z[0])
    target = np.asarray(ref["joint"])          # analytic, from Eq. (gibbs)
    levels = mass_levels(target, dz)

    fig, axes = plt.subplots(1, len(nfs), figsize=(WIDTH, 1.95))
    shown, anchors = [], set()
    for ax, nf in zip(np.atleast_1d(axes), nfs):
        rp = os.path.join(args.repdir,
                          f"report_nf{nf}_lbon_din2_L2H10_{tag}.json")
        rep = json.load(open(rp))
        # One anchor across the four panels, for the reason given in fig_din.py.
        a = rep.get("anchor", "evaluation")
        if anchors and a not in anchors:
            raise SystemExit(
                f"{rp} was scored against the {a} ensemble while an earlier "
                f"panel used {sorted(anchors)[0]}. Regenerate the reports with "
                f"a single --against setting.")
        anchors.add(a)
        run = rep["display_run"]
        gen, space = EV.load_generated(
            os.path.join(args.repdir, f"gen_{run}.npz"), ref, "standardized")
        s = EV.stats(np.asarray(gen, dtype=np.float64), ref, space)
        model = np.asarray(s["JOINT"])

        # THE NUMBER IN THE PANEL DESCRIBES THE CURVE IN THE PANEL. The cell's
        # median over seeds belongs in the table; annotating it beside a single
        # run's contours invites the reader to attach it to that run, and the
        # two differ in both directions (the displayed run is better than the
        # cell median at one n_f and worse at another).
        own = [p for p in rep["per_seed"] if p["run_tag"] == run][0]
        own_ratio = own["diagnostics"]["JOINT"] / rep["floor_eval"]["JOINT"]
        shown.append((nf, run, own_ratio,
                      rep["out_of_objective"]["JOINT"]["ratio"]))

        ax.contourf(z, z, target.T, levels=[levels[0], 1e9],
                    colors=["#eaf1f8"])
        ax.contour(z, z, target.T, levels=levels, colors=[REF_LINE],
                   linewidths=0.8, linestyles="solid")
        ax.contour(z, z, model.T, levels=levels, colors=[SINN_LINE],
                   linewidths=0.8, linestyles="dashed")
        # DISPLAY LABEL IS THE FILE LABEL PLUS ONE. Section 3 counts the frame
        # collection INCLUDING the coordinate frame, so the manuscript's
        # n_f = 1 is the coordinate frame alone. The file names, the CLI and
        # every report_*.json keep the original count of NON-coordinate frames,
        # nf = 0..3, and must not be renamed -- the data on disk is what it is.
        # Only what the reader sees shifts. Same for the table printed below.
        ax.set_title(rf"$n_{{\mathrm{{f}}}}={nf + 1}$")
        ax.set_xlim(-args.lim, args.lim)
        ax.set_ylim(-args.lim, args.lim)
        # EVERY PANEL, BOTH AXES. The right-hand panels hide their y tick
        # LABELS, not their ticks, so the minor locator is set on all of them
        # and the subdivisions line up across the row.
        #
        # THE ARGUMENT IS NOT OPTIONAL. AutoMinorLocator() with no argument
        # picks the number of subdivisions from the major step: the step here
        # is 2, whose mantissa is 2, so matplotlib uses four -- a minor tick
        # every 0.5, three of them inside each major interval. Passing 2 asks
        # for two subdivisions, i.e. exactly one minor tick halfway between
        # consecutive majors.
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x_1$")
        if ax is np.atleast_1d(axes)[0]:
            ax.set_ylabel(r"$x_2$")
        else:
            ax.set_yticklabels([])
        # NO NUMBER IN THE PANEL. The quantitative claim is the table, which is a
        # median over every seed. A number printed beside one run's contours gets
        # attached to that run, and the displayed run need not match its cell:
        # the selection rule is the median VALIDATION AGGREGATE, which is the sum
        # of five squared family errors and not the joint-density error, so a
        # panel can sit off the trend its own cell median follows. Choosing the
        # run by its joint-density error instead would be selecting on the
        # reported quantity. The figure carries the qualitative progression; the
        # table carries the numbers.

    h = [plt.Line2D([], [], color=REF_LINE, ls="-", lw=0.9),
         plt.Line2D([], [], color=SINN_LINE, ls="--", lw=0.9)]
    fig.legend(h, ["target", "SINN"], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.10), handletextpad=0.6, columnspacing=1.6)
    fig.tight_layout(pad=0.3)
    p = os.path.join(args.outdir, f"fig_joint_{tag}.pdf")
    fig.savefig(p)
    plt.close(fig)

    print(f"wrote {p}")
    print("contour levels enclose", ", ".join(f"{100*f:.0f}%" for f in MASS),
          "of the target mass")
    print(f"n_f below is the manuscript's count, including the coordinate frame; "
          f"the file label is the nf in the run name.")
    print(f"{'n_f':<5}{'displayed run':<34}{'panel (own)':<13}{'cell median':<12}")
    for nf, run, own, med in shown:
        print(f"{nf + 1:<5}{run:<34}{own:<13.2f}{med:<12.2f}")

    # Which panels sit off their own cell median. Computed, not hardcoded: it
    # depends on which runs the selection rule picks, so a sentence naming fixed
    # panels goes stale the moment anything is rerun.
    #
    # THE THRESHOLD IS RELATIVE AND NOT TIGHT. The point of the sentence is that
    # a panel can sit off the trend its cell median follows, which is a claim
    # about visible disagreement, not about the last printed digit. An absolute
    # cutoff flags 8.60 against 8.58 -- a 0.2% difference, invisible in the
    # contours -- and then every panel is named and the sentence says nothing.
    # Ten percent separates the two cases here by more than an order of
    # magnitude: 0.2% and 0.2% against 15% and 21%.
    off = [nf + 1 for nf, _, own, med in shown
           if med > 0 and abs(own - med) / med >= 0.10]
    if len(off) == 0:
        which = "above, every panel sits at its own cell median"
    elif len(off) == 1:
        which = f"above, n_f = {off[0]} does not"
    else:
        which = ("above, n_f = "
                 + " and ".join([", ".join(str(v) for v in off[:-1]), str(off[-1])])
                 + " do not")
    print(f"\nfor the caption: each panel shows one run, the one whose validation "
          f"aggregate is the median over that configuration's runs -- a rule fixed "
          f"before any contour was looked at. The panels carry no numbers on "
          f"purpose: the joint-density errors are medians over all seeds and "
          f"belong in the table, and a displayed run need not sit at its own "
          f"cell's median ({which}). Contours enclose "
          f"{', '.join(f'{100*f:.0f}' for f in MASS)}% of the target mass, "
          f"computed once from the target and reused for the SINN.")


if __name__ == "__main__":
    main()
