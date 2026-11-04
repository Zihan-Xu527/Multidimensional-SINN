#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Item 5: the JOINT diagnostic as a direct loss, swept over coupling.

WHAT CHANGED AGAINST fig_proj.py. That figure plotted

    median over runs of   e_JOINT(run) / e_JOINT(reference block) ,

a ratio to the discrepancy between two independent target ensembles. Here the
denominator is gone: the quantity plotted is the loss itself,

    L_JOINT(run) = e_JOINT(run)^2 ,

per run, and the median and interquartile range are taken over the runs of each
(coupling, frame configuration) cell. With no reference-block denominator there
is nothing special about the value 1, so the dashed line at 1 is deleted rather
than left to suggest a meaning it no longer has.

WHAT e_JOINT IS, EXACTLY, AND THE NORMALISATION THAT REMAINS. evaluate.py:275
defines it as _rel(sg, sr) = ||p_model - p_target|| / ||p_target||, RMS norms
over the two-dimensional output grid (evaluate.py:256). So squaring it gives a
RELATIVE squared error. It is not divided by any reference-block loss -- that
is the division this item removes -- but it is still divided by ||p_target||,
which is a fixed constant per coupling fixed by the quadrature-derived target
density. That constant differs between couplings, so the curves are compared
across lambda through it; its value and the resulting unnormalised loss are
printed below whenever the reference archive is at hand, so the reader can see
how much of any trend across lambda lives in the normaliser.

NOT BY SQUARING THE PUBLISHED RATIOS. Two reasons, and either alone is fatal.
The published numbers are medians over runs, and a median of squares is not the
square of a median. And they carry the reference-block denominator, which is
exactly what is being removed. Every value here is read per run from
per_seed[*]["diagnostics"]["JOINT"] in the reports and squared at that point,
before any aggregation.

NOTHING ABOUT THE EVALUATION PIPELINE IS TOUCHED. The target density, the
standardisation, the output grid, the 2-D bandwidth and the Gaussian smoothing
are whatever made these reports; this script only reads what report.py wrote.

THE BASIS, STATED ONCE. JOINT enters no training objective, so there is no
retained-checkpoint record of it. report.py computes it afterwards from an
ensemble regenerated at the fitting size. That is a different basis from the
r^val numbers elsewhere in Section 4, which is why the quantity keeps a
different name.

WHAT IS PLOTTED: THE STORED ERROR, NOT ITS SQUARE. --quantity selects between

    e(run)        the relative discrepancy report.py stored, plotted as it is
    L(run) = e^2  the loss

and the default is the error. This is NOT a relabelling of the same curve.
Quantiles commute with a monotone map only when they are attained values; every
cell here holds 10 or 20 runs, an EVEN count, so the median is the mean of the
two central runs and the quartiles are interpolated -- and a mean does not
commute with a square root. Taking the median of the stored errors therefore
gives a different number from the square root of the median loss, by a second-
order amount. The errors are read per run and aggregated as errors.

ONE SCRIPT, TWO FIGURES. --key selects the diagnostic: JOINT for the text
figure, PROJ_HOLDOUT for Appendix C.3's held-out projection figure on the same
definition. They are the same quantity computed from different entries of the
same per-run record, so they are drawn by the same code rather than by two
files that can drift apart in style. Everything above applies to either, with
the name substituted; for PROJ_HOLDOUT the four held-out directions are
aggregated exactly as evaluate.py already aggregates them (the four curves
concatenated into one vector, numerator and denominator alike), and the target
is the analytic quadrature-derived projected density with the archive's
projection directions, F-derived scales, KDE bandwidth and matched Gaussian
smoothing.

Usage
-----
    python fig_joint_loss.py --repdir reports_val
    python fig_joint_loss.py --repdir reports_val --key PROJ_HOLDOUT \\
                             --yscale log
    python fig_joint_loss.py --repdir reports_val --datadir data \\
                             --out joint_loss.json
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import (FixedLocator, FuncFormatter, LogLocator,
                               MaxNLocator, NullFormatter)

WIDTH = 6.02
RAMP_LIGHT, RAMP_DARK = "#86aecd", "#10304a"
MARKS = ["o", "s", "^", "D"]
PAT = re.compile(r"report_nf(\d)_lb(on|off)_din(\d)_L\d+H\d+_lam([0-9p]+)\.json$")


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
        # Minor ticks read as subdivisions: thinner and about half as long as
        # the majors, and never labelled.
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.minor.size": 1.8, "ytick.minor.size": 1.8, "lines.linewidth": 1.1,
        "axes.grid": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def ramp(n, light=RAMP_LIGHT, dark=RAMP_DARK):
    c0 = np.array([int(light[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    c1 = np.array([int(dark[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return [tuple(c0 + (c1 - c0) * t) for t in np.linspace(0, 1, n)]


def log_ticks(ax, vals, force="auto", majors=None):
    """Pick the vertical scale from the data, or take the one demanded.

    A log axis is right when the panel spans most of a decade or more. Below
    that it is a liability: matplotlib labels only the decades, falls back to
    labelling subdivisions, and the panel ends up with a column of
    '2.75 x 10^-2' strings down the side -- while the curves are nearly
    straight on either scale, so nothing is bought for the cost. The rule is
    mechanical and printed, so the two figures cannot silently end up on
    different scales: a factor of eight or more between the largest and
    smallest value plotted gets a log axis with ticks at 1, 2, 3 and 5 times
    each power of ten; anything narrower gets a linear axis.

    force="log" or "linear" overrides the rule. The span is still measured and
    printed, so an axis that the rule would not have chosen is visible in the
    output rather than only in the figure.

    majors, when given, is the exact list of LABELLED ticks. Everything else of
    the form k x 10^n inside the view becomes an unlabelled minor tick. The
    minors are built by subtracting the major list from the full grid rather
    than by a subs tuple, because a subs tuple applies the same multipliers in
    every decade and the labelled set here does not: 5 is labelled in one
    decade and 1, 2 and 5 in the next. A subs tuple would therefore draw a
    minor tick underneath a major one -- invisible, but present in the tick
    list and wrong if anyone reads it back.
    """
    v = np.asarray([x for x in vals if np.isfinite(x) and x > 0], dtype=float)
    span = float(v.max() / v.min()) if v.size else 1.0
    use_log = span >= 8.0 if force == "auto" else (force == "log")
    if use_log:
        ax.set_yscale("log")
        if majors:
            maj = sorted(float(m) for m in majors)
            ax.yaxis.set_major_locator(FixedLocator(maj))
            lo, hi = ax.get_ylim()
            grid = [k * 10.0 ** n for n in range(-9, 4) for k in range(1, 10)]
            mino = [t for t in grid if lo <= t <= hi
                    and not any(abs(t - m) <= 1e-9 * max(t, m) for m in maj)]
            ax.yaxis.set_minor_locator(FixedLocator(sorted(mino)))
            print(f"  y major ticks (labelled): "
                  + ", ".join(f"{m:g}" for m in maj)
                  + f"\n  y minor ticks (unlabelled): "
                  + ", ".join(f"{m:g}" for m in sorted(mino)))
            outside = [m for m in maj if not (lo <= m <= hi)]
            if outside:
                print(f"  NOTE: {', '.join(f'{m:g}' for m in outside)} lie "
                      f"outside the view [{lo:.4g}, {hi:.4g}] and are not "
                      f"drawn.")
        else:
            ax.yaxis.set_major_locator(LogLocator(base=10.0,
                                                  subs=(1.0, 2.0, 3.0, 5.0),
                                                  numticks=16))
            ax.yaxis.set_minor_locator(LogLocator(
                base=10.0, subs=tuple(np.arange(2, 10) * 0.1), numticks=64))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda x, _: (f"{x:g}" if 1e-3 <= abs(x) < 1e4
                          else rf"$10^{{{int(round(np.log10(abs(x))))}}}$")))
    else:
        ax.set_yscale("linear")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    print(f"  vertical scale: {'log' if use_log else 'linear'}"
          f"{'' if force == 'auto' else ' (forced)'} "
          f"(largest/smallest plotted value {span:.2f})")
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repdir", default="reports_val")
    ap.add_argument("--datadir", default="data",
                    help="only to read ||p_target|| for the printed "
                         "unnormalised column; the figure does not need it")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--nfs", default="0,1,2,3", help="file-name convention")
    ap.add_argument("--lb", choices=("on", "off"), default="on")
    ap.add_argument("--din", type=int, default=2)
    ap.add_argument("--key", default="JOINT",
                    choices=("JOINT", "PROJ_HOLDOUT"),
                    help="which out-of-objective diagnostic to plot")
    ap.add_argument("--quantity", default="error", choices=("error", "loss"),
                    help="plot the stored relative error, or its square")
    ap.add_argument("--yticks", default="",
                    help="comma-separated labelled y ticks, e.g. "
                         "0.05,0.1,0.2,0.5; empty keeps the automatic set")
    ap.add_argument("--yscale", default="auto",
                    choices=("auto", "log", "linear"),
                    help="auto picks log only when the span justifies it")
    ap.add_argument("--name", default=None,
                    help="output file name; default follows --key")
    ap.add_argument("--out", default=None, help="optional json dump")
    args = ap.parse_args()

    style()
    os.makedirs(args.outdir, exist_ok=True)
    nfs = [int(v) for v in args.nfs.split(",")]
    KEY = args.key
    # The label, the output name and the reference-archive entry that holds the
    # target this diagnostic is normalised by. Keyed off --key in one place, so
    # a third diagnostic needs one line and not a second copy of the script.
    ERR = args.quantity == "error"
    LABEL = ({"JOINT": r"$e_{\mathrm{JOINT}}$",
              "PROJ_HOLDOUT": r"$e_{\mathrm{PROJ_{hold}}}$"} if ERR else
             {"JOINT": r"$\mathcal{L}_{\mathrm{JOINT}}$",
              "PROJ_HOLDOUT": r"$\mathcal{L}_{\mathrm{PROJ_{hold}}}$"})[KEY]
    SYM = ("e_" if ERR else "L_") + KEY
    FNAME = args.name or {"JOINT": "fig_joint_loss.pdf",
                          "PROJ_HOLDOUT": "fig_projhold_loss.pdf"}[KEY]
    REFKEY = {"JOINT": "joint", "PROJ_HOLDOUT": "projected_holdout"}[KEY]

    # ---- per-run losses, squared at the point of reading --------------------
    L, anchors, ngen = {}, set(), set()
    for p in sorted(glob.glob(os.path.join(args.repdir, "report_*.json"))):
        m = PAT.search(os.path.basename(p))
        if not m:
            continue
        nf, lb, din = int(m.group(1)), m.group(2), int(m.group(3))
        if nf not in nfs or lb != args.lb or din != args.din:
            continue
        r = json.load(open(p))
        lam = float(m.group(4).replace("p", "."))
        ps = r.get("per_seed")
        if not ps:
            raise SystemExit(f"{p} has no per_seed block; the per-run losses "
                             f"cannot be formed and the medians from the "
                             f"summary block are the wrong statistic.")
        vals = []
        for s in ps:
            d = s.get("diagnostics", {})
            if KEY not in d:
                raise SystemExit(f"{p}: seed {s.get('seed')} carries no "
                                 f"{KEY} diagnostic.")
            # PER RUN, BEFORE ANY AGGREGATION. The stored value is the
            # relative discrepancy; it is squared here only when the loss was
            # asked for, never afterwards and never on an aggregate.
            v = float(d[KEY])
            vals.append(v if ERR else v ** 2)
        L[(lam, nf)] = np.asarray(vals, dtype=float)
        anchors.add(r.get("anchor", "evaluation"))
        ngen.add(int(r.get("n_gen", -1)))

    if not L:
        raise SystemExit(f"no matching report_*.json in {args.repdir}")
    lams = sorted({k[0] for k in L})
    missing = [(l, nf) for l in lams for nf in nfs if (l, nf) not in L]
    if missing:
        raise SystemExit(f"incomplete grid; missing {missing}")
    if len(anchors) > 1:
        raise SystemExit(f"{args.repdir} mixes anchors ({sorted(anchors)}); "
                         f"regenerate with one --against setting.")
    if len(ngen) > 1:
        raise SystemExit(f"{args.repdir} mixes ensemble sizes {sorted(ngen)}; "
                         f"a loss is not comparable across them.")

    med = {k: float(np.median(v)) for k, v in L.items()}
    q1 = {k: float(np.percentile(v, 25)) for k, v in L.items()}
    q3 = {k: float(np.percentile(v, 75)) for k, v in L.items()}

    # ---- the figure ---------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(WIDTH * 0.5, 2.2))
    x = np.arange(len(lams))
    cols = ramp(len(nfs))

    # DODGED, BECAUSE THE BARS WOULD OTHERWISE SIT ON TOP OF EACH OTHER. Four
    # series share every x position and their interquartile bars overlap at
    # several couplings; a small horizontal offset separates them without
    # moving any point far enough to be misread as a different coupling. The
    # ticks stay on the integers, which are the actual lambda values.
    off = (np.arange(len(nfs)) - (len(nfs) - 1) / 2.0) * 0.085

    for c, mk, nf, dx in zip(cols, MARKS, nfs, off):
        m = np.array([med[(l, nf)] for l in lams])
        lo = m - np.array([q1[(l, nf)] for l in lams])
        hi = np.array([q3[(l, nf)] for l in lams]) - m
        ax.errorbar(x + dx, m, yerr=np.vstack([lo, hi]), fmt=mk + "-",
                    ms=3.4, lw=1.1, color=c, ecolor=c, elinewidth=0.7,
                    capsize=1.8, capthick=0.7, zorder=3,
                    label=rf"$n_{{\mathrm{{f}}}}={nf + 1}$")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l:g}" for l in lams])
    ax.set_xlabel(r"coupling strength $\lambda$")
    # The name, not the formula: the caption carries the definition, and
    # "loss" is what distinguishes this from the ratio it replaces.
    ax.set_ylabel(LABEL)

    # Log, because a squared error spans more than the ratio did and the
    # interesting part -- whether the last frame buys anything -- is at the
    # bottom. No reference line: without the reference-block denominator there
    # is no value on this axis that means "as close as the data is to itself".
    log_ticks(ax, [q1[(l, nf)] for l in lams for nf in nfs]
               + [q3[(l, nf)] for l in lams for nf in nfs], force=args.yscale,
              majors=([float(v) for v in args.yticks.split(",")]
                      if args.yticks else None))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(nfs),
              handletextpad=0.5, columnspacing=1.3, fontsize=7.0,
              borderaxespad=0.15)
    fig.tight_layout(pad=0.4)
    path = os.path.join(args.outdir, FNAME)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")

    # ---- the numbers --------------------------------------------------------
    n_per = {k: int(v.size) for k, v in L.items()}
    print(f"\n{SYM}, per run"
          f"{', squared before aggregation' if not ERR else ' as stored'}. "
          f"n_f is the manuscript's count\n(file label + 1). anchor "
          f"{sorted(anchors)[0]}, n_gen {sorted(ngen)[0]}, d_in={args.din}, "
          f"balancing {args.lb}.\nruns per cell: "
          f"{sorted(set(n_per.values()))}")
    print(f"\n{'lam':<6}" + "".join(f"{'n_f=' + str(nf + 1):<26}" for nf in nfs))
    print(f"{'':<6}" + "".join(f"{'median [Q1, Q3]':<26}" for _ in nfs))
    for l in lams:
        row = f"{l:<6g}"
        for nf in nfs:
            k = (l, nf)
            row += (f"{med[k]:.4f} [{q1[k]:.4f}, {q3[k]:.4f}]"
                    .ljust(26))
        print(row)

    # The two claims the old ratio figure carried, re-checked on the loss.
    # Monotonicity is preserved by squaring -- it is a monotone map on
    # non-negative numbers -- but the MEDIANS are not the squares of the old
    # medians, so it is re-tested on the numbers actually plotted rather than
    # inherited.
    bad = [l for l in lams
           if any(med[(l, nfs[i + 1])] > med[(l, nfs[i])]
                  for i in range(len(nfs) - 1))]
    print(f"\nmonotone in n_f: "
          + ("at every coupling" if not bad else
             "NO -- fails at lambda = " + ", ".join(f"{v:g}" for v in bad)))
    print(f"last-frame gain on {SYM}, as a factor med(n_f=3)/med(n_f=4): "
          + "  ".join(f"lam {l:g}: {med[(l, nfs[-2])] / med[(l, nfs[-1])]:.2f}"
                      for l in lams))

    # ---- what the remaining normaliser is worth -----------------------------
    #
    # Printed, never plotted. e_JOINT is relative to ||p_target||, which
    # depends on lambda, so a trend across couplings could in principle be the
    # normaliser moving rather than the model. This prints the normaliser and
    # the unnormalised loss so that possibility can be checked instead of
    # assumed away. Absent archive is not an error: the figure does not use it.
    rows = []
    for l in lams:
        tag = f"lam{l:g}".replace(".", "p")
        p = os.path.join(args.datadir, f"reference_{tag}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=False)
        if REFKEY not in d.files:
            continue
        nrm = float(np.sqrt(np.mean(np.asarray(d[REFKEY],
                                                 dtype=float) ** 2)))
        rows.append((l, nrm))
    if rows:
        print(f"\n||p_target||_rms per coupling, and the unnormalised "
              f"quantity\nat n_f = {nfs[-1] + 1} (medians):")
        print(f"{'lam':<6}{'||p_target||':>14}{'relative':>15}"
              f"{'unnormalised':>19}")
        for l, nrm in rows:
            m = med[(l, nfs[-1])]
            print(f"{l:<6g}{nrm:>14.6f}{m:>15.6f}"
                  f"{m * (nrm if ERR else nrm ** 2):>19.3e}")
    else:
        print(f"\nno reference_lam*.npz under {args.datadir}, so the "
              f"normaliser is not shown.\nThe figure does not depend on it; "
              f"the across-lambda comparison does.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({
                "quantity": (f"{SYM} per run, e = "
                             "||p_model - p_target|| / ||p_target|| (RMS "
                             "norms); NOT divided by any reference-block "
                             "value. Aggregated as plotted, not derived from "
                             "the other power."),
                "basis": ("regenerated ensemble scored by report.py, not the "
                          "retained-checkpoint validation record"),
                "anchor": sorted(anchors)[0], "n_gen": sorted(ngen)[0],
                "d_in": args.din, "loss_balancing": args.lb,
                "cells": {f"lam{l:g}_nf{nf + 1}":
                          {"median": med[(l, nf)], "q1": q1[(l, nf)],
                           "q3": q3[(l, nf)], "n": n_per[(l, nf)],
                           "per_run": L[(l, nf)].tolist()}
                          for l in lams for nf in nfs},
                "target_norm": {f"lam{l:g}": v for l, v in rows},
            }, fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
