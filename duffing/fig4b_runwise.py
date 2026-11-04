#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 4(b), candidate: the family spread measured INSIDE each run.

WHAT IS DIFFERENT, AND WHY IT IS A DIFFERENT QUANTITY. The published panel
plots

    max_s med_runs r_s / min_s med_runs r_s ,

the ratio of two across-run medians. Its numerator and denominator come from
different families and, in general, from different runs, so it describes how far
apart the TYPICAL family error levels are. It says nothing about whether the
families inside any one run are close together. This candidate reverses the
order:

    per run and per step   spread_run = max_s r_s / min_s r_s
    across runs            median, with the interquartile band

and that is a statement about one run at a time. The two are not rescalings of
each other: within-run scatter enters every per-run ratio before the median is
taken, so this quantity sits above the published one by an amount that depends
on how noisy each setting is. They can order differently, and have: the earlier
version of this panel, computed this way, crossed in mid-training in the
opposite order to the appendix table.

THE BAND IS LEGITIMATE HERE AND IS NOT ON THE PUBLISHED PANEL. There, each
plotted value is already a summary over runs, so there is no per-run value to
spread. Here each run contributes its own curve, so quartiles across runs mean
something.

NOTHING IS RECOMPUTED FROM CHECKPOINTS. The histories, the run set and the
reference losses are read through fig_valloss.py's own loader and floors(), so
this figure and the published one rest on the same numbers by construction
rather than by agreement.

NO VERDICT IS BUILT IN. The axis is set from the data so that both bands fit
whole; crossings are drawn as they fall. The summary printed at the end reports
where the two medians sit and where they cross, without calling either setting
better.

Usage
-----
    python fig4b_runwise.py --rundir runs/grid --outdir figs
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# The published panel's own machinery: same reader, same reference losses,
# same colours, same rcParams. Importing beats re-implementing -- a second
# copy of floors() is a second thing that can drift.
from fig_valloss import WIDTH, C_ON, C_OFF, style, load, floors

PROTECTED = {"fig4_lam4.pdf", "fig4_spread_lam4.pdf", "fig4_total_lam4.pdf",
             "fig4_families_lam4.pdf", "fig4_weights_run_lam4.pdf"}


def runwise(runs, fl):
    """One curve per run: max_s r_s / min_s r_s at every validation step."""
    grid = runs[0]["iter"]
    fams = [f for f in runs[0]["fams"] if f in fl and fl[f] > 0]
    out = []
    for r in runs:
        if r["iter"].shape != grid.shape or not np.allclose(r["iter"], grid):
            raise SystemExit(f"{r['tag']} has a different validation grid; "
                             f"the runs cannot be pooled step by step.")
        # THE MAXIMUM AND THE MINIMUM COME FROM THE SAME RUN AND THE SAME STEP.
        # That is the whole point of this figure, so it is done before any
        # aggregation touches the values.
        R = np.vstack([np.asarray(r["L"][f]) / fl[f] ** 2 for f in fams])
        out.append(R.max(axis=0) / R.min(axis=0))
    return grid, np.vstack(out), fams


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rundir", default="runs/grid")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--nf", type=int, default=3, help="file-name convention")
    ap.add_argument("--din", type=int, default=2)
    ap.add_argument("--lam", type=float, default=4.0)
    ap.add_argument("--stem", default="fig4b_runwise_spread_candidate")
    args = ap.parse_args()

    style()
    os.makedirs(args.outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        if f"{args.stem}.{ext}" in PROTECTED:
            raise SystemExit(f"{args.stem}.{ext} is a published figure; this "
                             f"script only writes a candidate")

    on = load(args.rundir, args.nf, args.din, args.lam, "on")
    off = load(args.rundir, args.nf, args.din, args.lam, "off")
    if not on or not off:
        raise SystemExit(f"need both balancing settings at lambda="
                         f"{args.lam:g} in {args.rundir}; found {len(on)} on, "
                         f"{len(off)} off")
    print(f"lambda = {args.lam:g}, n_f = {args.nf + 1} (file nf{args.nf}), "
          f"d_in = {args.din}")
    print(f"runs: {len(off)} balancing off, {len(on)} balancing on")

    # One coupling, one set of reference losses, from both settings together --
    # exactly as the published panel does it.
    fl = floors(on + off)

    grid, M_off, fams = runwise(off, fl)
    _, M_on, fams_on = runwise(on, fl)
    if fams != fams_on:
        raise SystemExit(f"the two settings use different families "
                         f"({fams} vs {fams_on}); the spreads are not "
                         f"comparable.")
    print(f"families in the spread: {fams}")

    stat = {}
    for lab, M in (("off", M_off), ("on", M_on)):
        stat[lab] = (np.median(M, axis=0),
                     np.percentile(M, 25, axis=0),
                     np.percentile(M, 75, axis=0))

    fig, ax = plt.subplots(1, 1, figsize=(WIDTH * 0.46, 2.2))
    for lab, name, c in (("off", "balancing off", C_OFF),
                         ("on", "balancing on", C_ON)):
        med, q1, q3 = stat[lab]
        ax.fill_between(grid, q1, q3, color=c, alpha=0.16, lw=0)
        ax.plot(grid, med, "-", color=c, label=name)

    ax.set_yscale("log")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Run-level family spread")
    ax.xaxis.set_major_locator(MultipleLocator(5000))
    ax.xaxis.set_minor_locator(MultipleLocator(2500))
    ax.legend(loc="upper right", handletextpad=0.5, labelspacing=0.25)

    # THE WHOLE OF BOTH BANDS, NOT THE MEDIANS' RANGE. Clipping a quartile
    # would hide exactly the disagreement this figure exists to show.
    lo = min(stat[l][1].min() for l in ("off", "on"))
    hi = max(stat[l][2].max() for l in ("off", "on"))
    ax.set_ylim(lo / 1.25, hi * 1.25)

    fig.tight_layout(pad=0.4)
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(args.outdir, f"{args.stem}.{ext}")
        fig.savefig(p, dpi=300 if ext == "png" else None)
        paths.append(os.path.abspath(p))
        print(f"wrote {os.path.abspath(p)}")
    plt.close(fig)

    # ---- what the curves do, stated without a verdict ----------------------
    mo, mn = stat["off"][0], stat["on"][0]
    frac = float(np.mean(mn < mo))
    below = mn < mo
    i = len(below)
    while i > 0 and below[i - 1]:
        i -= 1
    first_persistent = float(grid[i]) if i < len(below) else float("nan")
    print(f"\n{'='*70}\nmedian run-level spread\n{'='*70}")
    print(f"{'iteration':<12}{'off':>12}{'on':>12}{'on/off':>10}")
    for j in (0, len(grid) // 4, len(grid) // 2, 3 * len(grid) // 4, -1):
        print(f"{grid[j]:<12.0f}{mo[j]:>12.3f}{mn[j]:>12.3f}"
              f"{mn[j] / mo[j]:>10.3f}")
    print(f"\non below off at {100 * frac:.0f}% of the {len(grid)} validation "
          f"steps")
    print(f"on stays below off from iteration "
          + (f"{first_persistent:.0f} onwards" if np.isfinite(first_persistent)
             else "never (it is above at the last step)"))
    print(f"\nlast step: off {mo[-1]:.3f} [{stat['off'][1][-1]:.3f}, "
          f"{stat['off'][2][-1]:.3f}]   on {mn[-1]:.3f} "
          f"[{stat['on'][1][-1]:.3f}, {stat['on'][2][-1]:.3f}]")
    ov = (stat["on"][1][-1] <= stat["off"][2][-1]
          and stat["off"][1][-1] <= stat["on"][2][-1])
    print(f"the two interquartile ranges at the last step "
          f"{'overlap' if ov else 'do not overlap'}")
    print(f"\nFor comparison, the published panel plots the ratio of across-run "
          f"medians,\nwhich is a different statistic and is NOT recoverable "
          f"from the numbers above.")


if __name__ == "__main__":
    main()
