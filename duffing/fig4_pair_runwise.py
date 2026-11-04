#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 4 candidate: (a) the adaptive weights, (b) the RUN-LEVEL family spread.

Panel (a) is the published one, unchanged and not re-implemented -- it is
fig_valloss.panel_weights, called here. Panel (b) is the candidate from
fig4b_runwise.py: the spread formed inside each run and then summarised across
runs, rather than the ratio of two across-run medians.

WHY THE TWO PANELS STILL MAY NOT BE READ AS A SEQUENCE OF EVENTS. The
published figure carries the same warning and it still applies: which comes
first, the weights settling or the spread separating, is decided by the
tolerance one picks for "settled", not by the data (timing_lb.py measured both
on twenty matched pairs). Putting them on one axis lets a reader compare the
two time courses; it does not license a causal reading.

THE PUBLISHED FIGURE IS NOT TOUCHED. This writes fig4_lam4_runwise_candidate
.pdf and .png and refuses the published names.

Usage
-----
    python fig4_pair_runwise.py --rundir runs/grid --repdir reports_val \\
                                --outdir figs
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from fig_valloss import (WIDTH, C_ON, C_OFF, style, load, floors,
                         panel_weights)
from fig4b_runwise import runwise, PROTECTED


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rundir", default="runs/grid")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--nf", type=int, default=3, help="file-name convention")
    ap.add_argument("--din", type=int, default=2)
    ap.add_argument("--lam", type=float, default=4.0)
    ap.add_argument("--stem", default="fig4_lam4_runwise_candidate")
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

    fl = floors(on + off)
    grid, M_off, fams = runwise(off, fl)
    _, M_on, fams_on = runwise(on, fl)
    if fams != fams_on:
        raise SystemExit(f"the two settings use different families "
                         f"({fams} vs {fams_on}).")
    print(f"families in the spread: {fams}")

    stat = {lab: (np.median(M, axis=0), np.percentile(M, 25, axis=0),
                  np.percentile(M, 75, axis=0))
            for lab, M in (("off", M_off), ("on", M_on))}

    # Same canvas as the published two-panel figure.
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH * 0.68, 2.15))
    panel_weights(axes[0], on)

    ax = axes[1]
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
    lo = min(stat[l][1].min() for l in ("off", "on"))
    hi = max(stat[l][2].max() for l in ("off", "on"))
    # Headroom for the panel label, which sits in the upper-left corner where
    # both curves begin.
    ax.set_ylim(lo / 1.25, hi * 2.0)

    for i, a in enumerate(axes):
        a.text(0.035, 0.965, f"({chr(97 + i)})", transform=a.transAxes,
               fontsize=8.5, ha="left", va="top")

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        p = os.path.join(args.outdir, f"{args.stem}.{ext}")
        fig.savefig(p, dpi=300 if ext == "png" else None)
        print(f"wrote {os.path.abspath(p)}")
    plt.close(fig)

    mo, mn = stat["off"][0], stat["on"][0]
    below = mn < mo
    i = len(below)
    while i > 0 and below[i - 1]:
        i -= 1
    first = float(grid[i]) if i < len(below) else float("nan")
    print(f"\n(b) median run-level spread: last step off {mo[-1]:.3f} "
          f"[{stat['off'][1][-1]:.3f}, {stat['off'][2][-1]:.3f}], on "
          f"{mn[-1]:.3f} [{stat['on'][1][-1]:.3f}, {stat['on'][2][-1]:.3f}]")
    print(f"    on below off at {100 * np.mean(below):.0f}% of "
          f"{len(grid)} steps; persistently from "
          + (f"{first:.0f}" if np.isfinite(first) else "never"))
    print("    (a) is the published weights panel, unchanged.")


if __name__ == "__main__":
    main()
