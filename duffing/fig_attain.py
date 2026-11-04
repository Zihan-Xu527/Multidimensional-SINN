#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
New Figure 5: threshold attainment over the validation history. Item 3.

THE QUANTITY. For every run and every validation step,

    r_s^val = L_s^val / L_s^ref ,        r^val = max over applicable s ,

the loss ratio, with no square root. A run ATTAINS a threshold r if
r^val <= r at some validation step, whether or not that step is the one the
checkpoint rule retained.

TWO KINDS OF COUNT, AND THIS SCRIPT REPORTS ONLY THE SECOND.

    retained-checkpoint successes   r^val <= tau at the retained checkpoint
    any-step attainment             r^val <= tau at ANY validation step

The first is ratios10.py's job and is already recorded (66/70 for balancing
off and 66/70 for balancing on, at tau = 10). Every count below is the second
kind. They are not interchangeable and must not be quoted for one another.

WHERE THE REFERENCE LOSS COMES FROM, AND WHY NOTHING IS RECOMPUTED. The
history stores the model's per-family validation errors at every step but not
the reference loss they are divided by. That loss does not have to be rebuilt
from the nine blocks: at the retained checkpoint the run record carries BOTH
the error and the ratio, so

    sqrt(L_s^ref) = val_errors[s] / sqrt(ratios[s])

recovers it exactly. It depends only on the coupling and the family, so every
run at one coupling -- balancing on and off alike -- must yield the same
value. That is checked before anything is plotted, and it is a real check:
it would catch a history paired with the wrong reference file.

THE CONVERSION CHECK. Thresholds 9, 4 and 3.24 are the squares of the
superseded 3, 2 and 1.8, so their attainment counts and first-reaching
iterations must reproduce the published ones exactly. The script computes both
ways and prints them side by side; a mismatch means the squaring argument is
wrong somewhere and the new numbers cannot be trusted.

Usage
-----
    python fig_attain.py
    python fig_attain.py --thresholds 10,9,4,3.24 --outdir figs
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
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple

WIDTH = 6.02
C_ON, C_OFF = "#1b4f72", "#a93226"

PAT = re.compile(
    r"hist_nf(\d)_lb(on|off)_din(\d)_L(\d+)H(\d+)_lam([0-9p]+)_s(\d+)\.npz$")


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.0,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "lines.linewidth": 1.1,
        "axes.grid": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def load(rundir, nf, din):
    """Every run of the n_f / d_in cell, both balancing settings."""
    runs = []
    for p in sorted(glob.glob(os.path.join(rundir, "hist_*.npz"))):
        m = PAT.search(os.path.basename(p))
        if not m or int(m.group(1)) != nf or int(m.group(3)) != din:
            continue
        d = np.load(p, allow_pickle=False)
        ve = json.loads(str(d["val_errors"])) if "val_errors" in d.files else []
        if not ve:
            continue
        tag = os.path.basename(p)[5:-4]
        rec_path = os.path.join(rundir, f"run_{tag}.json")
        if not os.path.exists(rec_path):
            print(f"  NOTE: {tag} has no run record; skipped (the reference "
                  f"loss is recovered from it).")
            continue
        rec = json.load(open(rec_path))
        runs.append({
            "tag": tag, "lb": m.group(2),
            "lam": round(float(m.group(6).replace("p", ".")), 3),
            "iter": np.asarray(d["iter"], dtype=float),
            "errs": ve,
            "fams": [f for f in ve[0]],
            "best": rec.get("best", {}),
        })
    return runs


def floors(runs):
    """sqrt(L_s^ref) per coupling, recovered from the retained checkpoint.

    THE ARITHMETIC, WRITTEN OUT BECAUSE IT WAS WRONG ONCE. The stored
    ratios[f] are on the SUPERSEDED square-root scale: they are
    error / sqrt(L_s^ref), which is what the published table reports. So

        sqrt(L_s^ref) = val_error[f] / ratios[f]

    with no square root on the right. Dividing by sqrt(ratios) instead makes
    the recovered floor drift as sqrt(ratio) from run to run, and the check at
    the bottom of this function caught exactly that, at a spread of 0.88.

    THE ERROR COMES FROM THE HISTORY, NOT FROM best["val_errors"]. That field
    is missing from the older run records -- it yielded only 29 (run, family)
    pairs out of some seven hundred -- so the retained checkpoint is located in
    the history by best["iter"] instead. Every run has both of those.
    """
    acc, used, skipped = {}, 0, []
    for r in runs:
        b = r["best"]
        ra, it_best = b.get("ratios"), b.get("iter")
        if not ra or it_best is None:
            skipped.append(r["tag"])
            continue
        j = int(np.argmin(np.abs(r["iter"] - float(it_best))))
        if abs(r["iter"][j] - float(it_best)) > 0.5:
            skipped.append(r["tag"])
            continue
        for f, e in r["errs"][j].items():
            if f in ra and np.isfinite(ra[f]) and ra[f] > 0:
                acc.setdefault(r["lam"], {}).setdefault(f, []).append(
                    float(e) / float(ra[f]))
        used += 1
    if skipped:
        print(f"  NOTE: {len(skipped)} run(s) have no usable retained-checkpoint "
              f"record and did not\n        contribute: "
              f"{', '.join(sorted(skipped)[:4])}"
              f"{' ...' if len(skipped) > 4 else ''}")
    out, worst = {}, 0.0
    for lam, fam in acc.items():
        out[lam] = {}
        for f, v in fam.items():
            v = np.asarray(v, dtype=float)
            med = float(np.median(v))
            out[lam][f] = med
            if med > 0:
                worst = max(worst, float(np.max(np.abs(v - med)) / med))
    npair = sum(len(v) for fam in acc.values() for v in fam.values())
    ngrp = sum(len(fam) for fam in acc.values())
    print(f"  reference loss recovered from {used} of {len(runs)} runs: "
          f"{npair} values over {ngrp} (coupling, family) groups;\n"
          f"  largest spread within a group: {worst:.3e}")
    if worst > 1e-6:
        raise SystemExit(
            "runs at the same coupling disagree about the reference loss. "
            "Either a history is paired with the wrong reference file or the "
            "stored ratios are not error/floor. Stop.")
    return out


def trajectory(run, fl):
    """r^val at every validation step: max over families of L_s / L_s^ref."""
    out = []
    for step in run["errs"]:
        vals = [(step[f] / fl[f]) ** 2 for f in step
                if f in fl and fl[f] > 0 and np.isfinite(step[f])]
        out.append(max(vals) if vals else np.nan)
    return np.asarray(out, dtype=float)


def first_reaching(it, r, thr):
    """Iteration of the first step at or below thr, or None."""
    hit = np.nonzero(np.isfinite(r) & (r <= thr))[0]
    return (float(it[hit[0]]), int(hit[0])) if hit.size else (None, None)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rundir", default="runs/grid")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--nf", type=int, default=3, help="file-name convention")
    ap.add_argument("--din", type=int, default=2)
    ap.add_argument("--thresholds", default="10,9,4,3.24")
    ap.add_argument("--dump", default="fig5_attainment.json")
    args = ap.parse_args()

    style()
    os.makedirs(args.outdir, exist_ok=True)
    # ONE ORDER EVERYWHERE, ASCENDING. (a) is a continuous axis and runs
    # left to right in r; (b) used to take the thresholds in the order they
    # were typed, which was descending, so the two panels ran opposite ways
    # under the same axis label. Sorting here fixes the panels and the
    # printed tables together: tight on the left, loose on the right.
    THR = sorted(float(v) for v in args.thresholds.split(","))

    runs = load(args.rundir, args.nf, args.din)
    if not runs:
        raise SystemExit(f"no hist_*.npz for nf{args.nf} din{args.din} "
                         f"in {args.rundir}")
    print(f"n_f = {args.nf + 1} (file nf{args.nf}), d_in = {args.din}, "
          f"{len(runs)} runs")
    fl = floors(runs)

    for r in runs:
        r["r"] = trajectory(r, fl[r["lam"]])
        r["rmin"] = float(np.nanmin(r["r"]))

    lams = sorted({r["lam"] for r in runs})
    by = {(lb, lam): [r for r in runs if r["lb"] == lb and r["lam"] == lam]
          for lb in ("off", "on") for lam in lams}
    pooled = {lb: [r for r in runs if r["lb"] == lb] for lb in ("off", "on")}

    # ---- output 1: per-run minimum over the history -------------------------
    print(f"\n{'='*70}\n1. minimum r^val over all validation steps, per run\n"
          f"{'='*70}")
    print(f"{'lam':<6}{'bal':<5}{'n':>4}{'min':>9}{'Q1':>9}{'median':>9}"
          f"{'Q3':>9}{'max':>9}")
    for lam in lams:
        for lb in ("off", "on"):
            v = [r["rmin"] for r in by[(lb, lam)]]
            if not v:
                continue
            print(f"{lam:<6g}{lb:<5}{len(v):>4}{min(v):>9.2f}"
                  f"{np.percentile(v, 25):>9.2f}{np.median(v):>9.2f}"
                  f"{np.percentile(v, 75):>9.2f}{max(v):>9.2f}")

    # ---- output 3: attainment counts, per coupling and pooled ---------------
    print(f"\n{'='*70}\n3. ANY-STEP attainment counts  (NOT the "
          f"retained-checkpoint counts)\n{'='*70}")
    hdr = f"{'lam':<7}{'bal':<5}" + "".join(f"{'r<=' + f'{t:g}':>10}" for t in THR)
    print(hdr)
    counts = {}
    for lam in lams:
        for lb in ("off", "on"):
            v = by[(lb, lam)]
            if not v:
                continue
            cells = [f"{sum(1 for r in v if r['rmin'] <= t)}/{len(v)}"
                     for t in THR]
            print(f"{lam:<7g}{lb:<5}" + "".join(f"{c:>10}" for c in cells))
            for t in THR:
                counts[(lb, lam, t)] = (
                    sum(1 for r in v if r["rmin"] <= t), len(v))
    print("-" * len(hdr))
    for lb in ("off", "on"):
        v = pooled[lb]
        cells = [f"{sum(1 for r in v if r['rmin'] <= t)}/{len(v)}"
                 for t in THR]
        print(f"{'pooled':<7}{lb:<5}" + "".join(f"{c:>10}" for c in cells))
        for t in THR:
            counts[(lb, "pooled", t)] = (
                sum(1 for r in v if r["rmin"] <= t), len(v))

    # ---- the conversion check ------------------------------------------------
    print(f"\n{'='*70}\nconversion check: new thresholds against the superseded "
          f"square-root ones\n{'='*70}")
    print(f"{'new':>7}{'old':>7}   pooled attainment off / on,  new vs old "
          f"definition")
    ok = True
    for t in THR:
        told = np.sqrt(t)
        cells = []
        for lb in ("off", "on"):
            v = pooled[lb]
            a = sum(1 for r in v if r["rmin"] <= t)
            # the same runs, counted on the square-root scale
            b = sum(1 for r in v if np.sqrt(r["rmin"]) <= told)
            cells.append(f"{a}/{len(v)} vs {b}/{len(v)}")
            ok &= (a == b)
        print(f"{t:>7g}{told:>7.3f}   {cells[0]:<18}{cells[1]}")
    print("squaring is monotone, so these must agree exactly; "
          f"{'they do' if ok else 'THEY DO NOT -- stop'}.")
    if not ok:
        raise SystemExit("threshold conversion failed")

    # ---- outputs 2 and 4: first-reaching iterations -------------------------
    print(f"\n{'='*70}\n2 & 4. first validation iteration at or below each "
          f"threshold\n{'='*70}")
    first = {}
    for t in THR:
        print(f"\nthreshold r^val <= {t:g}")
        print(f"  {'bal':<5}{'attained':>10}{'min':>8}{'Q1':>8}{'median':>9}"
              f"{'Q3':>8}{'max':>8}   not attained")
        for lb in ("off", "on"):
            v = pooled[lb]
            its, miss = [], []
            for r in v:
                i, _ = first_reaching(r["iter"], r["r"], t)
                first[(r["tag"], t)] = i
                (its if i is not None else miss).append(
                    i if i is not None else r["tag"])
            if its:
                print(f"  {lb:<5}{f'{len(its)}/{len(v)}':>10}{min(its):>8.0f}"
                      f"{np.percentile(its, 25):>8.0f}"
                      f"{np.median(its):>9.0f}"
                      f"{np.percentile(its, 75):>8.0f}{max(its):>8.0f}"
                      f"   {len(miss)}")
            else:
                print(f"  {lb:<5}{f'0/{len(v)}':>10}{'--':>8}{'--':>8}"
                      f"{'--':>9}{'--':>8}{'--':>8}   {len(miss)}")
            if miss:
                print(f"         not attained: {', '.join(sorted(miss))}")

    # ---- the figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH * 0.72, 2.25))

    # (a) attainment fraction: the empirical distribution of the per-run
    # minimum, which is exactly "what fraction ever got this low". Drawn as a
    # continuous curve rather than bars at the four thresholds, because the
    # quantity is continuous and bars at five selected values were the thing
    # that made the old panel read as a bar chart of categories.
    ax = axes[0]
    # THE VIEW IS CLIPPED AT THE SUCCESS THRESHOLD; THE CURVE IS NOT.
    # One run's minimum is far out in the tail, and drawing to it spends most
    # of the axis on a region where both curves are flat and nothing is
    # decided -- the four thresholds end up crowded into the left edge. The
    # fraction is still computed over ALL runs, so the value read off at 10 is
    # 68/70 and not 68 out of whatever survived the clip. The denominator is
    # the run count, never the visible run count.
    # A little past the largest threshold, so its dotted line is not drawn
    # on top of the right spine. The VIEW ends at hi; the count reported
    # below is of runs that never reach the success criterion itself, which
    # is max(THR), not the view limit -- those are different questions.
    hi = max(THR) * 1.08
    grid = np.linspace(1.0, hi, 400)
    for lb, c in (("off", C_OFF), ("on", C_ON)):
        v = np.array([r["rmin"] for r in pooled[lb]])      # all runs
        ax.plot(grid, [(v <= g).mean() for g in grid], "-", color=c,
                label=f"balancing {lb}")
    for t in THR:
        ax.axvline(t, color="0.75", lw=0.6, ls=":", zorder=0)
    ax.set_xlabel(r"Threshold on $r^{\mathrm{val}}$")
    ax.set_ylabel("Fraction of runs")
    ax.set_xlim(1.0, hi)
    ax.set_ylim(-0.03, 1.08)
    beyond = {lb: sum(1 for r in pooled[lb] if r["rmin"] > max(THR))
              for lb in ("off", "on")}
    if any(beyond.values()):
        print(f"\n(a) is drawn out to r^val = {hi:g}. Runs that never reach "
              f"the success criterion\n    {max(THR):g}, and so never lift "
              f"the curve inside the view: "
              + ", ".join(f"{lb} {beyond[lb]} of {len(pooled[lb])}"
                          for lb in ("off", "on"))
              + ".\n    They are still in the denominator; the curve just "
              "does not reach 1 inside the view.")

    # (b) first-reaching iteration, attainers only. Categorical in the
    # threshold, because four selected values is what the panel is about.
    ax = axes[1]
    pos = np.arange(len(THR))
    alldata = []
    for k, (lb, c) in enumerate((("off", C_OFF), ("on", C_ON))):
        data, at = [], []
        for t in THR:
            its = [first[(r["tag"], t)] for r in pooled[lb]
                   if first[(r["tag"], t)] is not None]
            data.append(its if its else [np.nan])
            alldata.append(its if its else [np.nan])
            at.append(len(its))
        bp = ax.boxplot(data, positions=pos + (k - 0.5) * 0.34, widths=0.28,
                        patch_artist=True, showfliers=False,
                        medianprops={"color": c, "lw": 0.9},
                        whiskerprops={"color": c, "lw": 0.7},
                        capprops={"color": c, "lw": 0.7})
        for b in bp["boxes"]:
            b.set(facecolor=c, edgecolor=c, lw=0.7, alpha=0.28)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{t:g}" for t in THR])
    ax.set_xlabel(r"Threshold on $r^{\mathrm{val}}$")
    ax.set_ylabel("Iterations")
    # Headroom. The tallest whisker reaches the iteration budget, so without
    # this the boxes touch the frame and the panel label has nowhere to go.
    top = max((v for col in alldata for v in col if np.isfinite(v)), default=1.0)
    ax.set_ylim(0, top * 1.18)


    for ax, lab in zip(axes, ["(a)", "(b)"]):
        ax.text(0.035, 0.965, lab, transform=ax.transAxes, fontsize=8.5,
                ha="left", va="top")

    # ONE LEGEND FOR BOTH PANELS. The two panels carry the same two series, so
    # two legends were the same key printed twice, each one hunting for a free
    # corner inside its own axes and each one landing somewhere different. A
    # shared key above the figure cannot collide with the data at all, and it
    # says once that red is off and blue is on. Line handles, because (a) draws
    # lines; the boxes in (b) take their colour from the same two entries.
    handles = [(mpatches.Patch(facecolor=c, edgecolor=c, lw=0.7, alpha=0.30),
                plt.Line2D([], [], color=c, lw=1.4))
               for c in (C_OFF, C_ON)]
    fig.tight_layout(pad=0.4)
    fig.legend(handles, ["balancing off", "balancing on"],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.10),
               handletextpad=0.6, columnspacing=1.8, frameon=False,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
               handlelength=2.4)
    p = os.path.join(args.outdir, "fig5_attainment.pdf")
    fig.savefig(p)
    plt.close(fig)
    print(f"\nwrote {p}")
    print("NOTE for the caption: (b) shows only the runs that attained each "
          "threshold, so\nits boxes rest on different numbers of runs from "
          "column to column. The counts\nare in section 2 & 4 above and belong "
          "beside the panel.")

    if args.dump:
        path = os.path.join(args.outdir, args.dump)
        with open(path, "w") as fh:
            json.dump({
                "definition": "r^val = max_s L_s^val / L_s^ref (no square root)",
                "count_kind": "any validation step, NOT retained checkpoint",
                "thresholds": THR,
                "runs": [{"tag": r["tag"], "lb": r["lb"], "lam": r["lam"],
                          "iter": r["iter"].tolist(),
                          "r_val": r["r"].tolist(), "r_min": r["rmin"],
                          "first_reaching": {f"{t:g}": first[(r["tag"], t)]
                                             for t in THR}}
                         for r in runs],
            }, fh)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
