#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The input-dimension comparison, scored on the evaluation ensemble T.

n_f = 4 (file label 3), balancing on, d_in = 1, 2, 3, six couplings. For every
run: regenerate N trajectories from its retained checkpoint and score them
against T.

WHAT IS PLOTTED AND TABULATED

    L_total^eval(run) = sum_s L_s^eval(run),   L_s^eval = e_s^2 ,

unweighted -- no adaptive weights -- and NOT divided by any reference loss. At
lambda = 0 the sum runs over four families: the population cross-correlation
vanishes identically there, so CCF carries no relative error (evaluate.py:286
reports the absolute discrepancy instead, and it is excluded here as it is from
the training aggregate). Five families at every other coupling.

THE REFERENCE STATISTICS COME FROM T, FOR ALL FIVE FAMILIES.
evaluate.reference_stats() substitutes the analytic densities for PDF, PROJ,
JOINT and PROJ_HOLDOUT whatever segment it is given (evaluate.py:288-296), so
calling it would score the two density families against the analytic reference
and only the correlation families against T. This script therefore builds the
reference statistics itself, with evaluate.stats() on the evaluation segment, so
that every family is compared with the same empirical ensemble. evaluate.py is
not modified: the substituting function is simply not the one called. The
analytic-density variant is computed as well and printed beside the first, so
the size of that choice is visible rather than argued about.

PREPROCESSING, GRID AND BANDWIDTH ARE THE FITTING ENSEMBLE'S.
standardise() applies mu and scale from reference_<tag>.npz (evaluate.py:199),
and the output grid, the fine grid, the 1-D and 2-D bandwidths and the lag count
all come from the same file. Nothing here recomputes any of them from T, which
would make the comparison depend on the segment being scored.

A THIRD NOISE STREAM, AS IN report.py. generate() seeds with seed + 30000;
training used seed + 10000 and validation seed + 20000. Scoring on the
validation stream would report the ensemble that won the checkpoint selection.

TWO KINDS OF NUMBER IN ONE TABLE, AND WHY THAT IS ALLOWED HERE. The loss column
is evaluation-side, measured here. The success column is validation-side, read
from each run's retained-checkpoint record, r^val = max_s L_s^val / L_s^ref <=
10. They are different quantities with different names and different
superscripts, which is exactly what the earlier version of this table got wrong
by printing two different r^val under one heading. The column totals are checked
against the independently computed 61 / 66 / 66 of 70 and a mismatch is
reported, not absorbed.

NO RETRAINING. Every checkpoint is read from disk and only sampled from.

Usage
-----
    python eval_din.py --datadir data --rundir runs/grid --outdir figs \\
                       --out din_eval.json
    python eval_din.py --lams 4 --dins 2          # one cell, for a quick check
"""

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import (FuncFormatter, LogLocator, MaxNLocator,
                               NullFormatter)

import torch

import evaluate as EV
from report import generate

WIDTH = 6.02
RAMP_LIGHT, RAMP_DARK = "#86aecd", "#10304a"
MARKS = ["o", "s", "^"]
FAMS = ("ACF", "CCF", "ACF2", "PDF", "PROJ")
CKPT = re.compile(r"net_nf(\d)_lb(on|off)_din(\d)_L(\d+)H(\d+)_lam([0-9p]+)_s(\d+)\.pt$")


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "lines.linewidth": 1.1,
        "axes.grid": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def ramp(n, light=RAMP_LIGHT, dark=RAMP_DARK):
    c0 = np.array([int(light[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    c1 = np.array([int(dark[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return [tuple(c0 + (c1 - c0) * t) for t in np.linspace(0, 1, n)]


def log_ticks(ax, vals):
    """Pick the vertical scale from the data, and label it plainly.

    A log axis is right when the panel spans most of a decade or more. Below
    that it is a liability: matplotlib labels only the decades, falls back to
    labelling subdivisions, and the panel ends up with a column of
    '2.75 x 10^-2' strings down the side -- while the curves are nearly
    straight on either scale, so nothing is bought for the cost. The rule is
    mechanical and printed, so the two figures cannot silently end up on
    different scales: a factor of eight or more between the largest and
    smallest value plotted gets a log axis with ticks at 1, 2, 3 and 5 times
    each power of ten; anything narrower gets a linear axis.
    """
    v = np.asarray([x for x in vals if np.isfinite(x) and x > 0], dtype=float)
    span = float(v.max() / v.min()) if v.size else 1.0
    if span >= 8.0:
        ax.set_yscale("log")
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
    print(f"  vertical scale: {'log' if span >= 8 else 'linear'} "
          f"(largest/smallest plotted value {span:.2f})")
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--rundir", default="runs/grid")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--nf", type=int, default=3, help="file-name convention")
    ap.add_argument("--lb", choices=("on", "off"), default="on")
    ap.add_argument("--dins", default="1,2,3")
    ap.add_argument("--lams", default=None,
                    help="default: every coupling with checkpoints present")
    ap.add_argument("--n-gen", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--tau-val", type=float, default=10.0,
                    help="success criterion on the NEW definition, r^val <= 10")
    ap.add_argument("--expect", default="61,66,66",
                    help="independently computed success totals, checked")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="din_eval.json")
    ap.add_argument("--cachedir", default=None,
                    help="per-cell shards, so an interrupted run resumes; "
                         "default: din_eval_cells/ beside --out")
    ap.add_argument("--force", action="store_true",
                    help="recompute every cell, ignoring the cache")
    args = ap.parse_args()

    style()
    os.makedirs(args.outdir, exist_ok=True)
    dins = [int(v) for v in args.dins.split(",")]

    # THE DEVICE IS PART OF THE RESULT. A CPU LSTM and a cuDNN LSTM do not agree
    # bitwise, the generated ensemble differs, and the KDE errors move at the
    # percent level -- enough for two tables to disagree with identical inputs.
    # report.py records it for the same reason; one device per table.
    dev = str(args.device)
    gpu = (torch.cuda.get_device_name(0)
           if dev.startswith("cuda") and torch.cuda.is_available() else None)
    if not dev.startswith("cuda"):
        print("WARNING: generating on the CPU. These losses will differ from "
              "GPU-generated ones\n         at the percent level. Do not mix "
              "them in one table.")

    # ---- which cells exist ---------------------------------------------------
    found = {}
    for p in sorted(glob.glob(os.path.join(args.rundir, "net_*.pt"))):
        m = CKPT.search(os.path.basename(p))
        if not m:
            continue
        nf, lb, din = int(m.group(1)), m.group(2), int(m.group(3))
        if nf != args.nf or lb != args.lb or din not in dins:
            continue
        lam = float(m.group(6).replace("p", "."))
        found.setdefault((lam, din), []).append(
            (p, int(m.group(4)), int(m.group(5)), int(m.group(7))))
    if not found:
        raise SystemExit(f"no checkpoints net_nf{args.nf}_lb{args.lb}_din*.pt "
                         f"in {args.rundir}")
    lams = (sorted(float(v) for v in args.lams.split(","))
            if args.lams else sorted({k[0] for k in found}))
    missing = [(l, d) for l in lams for d in dins if (l, d) not in found]
    if missing:
        raise SystemExit(f"incomplete grid; no checkpoints for {missing}")

    # ---- shards, so an interrupted job does not start over -------------------
    #
    # One file per (coupling, d_in) cell, written as soon as that cell finishes.
    # A job that hits its wall clock, or a login-node run that loses its
    # connection, keeps everything it had done; rerunning the same command picks
    # up from the first unfinished cell. --force recomputes regardless.
    #
    # A shard records the settings that would make its numbers incomparable --
    # the ensemble size and the device -- and a rerun under different ones is
    # refused rather than silently mixed into one table. That is the same rule
    # report.py applies to the anchor.
    cachedir = args.cachedir or os.path.join(
        os.path.dirname(os.path.abspath(args.out)) or ".", "din_eval_cells")
    os.makedirs(cachedir, exist_ok=True)

    def shard_path(lam, din):
        return os.path.join(
            cachedir, f"cell_nf{args.nf}_lb{args.lb}_"
                      f"din{din}_lam{lam:g}".replace(".", "p") + ".json")

    runs, prov = {}, {}
    t0 = time.time()
    for lam in lams:
        tag = f"lam{lam:g}".replace(".", "p")

        todo = [d for d in dins
                if args.force or not os.path.exists(shard_path(lam, d))]
        for d in dins:
            if d in todo:
                continue
            sh = json.load(open(shard_path(lam, d)))
            if (int(sh["n_gen"]) != int(args.n_gen)
                    or str(sh["device"]) != dev):
                raise SystemExit(
                    f"{shard_path(lam, d)} was computed with n_gen="
                    f"{sh['n_gen']} on {sh['device']}, this run is n_gen="
                    f"{args.n_gen} on {dev}. Delete the cache or match the "
                    f"settings; do not mix them in one table.")
            runs[(lam, d)] = sh["runs"]
            prov[lam] = sh["provenance"]
            print(f"lambda={lam:g} d_in={d}: {len(sh['runs'])} runs read from "
                  f"the cache")
        if not todo:
            continue

        traj_path = os.path.join(args.datadir, f"traj_{tag}.npy")
        ref = EV.load_reference(args.datadir, tag, traj_path)
        traj = np.load(traj_path, mmap_mode="r")
        split = EV.split_from_reference(ref)
        deg = EV.ccf_is_degenerate(ref)
        lo, hi = split["evaluation"]

        # ---- T, recorded rather than assumed --------------------------------
        prov[lam] = {
            "traj_file": os.path.abspath(traj_path),
            "archive_shape": list(traj.shape),
            "n_traj_total": int(traj.shape[1]),
            "l_seq": int(traj.shape[0]),
            "splits": {k: list(v) for k, v in split.items()},
            "T_segment": [int(lo), int(hi)],
            "T_n_traj": int(hi - lo),
            "b_fit": int(ref["b_fit"]),
            "n_gen": int(args.n_gen),
            "ccf_degenerate": bool(deg),
            "constants_sha256": str(ref["constants_sha256"])[:16],
            "device": dev, "gpu": gpu,
        }
        print(f"\nlambda={lam:g}  T = evaluation segment [{lo}, {hi}) of "
              f"{traj.shape[1]} trajectories, length {traj.shape[0]}\n"
              f"  {hi - lo} trajectories in T; B={int(ref['b_fit'])}, "
              f"n_gen={args.n_gen}; CCF "
              f"{'degenerate, excluded' if deg else 'included'}")

        # EMPIRICAL T FOR EVERY FAMILY (see the header), plus the substituting
        # version for the comparison column. "raw" because the archive is in
        # physical coordinates and the fixed affine map has to be applied once.
        sr_T = EV.stats(traj[:, lo:hi, :], ref, "raw")
        sr_an = EV.reference_stats(traj, ref, (lo, hi))
        fams = [f for f in FAMS if not (f == "CCF" and deg)]

        for din in todo:
            cells = sorted(found[(lam, din)], key=lambda t: t[3])
            for ck, layers, hidden, seed in cells:
                arr, it = generate(ck, ref, args.n_gen, din, layers, hidden,
                                   args.warmup, seed, dev)
                sg = EV.stats(arr, ref, "standardized")
                e_T = EV.errors(sg, sr_T, deg)
                e_an = EV.errors(sg, sr_an, deg)
                L = {f: float(e_T[f]) ** 2 for f in fams}
                L_an = {f: float(e_an[f]) ** 2 for f in fams}
                run_tag = os.path.basename(ck)[4:-3]
                rec_path = os.path.join(args.rundir, f"run_{run_tag}.json")
                rec = json.load(open(rec_path)) if os.path.exists(rec_path) else {}
                w = rec.get("best", {}).get("worst_ratio", float("nan"))
                runs.setdefault((lam, din), []).append({
                    "run_tag": run_tag, "seed": seed, "iter": it,
                    "L": L, "total": float(sum(L.values())),
                    "total_analytic_densities": float(sum(L_an.values())),
                    # validation side, new definition: the stored ratio is on
                    # the superseded square-root scale, so it is squared here,
                    # per run, exactly as ratios10.py does.
                    "r_val": (float(w) ** 2 if np.isfinite(w) else float("nan")),
                })
            n = len(runs[(lam, din)])
            with open(shard_path(lam, din), "w") as fh:
                json.dump({"lambda": lam, "d_in": din, "n_gen": int(args.n_gen),
                           "device": dev, "gpu": gpu,
                           "provenance": prov[lam],
                           "runs": runs[(lam, din)]}, fh, indent=1)
            print(f"  d_in={din}: {n} runs, {time.time() - t0:.0f}s elapsed, "
                  f"cell written")

    # ---- aggregates ----------------------------------------------------------
    med, q1, q3, nrun = {}, {}, {}, {}
    succ = {}
    for k, v in runs.items():
        t = np.asarray([r["total"] for r in v], dtype=float)
        med[k], q1[k], q3[k] = (float(np.median(t)),
                                float(np.percentile(t, 25)),
                                float(np.percentile(t, 75)))
        nrun[k] = int(t.size)
        rv = np.asarray([r["r_val"] for r in v], dtype=float)
        succ[k] = (int(np.sum(np.isfinite(rv) & (rv <= args.tau_val))), int(rv.size))

    # ---- the figure ----------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(WIDTH * 0.5, 2.2))
    x = np.arange(len(lams))
    off = (np.arange(len(dins)) - (len(dins) - 1) / 2.0) * 0.085
    for c, mk, din, dx in zip(ramp(len(dins)), MARKS, dins, off):
        m = np.array([med[(l, din)] for l in lams])
        lo_ = m - np.array([q1[(l, din)] for l in lams])
        hi_ = np.array([q3[(l, din)] for l in lams]) - m
        ax.errorbar(x + dx, m, yerr=np.vstack([lo_, hi_]), fmt=mk + "-",
                    ms=3.4, lw=1.1, color=c, ecolor=c, elinewidth=0.7,
                    capsize=1.8, capthick=0.7, zorder=3,
                    label=rf"$d_{{\mathrm{{in}}}}={din}$")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l:g}" for l in lams])
    ax.set_xlabel(r"coupling strength $\lambda$")
    ax.set_ylabel(r"$\mathcal{L}^{\mathrm{eval}}_{\mathrm{total}}$")
    log_ticks(ax, [q1[(l, d)] for l in lams for d in dins]
               + [q3[(l, d)] for l in lams for d in dins])
    # No reference line, as in the JOINT-loss figure: this is a loss, not a
    # ratio to a reference block, so no value on the axis means "as close as
    # the data is to itself".
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(dins),
              handletextpad=0.5, columnspacing=1.3, fontsize=7.0,
              borderaxespad=0.15)
    fig.tight_layout(pad=0.4)
    path = os.path.join(args.outdir, "fig_din_loss.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"\nwrote {path}")

    # ---- the table -----------------------------------------------------------
    print(f"\n{'='*78}\nn_f={args.nf + 1}, balancing {args.lb}; "
          f"L_total^eval = sum_s L_s^eval against T, unweighted;\nsuccess is "
          f"r^val <= {args.tau_val:g} at the retained checkpoint "
          f"(validation side)\n{'='*78}")
    print(f"{'lam':<6}" + "".join(f"d_in={d:<18}" for d in dins))
    print(f"{'':<6}" + "".join(f"{'L_total^eval  succ':<23}" for _ in dins))
    for l in lams:
        row = f"{l:<6g}"
        for d in dins:
            k, n = succ[(l, d)]
            row += f"{med[(l, d)]:<12.4f}{f'{k}/{n}':<11}"
        print(row)
    tot = {d: (sum(succ[(l, d)][0] for l in lams),
               sum(succ[(l, d)][1] for l in lams)) for d in dins}
    print(f"{'total':<6}" + "".join(f"{'':<12}{f'{tot[d][0]}/{tot[d][1]}':<11}"
                                    for d in dins))

    exp = [int(v) for v in args.expect.split(",")] if args.expect else []
    if exp and len(exp) == len(dins):
        got = [tot[d][0] for d in dins]
        if got == exp:
            print(f"\nsuccess totals match the independently computed "
                  f"{', '.join(map(str, exp))} of {tot[dins[0]][1]}.")
        else:
            print(f"\nMISMATCH: success totals are {got}, the independent "
                  f"computation gave {exp}.\nThe two were built from the same "
                  f"field of the same records, so they cannot both be right. "
                  f"Stop and find out why before using either.")

    print(f"\nmedian [Q1, Q3] of L_total^eval, the figure's numbers")
    print(f"{'lam':<6}" + "".join(f"{'d_in=' + str(d):<28}" for d in dins))
    for l in lams:
        print(f"{l:<6g}" + "".join(
            f"{med[(l, d)]:.4f} [{q1[(l, d)]:.4f}, {q3[(l, d)]:.4f}]".ljust(28)
            for d in dins))

    # The size of the one methodological choice that had to be made here.
    print(f"\nthe same medians with the ANALYTIC densities substituted for "
          f"PDF and PROJ\n(what evaluate.reference_stats would have given); "
          f"printed for comparison only:")
    print(f"{'lam':<6}" + "".join(f"{'d_in=' + str(d):<16}" for d in dins))
    for l in lams:
        row = f"{l:<6g}"
        for d in dins:
            v = float(np.median([r["total_analytic_densities"]
                                 for r in runs[(l, d)]]))
            row += f"{v:<16.4f}"
        print(row)

    with open(args.out, "w") as fh:
        json.dump({
            "quantity": ("L_total^eval = sum_s L_s^eval, L_s^eval = e_s^2 "
                         "against the evaluation ensemble T; unweighted, not "
                         "divided by any reference loss; CCF excluded where "
                         "the population cross-correlation vanishes"),
            "reference_statistics": ("empirical, from T, for all families "
                                     "(evaluate.stats on the evaluation "
                                     "segment, NOT reference_stats)"),
            "preprocessing": ("mu, scale, grids, bandwidths and lag count from "
                              "reference_<tag>.npz, i.e. the fitting ensemble"),
            "config": {"n_f": args.nf + 1, "nf_file": args.nf,
                       "loss_balancing": args.lb, "d_in": dins,
                       "n_gen": args.n_gen, "warmup": args.warmup,
                       "tau_val": args.tau_val},
            "provenance": {f"lam{l:g}": prov[l] for l in lams},
            "cells": {f"lam{l:g}_din{d}": {
                "median": med[(l, d)], "q1": q1[(l, d)], "q3": q3[(l, d)],
                "n_runs": nrun[(l, d)],
                "success": {"k": succ[(l, d)][0], "n": succ[(l, d)][1]},
                "runs": runs[(l, d)]} for l in lams for d in dins},
        }, fh, indent=1)
    print(f"\nwrote {args.out}  (per-run family losses and totals included)")
    print(f"total wall time {time.time() - t0:.0f}s on {gpu or dev}")


if __name__ == "__main__":
    main()
