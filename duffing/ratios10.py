#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Section 4 on the loss-ratio definition, with tau = 10. Items 1 and 4.

WHAT CHANGED. The family-level ratio is now the ratio of the losses,

    r_s^val = L_s^val / L_s^ref ,

where it used to be the square root of that. The run-level ratio is still the
largest applicable family ratio, and a run is successful when it is at most 10.

WHY NOTHING IS RERUN. What run_<tag>.json stores is EV.errors, which is an rms
discrepancy -- the square root of a loss -- so the stored ratios are exactly
the square roots of the new ones. Squaring each run's ratio recovers the new
definition with no loss of precision. Checkpoint selection used the validation
aggregate and is untouched, so the retained checkpoint of every run is the
same one as before.

THE ORDER OF OPERATIONS IS THE WHOLE POINT. Every statistic below is computed
from per-run ratios that were squared FIRST. A median of squares is not the
square of a median, so squaring the published table would give different and
wrong numbers -- for the family medians, and for the spread, which is a ratio
of two such medians. The old and new columns are printed side by side so the
size of that difference is visible rather than assumed.

TWO KINDS OF COUNT, NEVER MIXED. This script reports only the first:

  retained-checkpoint successes   r^val at the retained checkpoint <= tau
  any-step attainment             r^val <= tau at ANY validation step

The second needs the validation history, not the run record, and is reported
by the figure script. They answer different questions and differ by several
runs; the manuscript must not quote one as the other.

Usage
-----
    python ratios10.py
    python ratios10.py --tau 10 --rundir runs/grid
"""

import argparse
import glob
import json
import os

import numpy as np

FAMS = ("ACF", "CCF", "ACF2", "PDF", "PROJ")


def q(v, p):
    return float(np.percentile(np.asarray(v, dtype=float), p))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rundir", default="runs/grid")
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--tau-old", type=float, default=3.0,
                    help="the superseded threshold, for the comparison column")
    ap.add_argument("--out", default=None, help="optional json dump")
    args = ap.parse_args()

    # ---- load every run record, keyed by configuration and coupling ---------
    runs = {}
    for p in sorted(glob.glob(os.path.join(args.rundir, "run_*.json"))):
        r = json.load(open(p))
        b = r.get("best", {})
        if not b.get("ratios"):
            continue
        key = (int(r["n_f"]), bool(r["loss_balancing"]), int(r["d_in"]),
               round(float(r["lambda"]), 3))
        runs.setdefault(key, []).append({
            "tag": os.path.basename(p)[4:-5],
            # SQUARED PER RUN, HERE, BEFORE ANYTHING ELSE TOUCHES THEM.
            "r2": {k: float(v) ** 2 for k, v in b["ratios"].items()},
            "w2": float(b["worst_ratio"]) ** 2,
            "w_old": float(b["worst_ratio"]),
        })
    if not runs:
        raise SystemExit(f"no usable run_*.json in {args.rundir}")

    # ---- consistency: max-then-square must equal square-then-max ------------
    # Squaring is monotone on non-negative numbers, so the run-level ratio can
    # be taken either way. That is an argument, not a measurement; this checks
    # it against the stored field so a surprise in the records cannot pass.
    worst = 0.0
    for v in runs.values():
        for d in v:
            mx = max(d["r2"][k] for k in d["r2"]
                     if np.isfinite(d["r2"][k]))
            worst = max(worst, abs(mx - d["w2"]) / max(d["w2"], 1e-30))
    print(f"check: square-then-max vs stored worst^2, largest relative "
          f"difference {worst:.3e}")
    if worst > 1e-9:
        raise SystemExit("the stored worst_ratio is not the max of the stored "
                         "family ratios; stop and find out why.")

    lams = sorted({k[3] for k in runs})
    cfgs = sorted({k[:3] for k in runs})

    # ---- item 4: the loss-balancing table, default cell ---------------------
    print(f"\n{'='*74}\nitem 4: loss-balancing table, n_f=4 (file nf3), "
          f"d_in=2, new r_s^val = L_s^val / L_s^ref\n{'='*74}")
    hdr = f"{'lam':<5}{'bal':<5}" + "".join(f"{k:>8}" for k in FAMS) \
          + f"{'med r^val':>11}{'spread':>9}{'n':>4}"
    print(hdr)
    tab = {}
    for lam in lams:
        for lb in (False, True):
            key = (3, lb, 2, lam)
            if key not in runs:
                continue
            v = runs[key]
            med = {}
            for k in FAMS:
                vals = [d["r2"][k] for d in v
                        if k in d["r2"] and np.isfinite(d["r2"][k])]
                med[k] = float(np.median(vals)) if vals else float("nan")
            fin = [med[k] for k in FAMS if np.isfinite(med[k])]
            spread = max(fin) / min(fin)
            mw = float(np.median([d["w2"] for d in v]))
            tab[(lam, lb)] = {"fam": med, "med_worst": mw, "spread": spread,
                              "n": len(v)}
            print(f"{lam:<5g}{'on' if lb else 'off':<5}"
                  + "".join(f"{med[k]:>8.2f}" if np.isfinite(med[k])
                            else f"{'--':>8}" for k in FAMS)
                  + f"{mw:>11.2f}{spread:>9.2f}{len(v):>4}")

    print("\nthe same rows on the superseded square-root definition, for "
          "comparison only:")
    for lam in lams:
        for lb in (False, True):
            key = (3, lb, 2, lam)
            if key not in runs:
                continue
            v = runs[key]
            med = {}
            for k in FAMS:
                vals = [np.sqrt(d["r2"][k]) for d in v
                        if k in d["r2"] and np.isfinite(d["r2"][k])]
                med[k] = float(np.median(vals)) if vals else float("nan")
            fin = [med[k] for k in FAMS if np.isfinite(med[k])]
            print(f"{lam:<5g}{'on' if lb else 'off':<5}"
                  + "".join(f"{med[k]:>8.2f}" if np.isfinite(med[k])
                            else f"{'--':>8}" for k in FAMS)
                  + f"{float(np.median([d['w_old'] for d in v])):>11.2f}"
                  + f"{max(fin) / min(fin):>9.2f}")
    print("Note the spread column: it is a ratio of two medians of squares, so "
          "it is NOT\nthe square of the old spread. Do not derive one from the "
          "other.")

    # ---- item 1: retained-checkpoint successes, every configuration ---------
    print(f"\n{'='*74}\nitem 1: retained-checkpoint successes, "
          f"r^val <= {args.tau:g}\n{'='*74}")
    print(f"{'n_f':<5}{'bal':<5}{'d_in':<6}"
          + "".join(f"lam {l:<7g}" for l in lams) + f"{'total':>9}"
          + f"{'  (old tau=' + f'{args.tau_old:g})':>14}")
    counts = {}
    for cfg in cfgs:
        nf, lb, din = cfg
        row, tot, n_tot, tot_old = "", 0, 0, 0
        for lam in lams:
            v = runs.get((nf, lb, din, lam), [])
            if not v:
                row += f"{'--':<11}"
                continue
            k = sum(1 for d in v if np.isfinite(d["w2"]) and d["w2"] <= args.tau)
            ko = sum(1 for d in v
                     if np.isfinite(d["w_old"]) and d["w_old"] <= args.tau_old)
            tot += k; n_tot += len(v); tot_old += ko
            counts[(nf, lb, din, lam)] = (k, len(v), ko)
            row += f"{f'{k}/{len(v)}':<11}"
        print(f"{nf + 1:<5}{'on' if lb else 'off':<5}{din:<6}{row}"
              f"{f'{tot}/{n_tot}':>9}{f'{tot_old}/{n_tot}':>14}")
    print(f"\nr^val <= {args.tau:g} on the new definition is r_old <= "
          f"{np.sqrt(args.tau):.4f}, which is NOT the old\nthreshold of "
          f"{args.tau_old:g}. The two count columns above differ for that reason "
          f"alone.")

    # ---- the run-level distribution, for choosing what to say in the text ---
    print(f"\n{'='*74}\nrun-level r^val distribution, default cell "
          f"(n_f=4, balancing on, d_in=2)\n{'='*74}")
    print(f"{'lam':<6}{'min':>8}{'Q1':>8}{'median':>9}{'Q3':>8}{'max':>8}"
          f"{'n>tau':>7}")
    for lam in lams:
        v = runs.get((3, True, 2, lam), [])
        if not v:
            continue
        w = [d["w2"] for d in v]
        print(f"{lam:<6g}{min(w):>8.2f}{q(w, 25):>8.2f}"
              f"{float(np.median(w)):>9.2f}{q(w, 75):>8.2f}{max(w):>8.2f}"
              f"{sum(1 for x in w if x > args.tau):>7}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"tau": args.tau,
                       "definition": "r_s = L_s^val / L_s^ref (no square root)",
                       "table": {f"{l}_{'on' if b else 'off'}":
                                 {"families": t["fam"],
                                  "med_worst": t["med_worst"],
                                  "spread": t["spread"], "n": t["n"]}
                                 for (l, b), t in tab.items()},
                       "successes": {f"nf{a}_lb{'on' if b else 'off'}_din{c}_lam{d}":
                                     {"k": k, "n": n, "k_old_tau": ko}
                                     for (a, b, c, d), (k, n, ko)
                                     in counts.items()}}, fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
