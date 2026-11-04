#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The run-level spread column, from the archived retained checkpoints.

    per run     spread_run = max_s r_s / min_s r_s
    per cell    median over runs of spread_run

with r_s = L_s^val / L_s^ref, the CURRENT definition. This is NOT the column
the supplementary table carries today, which is

    max_s med_runs r_s / min_s med_runs r_s ,

a ratio of two across-run medians whose numerator and denominator generally
come from different families AND different runs. The two are different
statistics, not rescalings: within-run scatter enters every per-run ratio
before the median is taken, so this one sits above the other by an amount that
depends on how noisy each setting is. Both are printed below, side by side,
with the one that goes in the table marked. Do not derive either from the
other.

SQUARED ONCE, AND CHECKED. The archived best["ratios"] are on the superseded
square-root scale -- they are error / sqrt(L_s^ref). Squaring them once gives
the current definition. The guard that catches a second squaring, or a wrong
field, is that for every run

    max_s (stored ratio_s)^2  ==  (stored worst_ratio)^2

exactly, because worst_ratio is by construction the largest stored family
ratio. A double square breaks it at the fourth power; reading a field that is
already converted breaks it too. Any violation stops the script.

CCF AT ZERO COUPLING. The population cross-correlation vanishes there, so CCF
carries no relative ratio and the archive stores it as non-finite. Families are
selected by finiteness, so lambda = 0 is a four-family spread and every other
coupling is five. The count of families actually used is printed per cell.

NO FILTERING ON SUCCESS. Every run of the cell is included.

Usage
-----
    python spread_runwise.py --rundir runs/grid
    python spread_runwise.py --rundir runs/grid --out spread_runwise.json
"""

import argparse
import glob
import json
import os
import re

import numpy as np

FAMS = ("ACF", "CCF", "ACF2", "PDF", "PROJ")
# L2H10 pinned in the pattern: the capacity is part of the specification, and
# a stray L3H75 record from the capacity pilot must not be swept in.
PAT = re.compile(r"^run_nf(\d)_lb(on|off)_din(\d)_L2H10_lam([0-9p]+)_s(\d+)\.json$")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rundir", default="runs/grid")
    ap.add_argument("--nf", type=int, default=3, help="file-name convention")
    ap.add_argument("--din", type=int, default=2)
    ap.add_argument("--lams", default="0,0.5,1,2,4,8")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    want_lams = [float(v) for v in args.lams.split(",")]
    cells = {}
    worst_gate = 0.0
    n_files = 0

    for p in sorted(glob.glob(os.path.join(args.rundir, "run_*.json"))):
        m = PAT.match(os.path.basename(p))
        if not m:
            continue
        nf, lb, din = int(m.group(1)), m.group(2), int(m.group(3))
        lam = float(m.group(4).replace("p", "."))
        if nf != args.nf or din != args.din or not any(
                abs(lam - w) < 1e-12 for w in want_lams):
            continue
        n_files += 1
        rec = json.load(open(p))
        b = rec.get("best", {})
        ra = b.get("ratios")
        if not ra:
            raise SystemExit(f"{p}: the record carries no best['ratios']; "
                             f"this cell cannot be formed from the archive.")

        # SQUARE ONCE, HERE.
        r = {k: float(v) ** 2 for k, v in ra.items()
             if k in FAMS and np.isfinite(float(v))}
        if len(r) < 2:
            raise SystemExit(f"{p}: only {len(r)} applicable family ratios; "
                             f"a spread needs at least two.")

        # THE GATE. max over the squared stored ratios must equal the square of
        # the stored worst_ratio, exactly.
        w = b.get("worst_ratio")
        if w is None or not np.isfinite(float(w)):
            raise SystemExit(f"{p}: no usable best['worst_ratio'] to check "
                             f"the conversion against.")
        w2 = float(w) ** 2
        mx = max(r.values())
        worst_gate = max(worst_gate, abs(mx - w2) / max(w2, 1e-300))

        cells.setdefault((lam, lb), []).append({
            "tag": os.path.basename(p)[4:-5],
            "seed": int(m.group(5)),
            "fams": sorted(r),
            "spread_run": mx / min(r.values()),
            "r": r,
        })

    if not cells:
        raise SystemExit(
            f"no run_nf{args.nf}_lb*_din{args.din}_L2H10_lam*_s*.json in "
            f"{args.rundir}. Check the capacity in the file names.")

    print(f"{n_files} run records read from {os.path.abspath(args.rundir)}")
    print(f"gate: max_s(stored ratio)^2 vs (stored worst_ratio)^2, largest "
          f"relative difference {worst_gate:.3e}")
    if worst_gate > 1e-12:
        raise SystemExit(
            "the two disagree. Either the stored ratios were already squared "
            "(so this script squared them twice), or worst_ratio is not the "
            "maximum of the stored family ratios. Stop and find out which.")

    lams = sorted({k[0] for k in cells})
    missing = [(l, lb) for l in want_lams for lb in ("off", "on")
               if (l, lb) not in cells]
    if missing:
        print("\nMISSING CELLS (no data; nothing is substituted):")
        for l, lb in missing:
            print(f"  lambda={l:g}, balancing {lb}")

    # ---- the new column -----------------------------------------------------
    print(f"\n{'='*78}\nnew spread column: median over runs of "
          f"max_s r_s / min_s r_s, per run\n{'='*78}")
    print(f"{'lam':<6}{'bal':<5}{'n':>4}{'families':>10}"
          f"{'median spread (full)':>24}{'2 dp':>8}")
    out = {}
    for l in lams:
        for lb in ("off", "on"):
            v = cells.get((l, lb))
            if not v:
                continue
            nfam = sorted({len(d["fams"]) for d in v})
            s = np.asarray([d["spread_run"] for d in v], dtype=float)
            med = float(np.median(s))
            out[f"lam{l:g}_{lb}"] = {
                "n": int(s.size), "n_families": nfam,
                "median_spread_runwise": med,
                "q1": float(np.percentile(s, 25)),
                "q3": float(np.percentile(s, 75)),
                "min": float(s.min()), "max": float(s.max()),
                "per_run": {d["tag"]: d["spread_run"] for d in v},
            }
            print(f"{l:<6g}{lb:<5}{s.size:>4}"
                  f"{','.join(str(x) for x in nfam):>10}"
                  f"{med:>24.12f}{med:>8.2f}")

    # ---- the distribution, for judging the median ---------------------------
    print(f"\n{'='*78}\nthe same cells: distribution of the per-run spread\n"
          f"{'='*78}")
    print(f"{'lam':<6}{'bal':<5}{'n':>4}{'min':>9}{'Q1':>9}{'median':>9}"
          f"{'Q3':>9}{'max':>9}")
    for l in lams:
        for lb in ("off", "on"):
            k = f"lam{l:g}_{lb}"
            if k not in out:
                continue
            o = out[k]
            print(f"{l:<6g}{lb:<5}{o['n']:>4}{o['min']:>9.2f}{o['q1']:>9.2f}"
                  f"{o['median_spread_runwise']:>9.2f}{o['q3']:>9.2f}"
                  f"{o['max']:>9.2f}")

    # ---- the column it replaces, for comparison only ------------------------
    #
    # Printed so the size of the change is visible. It is NOT the column to
    # paste: the header says so, and the two differ by construction.
    print(f"\n{'='*78}\nFOR COMPARISON ONLY -- the current column "
          f"(ratio of across-run medians).\nDO NOT put these in the new "
          f"table.\n{'='*78}")
    print(f"{'lam':<6}{'bal':<5}{'current':>10}{'new':>10}{'new/current':>13}")
    for l in lams:
        for lb in ("off", "on"):
            v = cells.get((l, lb))
            if not v:
                continue
            med_fam = {}
            for f in FAMS:
                vals = [d["r"][f] for d in v if f in d["r"]]
                if vals:
                    med_fam[f] = float(np.median(vals))
            cur = max(med_fam.values()) / min(med_fam.values())
            new = out[f"lam{l:g}_{lb}"]["median_spread_runwise"]
            print(f"{l:<6g}{lb:<5}{cur:>10.2f}{new:>10.2f}{new / cur:>13.3f}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({
                "definition": ("per run: max_s r_s / min_s r_s with "
                               "r_s = (archived best.ratios[s])^2; then the "
                               "median over the runs of the cell"),
                "selection": ("all runs of nf3 / d_in 2 / L2H10 at the listed "
                              "couplings; no filtering on validation success"),
                "ccf": "excluded wherever the archived ratio is non-finite",
                "cells": out,
            }, fh, indent=1)
        print(f"\nwrote {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
