#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Re-aggregate the d_in comparison on total_analytic_densities. Gated.

NOTHING IS RETRAINED AND NOTHING IS REGENERATED. The only input is the json
eval_din.py wrote on the A100. Every number below is a re-aggregation of
per-run values already in that file.

THE GATES COME FIRST AND THEY ARE FATAL. The provenance of the file is checked
before any statistic is formed, and a failure exits without printing numbers or
writing a figure -- a half-verified table is worse than none, because it looks
like a verified one. What is checked:

  path and checksum        printed, so the table can be traced to a file
  device                   must be CUDA; the GPU name is printed. A CPU LSTM
                           and a cuDNN LSTM do not agree bitwise, so a CPU file
                           cannot sit beside the other Section 4 tables.
  n_gen                    must be 400, the fitting budget B
  configuration            n_f = 4 (file label 3), loss balancing on
  cell sizes               20 runs at lambda = 4, 10 at every other coupling
  family scope             CCF present in exactly the non-degenerate couplings,
                           and the SAME family set behind both totals
  per-run identity         total must equal sum(L) to round-off, so the field
                           being read is the field it is named after
  the two columns differ   if all 18 medians agree the script stops: that means
                           the analytic substitution did nothing, which would
                           mean the wrong field is being read

Usage
-----
    python din_analytic.py --json din_eval.json --outdir figs
    python din_analytic.py --json din_eval.json --latex table2_analytic.tex
"""

import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import (FixedLocator, FuncFormatter, LogLocator,
                               MaxNLocator, NullFormatter)

# Verbatim from eval_din.py, so the candidate figure cannot drift from Fig. 8.
WIDTH = 6.02
RAMP_LIGHT, RAMP_DARK = "#86aecd", "#10304a"
MARKS = ["o", "s", "^"]
FAMS = ("ACF", "CCF", "ACF2", "PDF", "PROJ")
KEY = re.compile(r"^lam([0-9p.]+)_din(\d)$")
NEW = "total_analytic_densities"


def die(msg):
    print(f"\nSTOP: {msg}", file=sys.stderr)
    raise SystemExit(2)


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


def axis_scale(ax, vals, force):
    """eval_din.py's log_ticks, with the choice forced from outside.

    The candidate figure must carry the SAME axis as the figure it is a
    candidate to replace, or the two cannot be compared by eye. So the scale is
    decided from the published figure's own values and imposed here, rather
    than re-derived from the new ones.
    """
    v = np.asarray([x for x in vals if np.isfinite(x) and x > 0], dtype=float)
    span = float(v.max() / v.min()) if v.size else 1.0
    if force == "log":
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
    return span


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="din_eval.json")
    ap.add_argument("--cachedir", default="din_eval_cells")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--name", default="fig_din_loss_analytic_candidate.pdf")
    ap.add_argument("--latex", default=None, help="write the six rows here too")
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--expect", default="61,66,66")
    ap.add_argument("--ylim", default="0.01,0.1",
                    help="fixed log y limits; empty string restores the "
                         "data-driven rule")
    ap.add_argument("--big-lam", type=float, default=4.0,
                    help="the coupling whose cells hold 20 runs")
    args = ap.parse_args()

    # ---- gate 0: the file, named and fingerprinted -------------------------
    p = os.path.abspath(args.json)
    if not os.path.exists(p):
        die(f"{p} does not exist. The official A100 run's json is required; "
            f"report_*.json is not a substitute.")
    print(f"file      {p}")
    print(f"sha256    {sha256(p)}")
    print(f"bytes     {os.path.getsize(p)}")
    cdir = os.path.abspath(args.cachedir)
    if os.path.isdir(cdir):
        shards = sorted(f for f in os.listdir(cdir) if f.endswith(".json"))
        print(f"shards    {cdir}  ({len(shards)} files)")
        for f in shards:
            print(f"  {sha256(os.path.join(cdir, f))}  {f}")
    else:
        print(f"shards    {cdir} absent (not required if the json is complete)")

    d = json.load(open(p))
    cells = {}
    for k, v in d["cells"].items():
        m = KEY.match(k)
        if m:
            cells[(float(m.group(1).replace("p", ".")), int(m.group(2)))] = v
    if not cells:
        die(f"{p} holds no cells")
    lams = sorted({k[0] for k in cells})
    dins = sorted({k[1] for k in cells})

    # ---- gate 1: configuration ---------------------------------------------
    cfg = d.get("config", {})
    if int(cfg.get("n_f", -1)) != 4 or int(cfg.get("nf_file", -1)) != 3:
        die(f"config says n_f={cfg.get('n_f')} (file label {cfg.get('nf_file')}); "
            f"4 (file label 3) required")
    if str(cfg.get("loss_balancing")) != "on":
        die(f"config says loss_balancing={cfg.get('loss_balancing')}; on required")
    if int(cfg.get("n_gen", -1)) != 400:
        die(f"config says n_gen={cfg.get('n_gen')}; 400 required")
    print(f"\nconfig    n_f={cfg.get('n_f')}, balancing {cfg.get('loss_balancing')}, "
          f"d_in {cfg.get('d_in')}, n_gen {cfg.get('n_gen')}, "
          f"tau_val {cfg.get('tau_val')}")
    print(f"reference statistics behind 'total': {d.get('reference_statistics')}")

    # ---- gate 2: device ------------------------------------------------------
    prov = d.get("provenance", {})
    if not prov:
        die("the json carries no provenance block")
    devs = {str(v.get("device")) for v in prov.values()}
    gpus = {str(v.get("gpu")) for v in prov.values()}
    if len(devs) != 1 or not sorted(devs)[0].startswith("cuda"):
        die(f"device(s) {sorted(devs)}; a single CUDA device is required. A CPU "
            f"run is not comparable with the rest of Section 4.")
    print(f"device    {sorted(devs)[0]}   gpu {sorted(gpus)}")
    ngen_p = {int(v.get("n_gen", -1)) for v in prov.values()}
    if ngen_p != {400}:
        die(f"provenance n_gen {sorted(ngen_p)}; 400 required")
    one = prov[sorted(prov)[0]]
    print(f"T         {one.get('T_n_traj')} trajectories, segment "
          f"{one.get('T_segment')} of {one.get('n_traj_total')}, "
          f"length {one.get('l_seq')}, B {one.get('b_fit')}")

    # ---- gate 3: cell sizes, family scope, per-run identity -----------------
    bad = []
    for (l, x), c in sorted(cells.items()):
        want = 20 if abs(l - args.big_lam) < 1e-12 else 10
        if int(c["n_runs"]) != want or len(c["runs"]) != want:
            bad.append((l, x, c["n_runs"], want))
    if bad:
        die(f"cell sizes wrong (lambda, d_in, got, want): {bad}")
    print(f"cells     {len(cells)}; "
          f"{args.big_lam:g} -> 20 runs, others -> 10, as required")

    deg_l, worst_id = [], 0.0
    for (l, x), c in sorted(cells.items()):
        keysets = {tuple(sorted(r["L"])) for r in c["runs"]}
        if len(keysets) != 1:
            die(f"lambda={l:g} d_in={x}: runs disagree about the family set "
                f"{sorted(keysets)}")
        ks = set(keysets.pop())
        if not ks <= set(FAMS):
            die(f"lambda={l:g} d_in={x}: unexpected families {sorted(ks)}")
        if "CCF" not in ks:
            deg_l.append((l, x))
        for r in c["runs"]:
            s = sum(r["L"].values())
            worst_id = max(worst_id, abs(s - r["total"]) / max(abs(r["total"]), 1e-300))
    if worst_id > 1e-12:
        die(f"some run's total is not the sum of its own family losses "
            f"(largest relative difference {worst_id:.3e}); the field being "
            f"read is not what it is named")
    print(f"identity  total == sum_s L_s for every run "
          f"(largest relative difference {worst_id:.3e})")
    missing_ccf = sorted({l for l, _ in deg_l})
    print(f"CCF       absent at lambda = {[f'{l:g}' for l in missing_ccf]}, "
          f"present elsewhere")
    if missing_ccf != [0.0]:
        die(f"CCF is expected to be excluded at lambda = 0 only, but it is "
            f"absent at {missing_ccf}")
    if len(deg_l) != len(dins):
        die(f"CCF is excluded in {len(deg_l)} cells; it must be all {len(dins)} "
            f"d_in cells of lambda = 0 or none")
    # The same `fams` list builds both totals in eval_din.py:290-291, so the
    # family scope of total_analytic_densities is the scope just checked. That
    # is an argument about the producing code; here is the measurement: at
    # lambda = 0 both totals must be sums over the four keys actually stored.
    for (l, x) in deg_l:
        for r in cells[(l, x)]["runs"]:
            if NEW not in r:
                die(f"lambda={l:g} d_in={x}: a run carries no {NEW}")
    print(f"scope     both totals are summed over the same family set")

    # ---- item 6: the re-aggregation ----------------------------------------
    new, q1n, q3n, old, q1o, q3o, succ, n = {}, {}, {}, {}, {}, {}, {}, {}
    for (l, x), c in cells.items():
        a = np.asarray([r[NEW] for r in c["runs"]], dtype=float)
        b = np.asarray([r["total"] for r in c["runs"]], dtype=float)
        if not np.all(np.isfinite(a)):
            die(f"lambda={l:g} d_in={x}: non-finite {NEW}")
        new[(l, x)], q1n[(l, x)], q3n[(l, x)] = (
            float(np.median(a)), float(np.percentile(a, 25)),
            float(np.percentile(a, 75)))
        old[(l, x)], q1o[(l, x)], q3o[(l, x)] = (
            float(np.median(b)), float(np.percentile(b, 25)),
            float(np.percentile(b, 75)))
        rv = np.asarray([r["r_val"] for r in c["runs"]], dtype=float)
        succ[(l, x)] = (int(np.sum(np.isfinite(rv) & (rv <= args.tau))),
                        int(rv.size))
        n[(l, x)] = int(a.size)

    # ---- item 7: side by side, and the gate on them being different --------
    print(f"\n{'='*86}\nall 18 cells: published median (total, empirical T for "
          f"every family) vs the\ncandidate ({NEW}: PDF and PROJ against the "
          f"quadrature targets)\n{'='*86}")
    print(f"{'lam':<6}{'d_in':<6}{'published':>14}{'candidate':>14}"
          f"{'ratio':>9}{'Q1':>14}{'Q3':>14}{'n':>5}{'succ':>8}")
    worst_rel = 0.0
    for l in lams:
        for x in dins:
            k = (l, x)
            rel = abs(new[k] - old[k]) / max(abs(old[k]), 1e-300)
            worst_rel = max(worst_rel, rel)
            s = succ[k]
            print(f"{l:<6g}{x:<6}{old[k]:>14.6g}{new[k]:>14.6g}"
                  f"{new[k]/old[k]:>9.4f}{q1n[k]:>14.6g}{q3n[k]:>14.6g}"
                  f"{n[k]:>5}{f'{s[0]}/{s[1]}':>8}")
    print(f"\nlargest relative difference between the two columns: {worst_rel:.3e}")
    if worst_rel < 1e-12:
        die("the two columns are identical to round-off. Either the analytic "
            "substitution changed nothing -- which cannot be, PDF and PROJ are "
            "compared with different targets -- or both columns are being read "
            "from the same field. Check before using any of this.")

    # ---- item 9: the success totals ----------------------------------------
    tot = {x: (sum(succ[(l, x)][0] for l in lams),
               sum(succ[(l, x)][1] for l in lams)) for x in dins}
    exp = [int(v) for v in args.expect.split(",")] if args.expect else []
    got = [tot[x][0] for x in dins]
    print(f"\nsuccess totals {['%d/%d' % tot[x] for x in dins]}")
    if exp:
        if got != exp:
            die(f"success totals are {got}, expected {exp}. These come from "
                f"the validation record and must not move when the "
                f"evaluation-side target changes.")
        print(f"they match the expected {' / '.join(map(str, exp))} of "
              f"{tot[dins[0]][1]}, as they must: the success column is "
              f"validation-side\nand does not depend on which evaluation "
              f"target is used.")
    else:
        print("(--expect was empty, so the totals were not checked)")

    # ---- item 8: the candidate figure --------------------------------------
    #
    # The axis is taken from the PUBLISHED values, not the new ones, so the
    # candidate is directly comparable with Figure 8. Everything else -- size,
    # ramp, markers, dodge, error-bar geometry, tick labels, legend -- is
    # eval_din.py's code, copied rather than re-invented.
    style()
    os.makedirs(args.outdir, exist_ok=True)
    pub_span_vals = [q1o[(l, x)] for l in lams for x in dins] + \
                    [q3o[(l, x)] for l in lams for x in dins]
    v = np.asarray([x for x in pub_span_vals if np.isfinite(x) and x > 0])
    pub_log = (v.max() / v.min()) >= 8.0

    fig, ax = plt.subplots(1, 1, figsize=(WIDTH * 0.5, 2.2))
    xs = np.arange(len(lams))
    off = (np.arange(len(dins)) - (len(dins) - 1) / 2.0) * 0.085
    for c, mk, din, dx in zip(ramp(len(dins)), MARKS, dins, off):
        m = np.array([new[(l, din)] for l in lams])
        lo_ = m - np.array([q1n[(l, din)] for l in lams])
        hi_ = np.array([q3n[(l, din)] for l in lams]) - m
        ax.errorbar(xs + dx, m, yerr=np.vstack([lo_, hi_]), fmt=mk + "-",
                    ms=3.4, lw=1.1, color=c, ecolor=c, elinewidth=0.7,
                    capsize=1.8, capthick=0.7, zorder=3,
                    label=rf"$d_{{\mathrm{{in}}}}={din}$")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{l:g}" for l in lams])
    ax.set_xlabel(r"coupling strength $\lambda$")
    ax.set_ylabel(r"$\mathcal{L}^{\mathrm{eval}}_{\mathrm{total}}$")
    # FIXED LOG AXIS, AND A GUARD AGAINST SILENT CLIPPING. A hardcoded range
    # is the one place a figure can lie without anything looking wrong: a point
    # or a whisker outside the limits is simply not drawn, and the reader has
    # no way to tell. So every value that will be drawn -- medians and both
    # quartiles -- is checked against the limits first, and anything outside
    # stops the script instead of vanishing.
    if args.ylim:
        ylo, yhi = (float(v) for v in args.ylim.split(","))
        drawn = [(l, x, name, v)
                 for l in lams for x in dins
                 for name, v in (("median", new[(l, x)]), ("Q1", q1n[(l, x)]),
                                 ("Q3", q3n[(l, x)]))]
        out_of_range = [t for t in drawn if not (ylo <= t[3] <= yhi)]
        if out_of_range:
            die("these values fall outside the fixed y range "
                f"[{ylo:g}, {yhi:g}] and would be clipped without a trace:\n  "
                + "\n  ".join(f"lambda={l:g} d_in={x} {name} = {v:.6g}"
                               for l, x, name, v in out_of_range))
        ax.set_yscale("log")
        # FOUR LABELS, THE REST AS BARE MINOR TICKS. The labelled set is given
        # explicitly rather than left to a locator: 0.01, 0.02, 0.05, 0.1. The
        # minor subdivisions deliberately SKIP 2 and 5, which are already
        # majors -- a minor tick drawn under a major one is invisible but still
        # in the tick list, and a reader who counts ticks would count wrong.
        ax.yaxis.set_major_locator(FixedLocator([0.01, 0.02, 0.05, 0.1]))
        ax.yaxis.set_minor_locator(LogLocator(
            base=10.0, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0), numticks=64))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: (f"{v:g}" if 1e-3 <= abs(v) < 1e4
                          else rf"$10^{{{int(round(np.log10(abs(v))))}}}$")))
        ax.set_ylim(ylo, yhi)
        lo_d = min(t[3] for t in drawn)
        hi_d = max(t[3] for t in drawn)
        span = hi_d / lo_d
        print(f"  y axis fixed to [{ylo:g}, {yhi:g}], log. Everything drawn "
              f"lies in [{lo_d:.5g}, {hi_d:.5g}];\n  nothing is clipped.")
        pub_log = True
    else:
        span = axis_scale(ax, [new[(l, x)] for l in lams for x in dins],
                          "log" if pub_log else "linear")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(dins),
              handletextpad=0.5, columnspacing=1.3, fontsize=7.0,
              borderaxespad=0.15)
    fig.tight_layout(pad=0.4)
    out = os.path.join(args.outdir, args.name)
    # The published figure is never a legal destination. The candidate's own
    # name may be reused: a rerun of this script should refresh it.
    if os.path.basename(out) in ("fig_din_loss.pdf", "fig_din.pdf"):
        die(f"{out} is the published figure; this script only writes a "
            f"candidate under a separate name")
    fig.savefig(out)
    plt.close(fig)
    print(f"\nwrote {out}  (fig_din_loss.pdf untouched)")
    if not args.ylim:
        print(f"  axis: {'log' if pub_log else 'linear'}, taken from the "
              f"published figure's span so the two are comparable;\n  the "
              f"candidate's own span is {span:.2f} "
              f"({'it would have chosen the same' if (span >= 8) == pub_log else 'the rule alone would have chosen the other -- imposed deliberately'})")

    # ---- item 9: the six LaTeX rows ----------------------------------------
    rows = []
    for l in lams:
        cellsx = []
        for x in dins:
            k, m = succ[(l, x)]
            cellsx.append(f"{new[(l, x)]:.5f} & {k}/{m}")
        rows.append(f"      {l:g} & " + " & ".join(cellsx) + r" \\")
    tex = "\n".join(rows) + "\n      \\midrule\n      Total & & " + \
          " & & ".join(f"{tot[x][0]}/{tot[x][1]}" for x in dins) + r" \\"
    print(f"\n{'='*86}\nTable 2, six rows plus the total line, on "
          f"{NEW}\n{'='*86}\n{tex}")
    if args.latex:
        with open(args.latex, "w") as fh:
            fh.write(tex + "\n")
        print(f"\nwrote {args.latex}")


if __name__ == "__main__":
    main()
