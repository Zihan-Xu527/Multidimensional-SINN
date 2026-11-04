#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
New Figure 4: validation loss against training iteration, with the weights.

WHAT EACH PANEL IS.

  (a) L_total^val = sum_s L_s^val against iteration, balancing off against on.
      Unweighted both times, so the two settings are compared on the same
      quantity rather than on each one's own objective.
  (b) the adaptive weights w_s = 1/(2 sigma_s^2) of the representative run.
      Same data as the published panel; recomputed here from that run's stored
      sigma so that one file feeds all three panels.
  (c) the five family losses L_s^val separately, on solid and off dashed, one
      colour per family.

(a) IS ALREADY ON DISK AND IS NOT RECONSTRUCTED. train_sinn.py stores

    agg = sum(e[f] ** 2 for f in fams)       # the selection scalar

which is exactly sum_s L_s^val over the applicable families, unweighted. So
panel (a) plots val_agg directly. Panel (c) plots val_errors[f] ** 2, from the
same records at the same steps, which makes the identity in (a) = sum over (c)
true by construction. The script still CHECKS it per run and per step, because
"by construction" is how the two-bases problem got into the paper.

WHAT THESE ARE NOT. Not training losses -- per-family training losses were
never written to the history, only their unweighted total, so a family-level
training curve cannot be drawn without retraining. Not validation ratios --
nothing here is divided by a reference loss. The axis labels say L^val for
that reason.

NOTE THE SAMPLING. Validation runs every 200 iterations, so these are 50-point
curves, not per-iteration ones.

Usage
-----
    python fig_valloss.py
    python fig_valloss.py --style representative
    python fig_valloss.py --lam 4 --outdir figs --dump fig4_histories.json
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
from matplotlib.ticker import MultipleLocator

WIDTH = 6.02
C_ON, C_OFF = "#1b4f72", "#a93226"   # as in Figure 5: blue on, red off

# THE PALETTE IS KEYED BY NAME, NOT BY POSITION. fig_lb.py assigns
# FAMC = [...] over sorted(w.items()), so the published panel (b) colours are
# the alphabetical order of the family names. Writing that mapping out means a
# family added or renamed later cannot silently recolour the others.
FAMC = {"ACF": "#1b4f72", "ACF2": "#a93226", "CCF": "#1e8449",
        "PDF": "#6c3483", "PROJ": "#b9770e"}
ORDER = ["ACF", "CCF", "ACF2", "PDF", "PROJ"]      # the manuscript's order


def lighten(hexcol, t=0.52):
    """Blend a colour towards white. Used for the balancing-off curves.

    WHY NOT LINE STYLE ALONE. Ten curves in five colours, on solid and off
    dashed, put the same hue twice in the same band: where the pair runs close
    together -- which is most of the panel for the three correlation families
    -- a reader cannot tell which of the two they are following. Hue carries
    the family and lightness carries the setting, so the two channels are
    independent and agree with the dash pattern rather than competing with it.
    It also survives greyscale printing, which dashes alone do not when the
    lines overlap.
    """
    c = np.array([int(hexcol[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    return tuple(c + (1.0 - c) * t)

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


def load(rundir, nf, din, lam, lb):
    """Every run of one cell: iteration grid, total, per-family, sigma."""
    tag = f"lam{lam:g}".replace(".", "p")
    pat = os.path.join(rundir, f"hist_nf{nf}_lb{lb}_din{din}_L*H*_{tag}_s*.npz")
    out = []
    for p in sorted(glob.glob(pat)):
        d = np.load(p, allow_pickle=False)
        ve = json.loads(str(d["val_errors"])) if "val_errors" in d.files else []
        if not ve:
            continue
        fams = [f for f in ORDER if f in ve[0]]
        run_tag = os.path.basename(p)[5:-4]
        # The run record, for its retained-checkpoint ratios: the family
        # spread needs the reference losses and the history does not carry
        # them. Absent record is not fatal here -- only the spread figure
        # needs it, and it says so if too few runs have one.
        rec_path = os.path.join(rundir, f"run_{run_tag}.json")
        rec = json.load(open(rec_path)) if os.path.exists(rec_path) else {}
        out.append({
            "tag": run_tag,
            "iter": np.asarray(d["iter"], dtype=float),
            "agg": np.asarray(d["val_agg"], dtype=float),
            # L_s^val = (rms error)^2, squared per step before anything else.
            "L": {f: np.array([r[f] ** 2 for r in ve], dtype=float)
                  for f in fams},
            "sigma": (json.loads(str(d["sigma"])) if "sigma" in d.files else []),
            "fams": fams,
            "best": rec.get("best", {}),
        })
    return out


def check_identity(runs, label):
    """val_agg must equal the sum of the family losses, step by step."""
    worst, n = 0.0, 0
    for r in runs:
        s = np.sum([r["L"][f] for f in r["fams"]], axis=0)
        m = np.isfinite(s) & np.isfinite(r["agg"]) & (np.abs(r["agg"]) > 0)
        if m.any():
            worst = max(worst, float(np.max(np.abs(s[m] - r["agg"][m])
                                            / np.abs(r["agg"][m]))))
        n += int(m.sum())
    print(f"  {label}: {len(runs)} runs, {n} validation steps, "
          f"|sum_s L_s - L_total| / L_total at most {worst:.3e}")
    return worst


def summarize(runs, getter, style_):
    """One curve (and optionally a band) from a set of runs."""
    grid = runs[0]["iter"]
    for r in runs:
        if r["iter"].shape != grid.shape or not np.allclose(r["iter"], grid):
            raise SystemExit(f"{r['tag']} has a different validation grid; "
                             f"the runs cannot be pooled step by step.")
    M = np.vstack([getter(r) for r in runs])
    if style_ == "median":
        return grid, np.median(M, axis=0), np.percentile(M, 25, axis=0), \
            np.percentile(M, 75, axis=0), M
    return grid, None, None, None, M


def representative(runs, repdir=None, nf=3, din=2, lam=4.0):
    """The run the published panel shows.

    PREFER THE REPORT'S OWN FIELD. efficiency.py built the published weights
    from the run named display_run in that coupling's report. This script
    could re-derive it from Section 4.1.3's rule -- order by validation score,
    take the larger of the two central runs -- and the two should agree,
    because display_run is set by that same rule on the same quantity. Should
    is not the same as does. So the report's field wins when it is available,
    the rule is the fallback, and a disagreement is printed rather than
    silently resolved: a panel (b) that shows a different run from the
    published one is exactly the kind of change that would pass unnoticed.
    """
    ranked = sorted(runs, key=lambda r: float(np.min(r["agg"])))
    by_rule = ranked[len(ranked) // 2]

    named = None
    if repdir:
        tag = f"lam{lam:g}".replace(".", "p")
        hits = glob.glob(os.path.join(
            repdir, f"report_nf{nf}_lbon_din{din}_L*H*_{tag}.json"))
        if hits:
            want = json.load(open(hits[0])).get("display_run")
            named = next((r for r in runs if r["tag"] == want), None)
            if want and named is None:
                print(f"  NOTE: {os.path.basename(hits[0])} names "
                      f"{want}, which has no history file here.")

    if named is None:
        print(f"  representative run from Section 4.1.3's rule: "
              f"{by_rule['tag']} (no report consulted)")
        return by_rule
    if named["tag"] != by_rule["tag"]:
        print(f"  WARNING: the report names {named['tag']} as the displayed "
              f"run, but Section 4.1.3's\n           rule applied here gives "
              f"{by_rule['tag']}. Using the report's, which is what the\n"
              f"           published panel shows -- but find out why they "
              f"differ before publishing.")
    else:
        print(f"  representative run: {named['tag']} "
              f"(report and Section 4.1.3's rule agree)")
    return named


def floors(runs):
    """sqrt(L_s^ref) per family, recovered from the retained checkpoints.

    Same arithmetic as fig_attain.py, and the same reason for it: the history
    stores the model's errors but not the reference loss. At the retained
    checkpoint the run record carries the error AND the ratio, and the stored
    ratios are on the superseded square-root scale, so

        sqrt(L_s^ref) = val_error[f] / ratios[f] ,

    with no square root on the right. The error is read from the history at
    best["iter"] rather than from best["val_errors"], which the older records
    do not carry. One coupling, so one set of floors: every run must agree,
    balancing on and off alike, and that is checked.
    """
    acc, used = {}, 0
    for r in runs:
        b = r["best"]
        ra, it_best = b.get("ratios"), b.get("iter")
        if not ra or it_best is None:
            continue
        j = int(np.argmin(np.abs(r["iter"] - float(it_best))))
        if abs(r["iter"][j] - float(it_best)) > 0.5:
            continue
        # This script stores L_s = error^2, so the error is its square root.
        # fig_attain.py keeps the raw errors and indexes them directly; the
        # arithmetic below is the same, one sqrt apart.
        for f in r["fams"]:
            if f in ra and np.isfinite(ra[f]) and ra[f] > 0:
                e = float(np.sqrt(r["L"][f][j]))
                acc.setdefault(f, []).append(e / float(ra[f]))
        used += 1
    if not acc:
        raise SystemExit("no run record carries both ratios and best['iter']; "
                         "the family spread cannot be computed.")
    out, worst = {}, 0.0
    for f, v in acc.items():
        v = np.asarray(v, dtype=float)
        med = float(np.median(v))
        out[f] = med
        if med > 0:
            worst = max(worst, float(np.max(np.abs(v - med)) / med))
    print(f"  reference losses from {used} runs, largest spread {worst:.3e}")
    if worst > 1e-6:
        raise SystemExit("runs disagree about the reference loss; stop.")
    return out


def band(ax, grid, M, color, label, ls="-"):
    """Median and interquartile band over runs."""
    ax.fill_between(grid, np.percentile(M, 25, axis=0),
                    np.percentile(M, 75, axis=0), color=color, alpha=0.16, lw=0)
    ax.plot(grid, np.median(M, axis=0), ls, color=color, label=label)


def standalone_total(on, off, outdir, lam):
    fig, ax = plt.subplots(1, 1, figsize=(WIDTH * 0.46, 2.2))
    for runs, lab, c in ((off, "balancing off", C_OFF),
                         (on, "balancing on", C_ON)):
        grid, _, _, _, M = summarize(runs, lambda r: r["agg"], "median")
        band(ax, grid, M, c, lab)
    ax.set_yscale("log")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"$\mathcal{L}^{\mathrm{val}}_{\mathrm{total}}$")
    ax.xaxis.set_major_locator(MultipleLocator(5000))
    ax.xaxis.set_minor_locator(MultipleLocator(2500))
    ax.legend(loc="upper right", handletextpad=0.5, labelspacing=0.25)
    fig.tight_layout(pad=0.4)
    p = os.path.join(outdir, f"fig4_total_lam{lam:g}.pdf")
    fig.savefig(p); plt.close(fig)
    print(f"wrote {p}")


def panel_spread(ax, on, off, fl):
    """max_s r_s^val / min_s r_s^val against iteration.

    This is the quantity Appendix C.2 reports in one column, drawn over the
    whole of training instead of at the retained checkpoint alone. It is a
    ratio of ratios, so the factor of nine between the correlation and density
    reference losses -- which has nothing to do with balancing and which makes
    the raw family losses sit in two bands -- divides out.
    """
    # THE SAME STATISTIC AS THE APPENDIX, NOT A COUSIN OF IT.
    #
    # Appendix C.2's spread is the largest family-level MEDIAN divided by the
    # smallest: the median over runs is taken first, per family, and the ratio
    # across families comes second. Taking max/min inside each run and then the
    # median over runs is a different number -- within-run scatter inflates
    # every per-run ratio, so the median of them sits above the ratio of the
    # medians, by an amount that depends on how noisy each setting is. Drawn
    # that way the curves even crossed in the middle of training, in the
    # opposite order to the table. Order of operations, again.
    #
    # There is no interquartile band here for the same reason: this quantity is
    # already a summary across runs, so there is no per-run value to spread.
    for runs, lab, c in ((off, "balancing off", C_OFF),
                         (on, "balancing on", C_ON)):
        grid = runs[0]["iter"]
        med = np.vstack([np.median(np.vstack([np.asarray(r["L"][f]) / fl[f] ** 2
                                              for r in runs]), axis=0)
                         for f in runs[0]["fams"] if f in fl and fl[f] > 0])
        ax.plot(grid, med.max(axis=0) / med.min(axis=0), "-", color=c,
                label=lab)
    # LOG. The first validation step sits above 100 -- the network has had 200
    # iterations and the families are nowhere near each other yet -- and on a
    # linear axis that opening transient sets the scale and squashes the whole
    # second half of training, which is the part the appendix reports, into the
    # bottom few percent of the panel. On a log axis the transient costs one
    # decade and the 3-to-5 region gets room.
    ax.set_yscale("log")
    ax.set_xlabel("Training iteration")
    # The formula lived on the axis and was both long and ambiguous about the
    # order of operations. A short name here; the caption carries the
    # definition: at each step, the across-run median of each family's ratio,
    # then the largest of those divided by the smallest.
    ax.set_ylabel("Family-level spread")
    ax.xaxis.set_major_locator(MultipleLocator(5000))
    ax.xaxis.set_minor_locator(MultipleLocator(2500))
    ax.legend(loc="upper right", handletextpad=0.5, labelspacing=0.25)
    # Headroom. The curves start at the top of the panel -- the first
    # validation step is the most unequal one -- and the panel label sits in
    # that corner, so the axis is opened above the data rather than letting
    # the label land on the first few points.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 2.0)


def standalone_spread(on, off, fl, outdir, lam):
    fig, ax = plt.subplots(1, 1, figsize=(WIDTH * 0.46, 2.2))
    panel_spread(ax, on, off, fl)
    fig.tight_layout(pad=0.4)
    p = os.path.join(outdir, f"fig4_spread_lam{lam:g}.pdf")
    fig.savefig(p); plt.close(fig)
    print(f"wrote {p}")


def standalone_families(on, off, style_, outdir, lam):
    """The five family losses, kept as an alternative to the spread panel."""
    fig, ax = plt.subplots(1, 1, figsize=(WIDTH * 0.46, 2.2))
    panel_c(ax, on, off, style_)
    fig.tight_layout(pad=0.4)
    p = os.path.join(outdir, f"fig4_families_lam{lam:g}.pdf")
    fig.savefig(p); plt.close(fig)
    print(f"wrote {p}")


def panel_a(ax, on, off, style_):
    for runs, lab, ls, col in ((off, "balancing off", "--", "0.45"),
                               (on, "balancing on", "-", "#1b4f72")):
        grid, med, lo, hi, M = summarize(runs, lambda r: r["agg"], style_)
        if style_ == "median":
            ax.fill_between(grid, lo, hi, color=col, alpha=0.16, lw=0)
            ax.plot(grid, med, ls, color=col, label=lab)
        elif style_ == "representative":
            r = representative(runs)
            ax.plot(r["iter"], r["agg"], ls, color=col, label=lab)
        else:
            for row in M:
                ax.plot(grid, row, ls, color=col, lw=0.5, alpha=0.5)
            ax.plot([], [], ls, color=col, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"$\mathcal{L}^{\mathrm{val}}_{\mathrm{total}}$")
    ax.xaxis.set_major_locator(MultipleLocator(5000))
    ax.xaxis.set_minor_locator(MultipleLocator(2500))
    ax.legend(loc="upper right", handletextpad=0.5, labelspacing=0.25)


def panel_weights(ax, runs):
    """w_s across ALL the balanced runs: median per family, interquartile band.

    WHY POOLED AND NOT THE ONE REPRESENTATIVE RUN. The two panels share an
    iteration axis, so a reader will read them against each other in time. A
    single run beside a twenty-run summary makes that reading false: the
    plateau would belong to one noise realisation and the spread to the
    population, and any alignment between them would be an artefact of two
    unrelated aggregations. Both panels are now the same balanced runs, so a
    feature at a given iteration in one refers to the same runs as a feature at
    that iteration in the other.

    WHAT THIS PANEL MAY NOT BE USED TO CLAIM. Not that the advantage begins
    once the weights settle. timing_lb.py measured both timings per seed on
    twenty matched pairs: which event comes first is decided by the tolerance
    chosen for "settled" -- the median difference moves from -1000 iterations
    at +-5% to +1300 at +-20%, its interquartile range straddles zero at every
    setting, six of twenty pairs never show a persistent advantage, and both
    events land in the last fifth of training. The ordering is not a property
    of the data. The band drawn here is what shows that: the trajectories are
    not tight enough across runs for a shared settling iteration.
    """
    grid = runs[0]["iter"]
    use = [r for r in runs if r["sigma"] and len(r["sigma"]) == len(r["iter"])
           and r["iter"].shape == grid.shape and np.allclose(r["iter"], grid)]
    if not use:
        ax.text(0.5, 0.5, "no sigma history", ha="center", va="center",
                transform=ax.transAxes)
        return
    if len(use) < len(runs):
        print(f"  NOTE: {len(runs) - len(use)} of {len(runs)} balanced runs "
              f"carry no usable sigma history and are left out of (a).")
    for f in ORDER:
        rows = [[1.0 / (2.0 * s[f] ** 2) for s in r["sigma"]]
                for r in use if f in r["sigma"][0]]
        if not rows:
            continue
        M = np.vstack(rows)
        ax.fill_between(grid, np.percentile(M, 25, axis=0),
                        np.percentile(M, 75, axis=0), color=FAMC[f],
                        alpha=0.15, lw=0)
        ax.plot(grid, np.median(M, axis=0), color=FAMC[f], label=f)
    ax.set_yscale("log")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"$w_s = 1/(2\sigma_s^{2})$")
    ax.xaxis.set_major_locator(MultipleLocator(5000))
    ax.xaxis.set_minor_locator(MultipleLocator(2500))
    ax.legend(loc="lower right", ncol=1, handletextpad=0.5, labelspacing=0.25)
    print(f"  (a) pooled over {len(use)} balanced runs")


def standalone_weights_run(run, outdir, lam):
    """The old single-run weights panel, kept as a figure of its own.

    Showing one run's weights with no averaging is a legitimate thing to do; it
    just belongs somewhere that does not invite a reader to line it up against
    a population-level curve.
    """
    fig, ax = plt.subplots(1, 1, figsize=(WIDTH * 0.46, 2.2))
    panel_b(ax, run)
    fig.tight_layout(pad=0.4)
    p = os.path.join(outdir, f"fig4_weights_run_lam{lam:g}.pdf")
    fig.savefig(p); plt.close(fig)
    print(f"wrote {p}  (run {run['tag']} alone, NOT the panel in fig4)")


def panel_b(ax, run):
    if not run["sigma"]:
        ax.text(0.5, 0.5, "no sigma stored", ha="center", va="center",
                transform=ax.transAxes)
        return
    for f in ORDER:
        if f not in run["sigma"][0]:
            continue
        w = np.array([1.0 / (2.0 * s[f] ** 2) for s in run["sigma"]])
        ax.plot(run["iter"], w, color=FAMC[f], label=f)
    ax.set_yscale("log")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"$w_s = 1/(2\sigma_s^{2})$")
    ax.xaxis.set_major_locator(MultipleLocator(5000))
    ax.xaxis.set_minor_locator(MultipleLocator(2500))
    ax.legend(loc="lower right", ncol=1, handletextpad=0.5, labelspacing=0.25)


def panel_c(ax, on, off, style_):
    # SAME RUNS AND SAME SUMMARY AS (a). No bands here: ten overlapping
    # interquartile ribbons on one axis cannot be read, and the point of (c)
    # is which family sits where, not the run-to-run spread.
    drawn = []
    for f in ORDER:
        for runs, ls in ((on, "-"), (off, "--")):
            if f not in runs[0]["L"]:
                continue
            grid, med, _, _, M = summarize(runs, lambda r, f=f: r["L"][f],
                                           "median")
            y = med if style_ != "representative" else representative(runs)["L"][f]
            drawn.append(np.asarray(y, dtype=float))
            col = FAMC[f] if ls == "-" else lighten(FAMC[f])
            ax.plot(grid, y, ls, color=col, lw=1.0,
                    label=f if ls == "-" else None)
    ax.set_yscale("log")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"$\mathcal{L}^{\mathrm{val}}_{s}$")
    ax.xaxis.set_major_locator(MultipleLocator(5000))
    ax.xaxis.set_minor_locator(MultipleLocator(2500))

    # NO FAMILY KEY HERE. Panel (a) already names the five colours and this
    # panel uses the same mapping, so repeating it spent the only free corner
    # on information the reader has just been given. What this panel does add
    # is the line style, and that key goes where the family key used to be:
    # the curves fall left to right, so the upper right is empty.
    h = [plt.Line2D([], [], color="0.30", ls="-"),
         plt.Line2D([], [], color=lighten("#4d4d4d"), ls="--")]
    ax.legend(h, ["on", "off"], loc="upper right", handletextpad=0.5,
              labelspacing=0.22)

    # Headroom at the top for the panel label and the key, and a little air
    # underneath. Computed from the curves actually drawn, so a rerun on
    # different numbers cannot clip them.
    lo = min(v for col in drawn for v in col if np.isfinite(v) and v > 0)
    hi = max(v for col in drawn for v in col if np.isfinite(v))
    ax.set_ylim(lo / 1.6, hi * 3.2)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rundir", default="runs/grid")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--repdir", default="reports_val",
                    help="only to read display_run for panel (b)")
    ap.add_argument("--lam", type=float, default=4.0)
    ap.add_argument("--nf", type=int, default=3, help="file-name convention")
    ap.add_argument("--din", type=int, default=2)
    ap.add_argument("--style", choices=("median", "representative", "all"),
                    default="median",
                    help="how (a) and (c) summarize the runs")
    ap.add_argument("--dump", default="fig4_histories.json")
    args = ap.parse_args()

    style()
    os.makedirs(args.outdir, exist_ok=True)

    on = load(args.rundir, args.nf, args.din, args.lam, "on")
    off = load(args.rundir, args.nf, args.din, args.lam, "off")
    if not on or not off:
        raise SystemExit(f"need both balancing settings at lambda={args.lam:g} "
                         f"in {args.rundir}; found {len(on)} on, {len(off)} off")

    print(f"lambda = {args.lam:g}, n_f = {args.nf + 1} (file nf{args.nf}), "
          f"d_in = {args.din}, summary = {args.style}")
    print(f"families present: {on[0]['fams']}")
    print("identity check, L_total^val vs the sum of the family losses:")
    w1 = check_identity(on, "balancing on ")
    w2 = check_identity(off, "balancing off")
    if max(w1, w2) > 1e-6:
        raise SystemExit("the stored total is not the sum of the stored family "
                         "losses; panels (a) and (c) would disagree. Stop.")

    rep = representative(on, args.repdir, args.nf, args.din, args.lam)
    if args.style == "all":
        print("NOTE: --style all draws every run in (a), but (c) stays on the "
              "median:\n      five families times two settings times "
              f"{len(on) + len(off)} runs is unreadable. (a) and (c) are "
              "therefore\n      NOT summarized the same way under this style. "
              "Use median or representative\n      for the figure that goes in "
              "the paper.")

    # ---- the figure: weights, then the family losses ------------------------
    #
    # THE TOTAL-LOSS PANEL IS GONE. It was dropped after being drawn at all six
    # couplings and separating balancing-off from balancing-on at none of them.
    # That is not a plotting failure: the unweighted total is about four fifths
    # correlation families, and balancing improves the two density families
    # while making the correlation families slightly worse, so the sum barely
    # moves. The totals are still printed and dumped below, because "the totals
    # are comparable" is a result worth stating -- it just does not need a
    # third of the figure to say it.
    fl = floors(on + off)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH * 0.68, 2.15))
    panel_weights(axes[0], on)
    panel_spread(axes[1], on, off, fl)

    # ABOVE THE AXES, NOT INSIDE. Figure 5 puts its panel labels in the
    # upper-left corner, which works there because its legend sits outside the
    # axes. Both panels here carry their own legend -- they show different
    # series, so one shared key is not possible -- and an in-panel label was
    # overdrawn by the legend.
    #
    # LABELLED BY POSITION, NOT BY SOURCE. The weights panel used to be (b) of
    # three; it is (a) of two now, and reading the label off the old letter
    # would have printed (b) and (c) on a two-panel figure.
    for i, ax in enumerate(axes):
        ax.text(0.035, 0.965, f"({chr(97 + i)})", transform=ax.transAxes,
                fontsize=8.5, ha="left", va="top")
    fig.tight_layout(pad=0.4)
    path = os.path.join(args.outdir, f"fig4_lam{args.lam:g}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
    standalone_total(on, off, args.outdir, args.lam)
    standalone_spread(on, off, fl, args.outdir, args.lam)
    standalone_families(on, off, args.style, args.outdir, args.lam)
    standalone_weights_run(rep, args.outdir, args.lam)

    print(f"  (a) the adaptive weights over all {len(on)} balanced runs: "
          f"per family, the across-run\n      median with its interquartile "
          f"band.\n  (b) the family-level spread: at each validation step, the "
          f"across-run median of\n      each family's ratio, then the largest "
          f"of those divided by the smallest,\n      over the same {len(on)} "
          f"balanced runs and {len(off)} unbalanced ones.\n"
          "  Both panels are the same runs on the same axis, so they may be "
          "read against\n  each other -- but NOT as a sequence of events. See "
          "panel_weights' docstring\n  and timing_lb.py: which of the two "
          "events comes first is set by the tolerance\n  chosen for "
          "'settled', not by the data. (b) is a ratio and (a) is a weight, so "
          "the\n  two vertical axes are still unrelated.")

    # ---- the numbers ---------------------------------------------------------
    print(f"\nL_total^val at the last validation step, median [IQR] over runs")
    for runs, lab in ((off, "off"), (on, "on ")):
        v = [float(r["agg"][-1]) for r in runs]
        print(f"  balancing {lab}: {np.median(v):.4e}  "
              f"[{np.percentile(v, 25):.4e}, {np.percentile(v, 75):.4e}]  "
              f"n={len(v)}")
    print(f"\nL_s^val at the last validation step, median over runs")
    print(f"  {'family':<8}{'off':>12}{'on':>12}{'on/off':>9}")
    for f in ORDER:
        if f not in on[0]["L"]:
            continue
        a = float(np.median([r["L"][f][-1] for r in off]))
        b = float(np.median([r["L"][f][-1] for r in on]))
        print(f"  {f:<8}{a:>12.4e}{b:>12.4e}{b / a:>9.2f}")

    if args.dump:
        path = os.path.join(args.outdir, args.dump)
        with open(path, "w") as fh:
            json.dump({
                "lam": args.lam, "n_f_manuscript": args.nf + 1,
                "d_in": args.din, "representative_run": rep["tag"],
                "note": ("L_total^val is train_sinn.py's val_agg, the unweighted "
                         "sum of the family validation losses; L_s^val is the "
                         "stored rms error squared. Neither is a ratio."),
                "iter": on[0]["iter"].tolist(),
                "runs": {lb: [{"tag": r["tag"],
                               "L_total": r["agg"].tolist(),
                               "L": {f: r["L"][f].tolist() for f in r["fams"]}}
                              for r in runs]
                         for lb, runs in (("on", on), ("off", off))},
            }, fh)
        print(f"\nwrote {path}  (every run's full history, both settings)")


if __name__ == "__main__":
    main()
