#!/usr/bin/env python3
"""Draw the four diagnostic figures from `llv_diag_12k.py`'s saved arrays.

Four figures, one pair per segment:

    diag12k_correlation_in.pdf/.png    scored,       retained t = 1..400
    diag12k_correlation_out.pdf/.png   continuation, retained t = 401..800
    diag12k_density_in.pdf/.png
    diag12k_density_out.pdf/.png

The style is imported from `llv_figures`, not copied, so these match the
manuscript figures exactly: black solid target, red dashed SINN, the four
panel titles, the shared vertical range [-1, 1] with majors at -1, -0.5, 0,
0.5, 1 and minors between them, and for the densities the same four-step blue
ramp, the same contour levels at 0.2/0.4/0.6/0.8 of the target peak, the same
display windows and equal aspect.

DIAGNOSTIC ONLY. These are separate files with their own names; nothing here
writes into `figures_llv`, and the manuscript figures are not regenerated.

The curves are read exactly as saved: `llv_diag_12k.py` computed them with
the pair-count estimator directly, so this script applies NO T/(T-k) factor.

usage:
    python3 llv_diag_figures.py --diag-dir diag12k_llv --out diag12k_llv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import llv_figures as F
import llv_spec as S

SEGMENTS = (("in", "scored", (1, 400)),
            ("out", "continuation", (401, 800)))


def correlation_figure(z, key, label, out_dir, plt):
    from matplotlib.lines import Line2D

    lag = np.asarray(z["lags"], np.int64)
    tgt = np.asarray(z["curve_target_" + key], np.float64)
    sinn = np.asarray(z["curve_sinn_" + key], np.float64)
    names = [str(x) for x in z["curve_names"]]
    c_t, c_s = F.SERIES_INK["target"], F.SERIES_INK["sinn"]

    fig, axes = plt.subplots(2, 2, figsize=(F.STYLE["width_full"], 3.6),
                             sharex=True)
    extremes = {}
    for i, (ax, nm) in enumerate(zip(axes.ravel(), names)):
        ax.plot(lag, tgt[i], color=c_t, ls=F.STYLE["target_ls"],
                lw=F.STYLE["line_lw"])
        ax.plot(lag, sinn[i], color=c_s, ls=F.STYLE["sinn_ls"],
                lw=F.STYLE["line_lw"])
        ax.axhline(0.0, color="0.8", lw=0.5, zorder=0)
        ax.set_title(F.PANEL_TITLES.get(nm, nm), pad=3)
        ax.set_xlim(0, int(lag[-1]) + 1)
        ax.set_xticks(F.SHORTLAG_XTICKS_MAJOR)
        ax.set_xticks(F.SHORTLAG_XTICKS_MINOR, minor=True)
        ax.set_ylim(*F.SHORTLAG_YLIM)
        ax.set_yticks(F.SHORTLAG_YTICKS_MAJOR)
        ax.set_yticks(F.SHORTLAG_YTICKS_MINOR, minor=True)
        ax.tick_params(which="minor", labelbottom=False, labelleft=False,
                       **F.STYLE["minor_tick"])
        F.panel_label(ax, i)
        if i >= 2:
            ax.set_xlabel(r"lag index $k$")
        if i % 2 == 0:
            ax.set_ylabel("correlation")
        extremes[nm] = (float(min(tgt[i].min(), sinn[i].min())),
                        float(max(tgt[i].max(), sinn[i].max())))
    tol = F.SHORTLAG_YLIM_TOL
    out = {nm: v for nm, v in extremes.items()
           if v[0] < F.SHORTLAG_YLIM[0] - tol
           or v[1] > F.SHORTLAG_YLIM[1] + tol}
    if out:
        raise SystemExit("the fixed range %s would clip %s"
                         % (list(F.SHORTLAG_YLIM),
                            ", ".join("%s (%.12g, %.12g)" % (nm, v[0], v[1])
                                      for nm, v in sorted(out.items()))))
    proxies = [Line2D([], [], color=c_t, ls=F.STYLE["target_ls"],
                      lw=F.STYLE["line_lw"]),
               Line2D([], [], color=c_s, ls=F.STYLE["sinn_ls"],
                      lw=F.STYLE["line_lw"])]
    fig.legend(proxies, ["target", "SINN"], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.045))
    fig.tight_layout()
    paths = {}
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, "diag12k_correlation_%s.%s" % (key, ext))
        fig.savefig(p)
        paths[ext] = p
    plt.close(fig)
    return {"paths": paths, "panel_extremes": extremes, "clipped": False}


def density_figure(z, key, label, out_dir, plt):
    from matplotlib.lines import Line2D

    ga = np.asarray(z["grid_rho_A"], np.float64)
    gb = np.asarray(z["grid_rho_B"], np.float64)
    mu = np.asarray(z["mu"], np.float64)
    sd = np.asarray(z["s"], np.float64)
    f_t = np.asarray(z["f2d_rho_target_" + key], np.float64)
    f_s = np.asarray(z["f2d_rho_sinn_" + key], np.float64)

    def phys(sigma, i):
        return (float(-sigma * sd[i] + mu[i]), float(sigma * sd[i] + mu[i]))

    def inwin(grid, lim):
        g = np.asarray(grid, np.float64)
        return (g >= lim[0]) & (g <= lim[1])

    sj = float(F.STYLE["joint_display_sigma"])
    sm = float(F.STYLE["marginal_display_sigma"])
    xlim, ylim = phys(sj, 0), phys(sj, 1)
    win = np.ix_(inwin(ga, xlim), inwin(gb, ylim))
    f_max = float(f_t[win].max())
    levels = [f * f_max for f in F.STYLE["contour_levels"]]
    fill_levels = levels + [float(max(f_max, f_t.max()))]

    height = F.DENSITY_HEIGHT
    bot, top = 0.185, 0.975
    h = top - bot
    aspect = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    w_a = (h * height / aspect) / F.STYLE["width_full"]
    left_a, gap, right_edge = 0.082, 0.095, 0.995
    fig = plt.figure(figsize=(F.STYLE["width_full"], height))
    left_b = left_a + w_a + gap
    X, Y = np.meshgrid(ga, gb, indexing="ij")

    ax = fig.add_axes([left_a, bot, w_a, h])
    ax.contourf(X, Y, f_t, levels=fill_levels, colors=F.STYLE["contour_ramp"])
    ax.contour(X, Y, f_t, levels=levels, colors=F.SERIES_INK["target"],
               linewidths=F.STYLE["contour_lw"])
    ax.contour(X, Y, f_s, levels=levels, colors=F.SERIES_INK["sinn"],
               linewidths=F.STYLE["contour_lw"], linestyles="dashed")
    ax.set_xlabel(r"$\rho_A$")
    ax.set_ylabel(r"$\rho_B$")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_yticks(F.DENSITY_B_YTICKS_MAJOR)
    ax.set_yticks(F.DENSITY_B_YTICKS_MINOR, minor=True)
    ax.tick_params(axis="y", which="minor", labelleft=False,
                   **F.STYLE["minor_tick"])
    ax.set_aspect("equal", adjustable="box")
    ax.legend([Line2D([], [], color=F.SERIES_INK["target"], ls="-",
                      lw=F.STYLE["contour_lw"]),
               Line2D([], [], color=F.SERIES_INK["sinn"], ls="--",
                      lw=F.STYLE["contour_lw"])],
              ["target", "SINN"], loc="upper right", handlelength=2.4)
    F.panel_label(ax, 0)

    ax = fig.add_axes([left_b, bot, right_edge - left_b, h])
    marg_lims = {c: phys(sm, i) for c, i in (("A", 0), ("B", 1))}
    union = (min(l[0] for l in marg_lims.values()),
             max(l[1] for l in marg_lims.values()))
    peak = 0.0
    for comp, grid, i in (("A", ga, 0), ("B", gb, 1)):
        c = F.RHO_COLORS[comp]
        lim = marg_lims[comp]
        mt = np.asarray(z["marg_rho_%s_target_%s" % (comp, key)], np.float64)
        ms = np.asarray(z["marg_rho_%s_sinn_%s" % (comp, key)], np.float64)
        k = inwin(grid, lim)
        g = np.asarray(grid, np.float64)[k]
        ax.plot(g, mt[k], color=c, ls=F.STYLE["target_ls"],
                lw=F.STYLE["line_lw"])
        ax.plot(g, ms[k], color=c, ls=F.STYLE["sinn_ls"],
                lw=F.STYLE["line_lw"])
        peak = max(peak, float(mt[k].max()), float(ms[k].max()))
    ax.set_xlabel(r"population density $\rho$")
    ax.set_ylabel(r"$f_\rho$")
    ax.set_xlim(*union)
    ax.set_ylim(0.0, 1.34 * peak)
    handles, labels = [], []
    for ls, src in ((F.STYLE["target_ls"], "target"),
                    (F.STYLE["sinn_ls"], "SINN")):
        for comp in ("A", "B"):
            handles.append(Line2D([], [], color=F.RHO_COLORS[comp], ls=ls,
                                  lw=F.STYLE["line_lw"]))
            labels.append(r"$\rho_%s$ (%s)" % (comp, src))
    ax.legend(handles, labels, ncol=2, loc="upper right",
              bbox_to_anchor=(1.0, 1.0), handlelength=1.7,
              columnspacing=1.0, labelspacing=0.35, handletextpad=0.45,
              borderaxespad=0.0)
    F.panel_label(ax, 1)

    paths = {}
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, "diag12k_density_%s.%s" % (key, ext))
        fig.savefig(p)
        paths[ext] = p
    plt.close(fig)
    return {"paths": paths, "contour_levels": levels,
            "target_peak_in_window": f_max}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag-dir", default="diag12k_llv")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out_dir = args.out or args.diag_dir
    os.makedirs(out_dir, exist_ok=True)

    plt = F.apply_style()
    zc = np.load(os.path.join(args.diag_dir, "diag12k_curves.npz"),
                 allow_pickle=True)
    zd = np.load(os.path.join(args.diag_dir, "diag12k_density.npz"))
    with open(os.path.join(args.diag_dir, "diag12k.json")) as fh:
        doc = json.load(fh)

    print("diagnostic figures, N = %d, both segments" % doc["ensemble"]["N"])
    print("DIAGNOSTIC ONLY -- separate filenames; figures_llv is untouched\n")
    rec = {}
    for key, role, (t_lo, t_hi) in SEGMENTS:
        label = "%s, retained t = %d..%d" % (role, t_lo, t_hi)
        c = correlation_figure(zc, key, label, out_dir, plt)
        d = density_figure(zd, key, label, out_dir, plt)
        rec[key] = {"correlation": c, "density": d, "label": label}
        print("%-13s -> diag12k_correlation_%s.pdf/.png, "
              "diag12k_density_%s.pdf/.png" % (role, key, key))
        print("   panel extrema %s"
              % ", ".join("%s %+.4f..%+.4f" % (nm, v[0], v[1])
                          for nm, v in c["panel_extremes"].items()))
        print("   contour levels %s (target peak %.4g)"
              % (["%.1f" % v for v in d["contour_levels"]],
                 d["target_peak_in_window"]))

    p = os.path.join(out_dir, "diag12k_figures.json")
    with open(p, "w") as fh:
        json.dump({"diagnostic_only": True,
                   "wrote_into_figures_llv": False,
                   "style_imported_from": "llv_figures",
                   "pair_count_conversion_applied_here": False,
                   "source_sha256": {n: S.sha256_file(n) for n in
                                     ("llv_figures.py", "llv_diag_figures.py")
                                     if os.path.exists(n)},
                   "segments": rec}, fh, indent=2, default=str)
    print("\nwrote %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
