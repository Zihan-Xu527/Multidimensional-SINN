"""Figure 10 candidate: one common y axis on all four panels.

Only the vertical axis changes.  The curves are read verbatim from
data/fig_corr_d3_unbiased.npz, which already carries the N(T-k) weighting;
no weight is applied here.  Checkpoint, colours, line styles, legends, fonts,
horizontal axis and the 2x2 layout are those of render_v5.py.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import (AutoMinorLocator, FixedLocator,
                               FormatStrFormatter, MultipleLocator,
                               NullLocator)

BASE, TITLE, SMALL = 8.5, 9.0, 7.5
AXW, LW = 0.6, 0.9
WIDTH = 6.02

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": BASE,
    "axes.titlesize": TITLE,
    "axes.labelsize": BASE,
    "xtick.labelsize": SMALL,
    "ytick.labelsize": SMALL,
    "legend.fontsize": SMALL,
    "axes.linewidth": AXW,
    "xtick.major.width": AXW,
    "ytick.major.width": AXW,
    "xtick.minor.width": AXW,
    "ytick.minor.width": AXW,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.top": False,
    "ytick.right": False,
    "lines.linewidth": LW,
    "legend.frameon": False,
    "legend.handlelength": 2.4,
    "legend.handletextpad": 0.8,
    "legend.columnspacing": 2.0,
    "legend.borderaxespad": 0.3,
    "pdf.fonttype": 3,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

BLUE, RED, GREEN = "#1f4e79", "#c0392b", "#1e8449"
ZERO_GREY = "0.8"
Q = lambda s: r"$\tilde{q}_{%d}$" % s

YLIM = (-0.2, 1.0)                      # common to all four panels


def panel_label(ax, tag):
    ax.text(0.03, 0.97, "(%s)" % tag, transform=ax.transAxes,
            ha="left", va="top", fontsize=BASE)


def minor(ax, xloc=None, yloc=None):
    ax.tick_params(which="minor", top=False, right=False, length=2.0,
                   width=AXW)
    ax.xaxis.set_minor_locator(xloc if xloc is not None else NullLocator())
    ax.yaxis.set_minor_locator(yloc if yloc is not None else NullLocator())


def figure10(src="data/fig_corr_d3_unbiased.npz", out="fig_corr_d3_commonylim",
             ymajor=0.2):
    d = np.load(src)
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.22))
    lag = d["lag"].astype(float)
    order = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    colors = {0: [BLUE, RED, GREEN], 1: [BLUE, RED], 2: [BLUE, RED],
              3: [BLUE, RED]}
    labels = {0: [Q(25), Q(50), Q(75)],
              1: [Q(25) + "," + Q(50), Q(50) + "," + Q(25)],
              2: [Q(25) + "," + Q(75), Q(75) + "," + Q(25)],
              3: [Q(50) + "," + Q(75), Q(75) + "," + Q(50)]}

    for bi, ax in enumerate(order):
        # every panel now spans negative values, so the zero line is drawn
        # in all four, panel (a) included
        ax.axhline(0.0, color=ZERO_GREY, lw=AXW, zorder=1)
        hs, hd = [], []
        for ci, col in enumerate(colors[bi]):
            r = d["panel%d_c%d_ref" % (bi, ci)]
            q = d["panel%d_c%d_sinn" % (bi, ci)]
            ax.plot(lag, r, color=col, lw=LW, ls="-", zorder=3)
            ax.plot(lag, q, color=col, lw=LW, ls="--", zorder=3)
            hs.append(Line2D([], [], color=col, lw=LW, ls="-"))
            hd.append(Line2D([], [], color=col, lw=LW, ls="--"))
        ax.set_xlim(0.0, 400.0)
        ax.set_ylim(*YLIM)
        ax.xaxis.set_major_locator(FixedLocator([0, 100, 200, 300, 400]))
        ax.yaxis.set_major_locator(MultipleLocator(ymajor))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        minor(ax, xloc=FixedLocator([50, 150, 250, 350]),
              yloc=AutoMinorLocator(2))
        ax.legend(hs + hd,
                  [l + " (target)" for l in labels[bi]]
                  + [l + " (SINN)" for l in labels[bi]],
                  ncol=2, loc="upper right", fontsize=SMALL,
                  frameon=False, handlelength=1.4, handletextpad=0.5,
                  labelspacing=0.3, columnspacing=1.0, borderaxespad=0.4)
        panel_label(ax, "abcd"[bi])
        if bi in (0, 1):
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r"lag index $k$")
        if bi in (0, 2):
            ax.set_ylabel(r"$C(k)$")

    fig.tight_layout(pad=0.4, w_pad=1.1, h_pad=0.7)
    for ext in ("pdf", "png"):
        fig.savefig("%s.%s" % (out, ext))
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(figure10())
    print(figure10(out="fig_corr_d3_commonylim_major04", ymajor=0.4))
