"""Re-render Figures 10 and 11 in the Section 4 style.

The arrays come from `extract_arrays.py`, which read them back out of the
accepted PDFs; nothing is recomputed.  Panel order, layouts, axis ranges,
curves, contours, contour levels, colours, legends and line-style meanings are
carried over unchanged.  Only the style is new.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import (AutoMinorLocator, FixedLocator,
                               FormatStrFormatter,
                               MultipleLocator, NullLocator)

# ------------------------------------------------------------- Section 4 style
BASE, TITLE, SMALL = 8.5, 9.0, 7.5
LEG10 = 8.5        # Figure 10 in-panel component legends
LEGTOP = SMALL     # shared target/SINN legend, both figures
AXLAB11 = 9.0      # Figure 11 x/y axis labels only
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
LEGEND_GREY = "0.35"
ZERO_GREY = "0.8"
RAMP = ["#eaf1f8", "#cfe0ef", "#aecde4", "#8ab8d8"]
Q = lambda s: r"$\tilde{q}_{%d}$" % s


def panel_label(ax, tag):
    ax.text(0.03, 0.97, "(%s)" % tag, transform=ax.transAxes,
            ha="left", va="top", fontsize=BASE)


def spines(ax):
    ax.tick_params(which="both", top=False, right=False)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())


def minor(ax, xloc=None, yloc=None):
    """Minor ticks at the matplotlib default length, bottom and left only."""
    ax.tick_params(which="minor", top=False, right=False, length=2.0,
                   width=AXW)
    ax.xaxis.set_minor_locator(xloc if xloc is not None else NullLocator())
    ax.yaxis.set_minor_locator(yloc if yloc is not None else NullLocator())


# ---------------------------------------------------------------- Figure 10
def figure10(src="data/fig_corr_d3_unbiased.npz",
             out="fig_corr_d3.pdf"):
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
    # Three to four major ticks per panel, one minor between each pair.
    # Panel (c) spans only ~0.43, so a 0.2 step needs its top lifted to the
    # next major for a third tick to appear; the alternative, a 0.1 step,
    # puts nine ticks on the shortest axis in the figure.
    ystep = {0: 0.5, 1: 0.2, 2: 0.2, 3: 0.2}
    ytop = {2: 0.40}

    for bi, ax in enumerate(order):
        if bi:
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
        vals = np.concatenate([d["panel%d_c%d_%s" % (bi, c, w)]
                               for c in range(len(colors[bi]))
                               for w in ("ref", "sinn")])
        lo, hi = min(0.0, vals.min()), vals.max()
        pad = 0.06 * (hi - lo)
        ax.set_ylim(lo - pad, ytop.get(bi, hi + pad))
        ax.xaxis.set_major_locator(FixedLocator([0, 100, 200, 300, 400]))
        ax.yaxis.set_major_locator(MultipleLocator(ystep[bi]))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        minor(ax, xloc=FixedLocator([50, 150, 250, 350]),
              yloc=AutoMinorLocator(2))
        # Two columns: solid handles labelled (target) on the left, the
        # same colours dashed and labelled (SINN) on the right, so the line
        # style is read off the panel itself.
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

    # No shared target/SINN legend: solid = target, dashed = SINN is stated
    # in the caption instead.
    fig.tight_layout(pad=0.4, w_pad=1.1, h_pad=0.7)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- Figure 11
def figure11(src="data/fig_joint_d3_arrays.npz", out="fig_joint_d3.pdf"):
    d = np.load(src)
    pairs = [(25, 50), (25, 75), (50, 75)]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.12))
    for bi, ax in enumerate(axes):
        for li in range(4):                       # outermost level first
            p = d["panel%d_ref%d" % (bi, li)]
            ax.fill(p[:, 0], p[:, 1], facecolor=RAMP[li], edgecolor="none",
                    zorder=1 + li)
        for li in range(4):
            p = d["panel%d_ref%d" % (bi, li)]
            ax.plot(p[:, 0], p[:, 1], color="k", lw=LW, ls="-", zorder=6)
        for li in range(4):
            p = d["panel%d_sinn%d" % (bi, li)]
            ax.plot(p[:, 0], p[:, 1], color=RED, lw=LW, ls="--", zorder=7)
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-3.6, 3.6)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.set_box_aspect(1)
        minor(ax, xloc=MultipleLocator(1), yloc=MultipleLocator(1))
        ax.set_xlabel(Q(pairs[bi][0]), fontsize=AXLAB11)
        ax.set_ylabel(Q(pairs[bi][1]), fontsize=AXLAB11)
        panel_label(ax, "abc"[bi])

    # each handle in the colour of the contour it stands for
    fig.legend(handles=[Line2D([], [], color="k", ls="-", lw=LW),
                        Line2D([], [], color=RED, ls="--", lw=LW)],
               labels=["target", "SINN"], ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=LEGTOP)
    fig.tight_layout(pad=0.4, w_pad=1.4, rect=[0, 0, 1, 0.945])
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(figure10())
    print(figure11())
