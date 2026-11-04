"""Shared plotting style for the Section 5.1 tau10 refresh.

Constants are taken verbatim from the Section 4 figure scripts
(toolbox/Duffing_dimer/fig_joint.py, fig_din.py) so that the new figures are
typographically identical to the ones already in the manuscript.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH = 6.02                       # \textwidth in inches, Section 4 value

FAMILIES = ("ACF", "CCF", "ACF2", "PDF", "PROJ")

# Family-ratio figure: monochrome on purpose.  The horizontal axis already
# names the five families, so colouring the boxes by family would encode the
# same variable twice.  The only distinction the marks have to carry is
# SINN against reference, and accepted against rejected.
SINN_FACE = "white"                # SINN boxes, accepted runs only
SINN_EDGE = "black"                # box edge, median, whiskers, caps
REF_FACE = "0.86"                  # reference-block boxes, light gray
REF_EDGE = "0.45"                  # their edges and their open circles
MARK = "black"                     # accepted circles and rejected squares

# Curve and contour figures: here colour is not redundant.  Line style
# already separates reference from SINN, and these distinguish the three
# site components, which no axis labels.
REF_LINE = "#1b4f72"               # target / final-evaluation contours
REF_FILL = "#eaf1f8"
SINN_LINE = "#a93226"              # SINN contours


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "lines.linewidth": 0.9,
        "axes.grid": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def panel(ax, letter):
    ax.text(0.03, 0.97, "(%s)" % letter, transform=ax.transAxes,
            ha="left", va="top", fontsize=8.5)


def jitter(key, amp=0.075):
    """Deterministic offset keyed by seed or block index; never random."""
    import numpy as np
    return float(np.random.default_rng(int(key) + 10_000).uniform(-amp, amp))
