"""Denim-themed matplotlib palette + stylesheet.

The aesthetic is lifted from the CS7642 Project 4 report: denim primary,
amber complement, plum/sage accents, parchment background. Used by every
figure-emitting module so the IEEE PDF reads as one continuous voice with
the R sanity-check Rmd.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- Palette -------------------------------------------------------------
DENIM        = "#2C6FBB"  # primary
DENIM_DARK   = "#1B3A5C"  # axes, headers
DENIM_MID    = "#3A6B8C"  # secondary structure
DENIM_BRIGHT = "#5B9BD5"  # highlight
DENIM_ROW    = "#E6EEF5"  # zebra-striped table rows
AMBER        = "#D4A017"  # complementary
TERRACOTTA   = "#E07B39"
PLUM         = "#7B2D8E"
PLUM_COMP    = "#408E2D"
SAGE         = "#6BB38A"
SIENNA       = "#B8552E"
RUST         = "#9C4A2A"
PARCHMENT    = "#F4EFE5"
SUCCESS      = "#2CA02C"
FAILURE      = "#D62728"

# Ordered palette for categorical encodings (5 model families, 7 classes).
MODEL_PALETTE = {
    "dt":     DENIM_DARK,
    "knn":    AMBER,
    "svm":    PLUM,
    "mlp_sk": SAGE,
    "mlp_pt": DENIM_BRIGHT,
    # Pretty-name aliases for tables/figures that use display labels.
    "DecisionTree": DENIM_DARK,
    "kNN":          AMBER,
    "SVM":          PLUM,
    "MLP-sklearn":  SAGE,
    "MLP-torch":    DENIM_BRIGHT,
}

# Cover Type class palette — perceptually ordered by tree-cover density.
CLASS_PALETTE = [
    "#1B3A5C",  # 1 Spruce/Fir
    "#2C6FBB",  # 2 Lodgepole Pine
    "#6BB38A",  # 3 Ponderosa Pine
    "#D4A017",  # 4 Cottonwood/Willow
    "#E07B39",  # 5 Aspen
    "#7B2D8E",  # 6 Douglas-fir
    "#9C4A2A",  # 7 Krummholz
]

CLASS_NAMES = [
    "Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine",
    "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz",
]


def apply_style() -> None:
    """Install the project-wide matplotlib rcParams.

    Idempotent — safe to call from any entry point. Optimised for IEEE
    two-column figures (small fonts, tight padding).
    """
    mpl.rcParams.update({
        "figure.facecolor":      "white",
        "axes.facecolor":        "white",
        "axes.edgecolor":        DENIM_DARK,
        "axes.linewidth":        0.7,
        "axes.labelcolor":       DENIM_DARK,
        "axes.titlesize":        9.0,
        "axes.titleweight":      "bold",
        "axes.titlecolor":       DENIM_DARK,
        "axes.labelsize":        8.0,
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "axes.prop_cycle":       mpl.cycler(color=[
            DENIM, AMBER, PLUM, SAGE, SIENNA, DENIM_BRIGHT, RUST,
        ]),
        "xtick.color":           DENIM_DARK,
        "ytick.color":           DENIM_DARK,
        "xtick.labelsize":       7.0,
        "ytick.labelsize":       7.0,
        "legend.fontsize":       7.0,
        "legend.frameon":        False,
        "legend.title_fontsize": 7.5,
        "grid.color":            DENIM_ROW,
        "grid.linewidth":        0.5,
        "grid.alpha":            0.8,
        "lines.linewidth":       1.3,
        "lines.markersize":      3.5,
        "font.family":           "DejaVu Sans",
        "font.size":             8.0,
        "figure.dpi":            150,
        "savefig.dpi":           300,
        "savefig.bbox":          "tight",
        "savefig.pad_inches":    0.04,
    })


def shaded_band(ax, x, mean, std, color, alpha: float = 0.18, label=None):
    """Mean line with ±1 SD shaded band — used everywhere for LC/MC curves."""
    line, = ax.plot(x, mean, color=color, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha,
                    linewidth=0)
    return line
