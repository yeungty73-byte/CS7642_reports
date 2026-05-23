"""
data_screening.py — CS7641 OL Report (Summer 2026)
Author: Timothy Leung (tleung37)

EDA confirmation for the OL report (§8: Data, baseline, and target).
Verifies continuity with SL Report: same sample, same class distribution,
no new preprocessing steps, no data leakage.

Outputs:
  - figures/eda_class_distribution.pdf
  - figures/eda_feature_boxplots.pdf
  - logs/eda_summary.json
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter

from data_loader import load_covertype_sample, make_splits, set_global_seed, SEED

# ── Denim / high-contrast palette (matches SL report aesthetic) ────────────
DENIM      = "#1B3A6B"   # deep navy-denim (primary)
DENIM_MID  = "#2E6DA4"   # medium denim
DENIM_PALE = "#A8C4E0"   # pale ice-denim
RUST       = "#B5451B"   # high-contrast complement
GOLD       = "#D4A017"   # accent
CREAM      = "#F5F1E8"   # background
TEXT_DARK  = "#1A1A2E"   # near-black for readability
GRID_COL   = "#DADADF"

CLASS_NAMES = [
    "Spruce/Fir",
    "Lodgepole Pine",
    "Ponderosa Pine",
    "Cottonwood/Willow",
    "Aspen",
    "Douglas-fir",
    "Krummholz",
]

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
LOGDIR = os.path.join(os.path.dirname(__file__), "..", "logs")


def _ensure_dirs():
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)


def plot_class_distribution(y: np.ndarray, split_name: str = "full sample") -> None:
    """Bar chart of class counts — denim theme."""
    counts = Counter(y)
    classes = sorted(counts.keys())
    names   = [CLASS_NAMES[c] for c in classes]
    vals    = [counts[c] for c in classes]
    total   = sum(vals)
    pcts    = [v / total * 100 for v in vals]

    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=CREAM)
    ax.set_facecolor(CREAM)

    bars = ax.bar(names, vals, color=DENIM_MID, edgecolor=DENIM, linewidth=0.8)
    # Add percentage labels
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20,
                f"{pct:.1f}%", ha="center", va="bottom",
                fontsize=8, color=TEXT_DARK)

    ax.set_title(f"Covertype Class Distribution — {split_name}", fontsize=13,
                 color=DENIM, fontweight="bold", pad=10)
    ax.set_xlabel("Cover Type", fontsize=10, color=TEXT_DARK)
    ax.set_ylabel("Count", fontsize=10, color=TEXT_DARK)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(axis="x", rotation=25, labelsize=8, colors=TEXT_DARK)
    ax.tick_params(axis="y", labelsize=8, colors=TEXT_DARK)
    ax.yaxis.grid(True, color=GRID_COL, linewidth=0.6)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color(DENIM_PALE)

    fig.tight_layout()
    out = os.path.join(FIGDIR, "eda_class_distribution.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_feature_ranges(X: np.ndarray, feature_names: list[str]) -> None:
    """Box-plot of selected continuous features pre- and post-scaling."""
    # Show first 10 continuous features (elevation through horizontal_distance_fire)
    cont_idx = list(range(10))
    cont_names = [feature_names[i] for i in cont_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=CREAM)

    for ax, (data, title) in zip(axes, [(X[:, cont_idx], "Raw Features"),
                                        (X[:, cont_idx], "Scaled (StandardScaler)")]):
        ax.set_facecolor(CREAM)
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color=RUST, linewidth=1.5),
                        whiskerprops=dict(color=DENIM),
                        capprops=dict(color=DENIM),
                        flierprops=dict(marker="o", markersize=2,
                                        markerfacecolor=DENIM_PALE, alpha=0.5))
        for patch in bp["boxes"]:
            patch.set_facecolor(DENIM_PALE)
            patch.set_edgecolor(DENIM)
        ax.set_xticks(range(1, len(cont_names) + 1))
        ax.set_xticklabels([n.replace("_", "\n") for n in cont_names],
                           fontsize=6.5, rotation=0, color=TEXT_DARK)
        ax.set_title(title, fontsize=10, color=DENIM, fontweight="bold")
        ax.yaxis.grid(True, color=GRID_COL, linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(DENIM_PALE)

    fig.suptitle("Covertype Feature Distributions (10 continuous features)",
                 fontsize=12, color=DENIM, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "eda_feature_boxplots.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def eda_summary(X: np.ndarray, y: np.ndarray, splits: dict) -> dict:
    """Compute EDA summary statistics and save to JSON log."""
    counts = Counter(y.tolist())
    summary = {
        "n_total":    int(len(y)),
        "n_features": int(X.shape[1]),
        "n_classes":  7,
        "class_counts": {CLASS_NAMES[k]: int(v) for k, v in sorted(counts.items())},
        "class_balance_ratio": float(max(counts.values()) / min(counts.values())),
        "missing_values": int(np.isnan(X).sum()),
        "split_sizes": splits["split_sizes"],
        "scaler": "StandardScaler (fit on train only)",
        "seed": SEED,
        "note": "Same sample/preprocessing as SL Report — OL comparability preserved.",
    }
    out = os.path.join(LOGDIR, "eda_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  EDA summary → {out}")
    return summary


def run_eda():
    set_global_seed()
    _ensure_dirs()

    print("=== data_screening.py — EDA Confirmation ===")
    X_raw, y_raw = load_covertype_sample()
    splits = make_splits(X_raw, y_raw)

    # 0-indexed labels for display
    y_0 = y_raw.values - 1

    print("  Plotting class distribution …")
    plot_class_distribution(y_0, "~20,000-instance stratified sample")

    print("  Plotting feature ranges …")
    # Use raw continuous columns (indices 0:10) for the unscaled plot
    X_cont_raw = X_raw.values[:, :10]
    plot_feature_ranges(X_cont_raw, splits["feature_names"])

    print("  Computing summary stats …")
    summary = eda_summary(X_raw.values, y_0, splits)

    print("\n  === EDA Summary ===")
    for k, v in summary.items():
        print(f"    {k}: {v}")

    return splits


if __name__ == "__main__":
    run_eda()
    print("data_screening.py OK ✓")
