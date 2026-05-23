"""
report_generator.py — CS7641 OL Report (Summer 2026)
Author: Timothy Leung (tleung37)

Generates all publication-quality figures and summary tables for:
  Part 1 — RO progress curves (best-so-far vs. function evaluations)
  Part 2 — Optimizer loss trajectories, speed-to-threshold, heatmaps, stability
  Part 3 — Regularization comparison (train/val gap, test metrics)
  Summary — Final metric table, confusion matrices, class-level diagnostics

Aesthetic: denim theme (navy-denim primary, rust complement, cream background)
           high-contrast, colorblind-safe, publication-ready (PDF vector output)
           all internal clickable links mirrored in LaTeX via \hyperref

References:
  [1] Kingma & Ba (2014). Adam. arXiv:1412.6980
  [2] Loshchilov & Hutter (2019). AdamW. ICLR 2019.
  [3] Mitchell (1997). Machine Learning. Ch. 9.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
LOGDIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(FIGDIR, exist_ok=True)

# ── Denim palette ────────────────────────────────────────────────────────────
DENIM       = "#1B3A6B"
DENIM_MID   = "#2E6DA4"
DENIM_PALE  = "#A8C4E0"
RUST        = "#B5451B"
RUST_PALE   = "#E8A080"
GOLD        = "#D4A017"
SAGE        = "#4A7C59"
CREAM       = "#F5F1E8"
TEXT_DARK   = "#1A1A2E"
GRID_COL    = "#DADADF"
DIVERGE_NEG = "#B5451B"
DIVERGE_POS = "#2E6DA4"

OPT_COLORS = {
    "sgd_no_momentum": DENIM,
    "sgd_momentum":    DENIM_MID,
    "nesterov":        DENIM_PALE,
    "adam":            RUST,
    "adam_no_bias":    RUST_PALE,
    "adam_b1_zero":    GOLD,
    "adamw":           SAGE,
}
OPT_LABELS = {
    "sgd_no_momentum": "SGD (no mom.)",
    "sgd_momentum":    "SGD + mom.",
    "nesterov":        "Nesterov",
    "adam":            "Adam",
    "adam_no_bias":    "Adam (no bias-corr.)",
    "adam_b1_zero":    r"Adam ($\beta_1=0$)",
    "adamw":           "AdamW",
}
RO_COLORS = {"RHC": DENIM, "SA": RUST, "GA": SAGE}
REG_COLORS = {
    "adam_baseline": DENIM,
    "l2_1e-4": DENIM_MID,
    "l2_1e-3": DENIM_PALE,
    "early_stop_p10": RUST,
    "dropout_0.2": RUST_PALE,
    "dropout_0.3": GOLD,
    "label_smooth_0.1": SAGE,
    "input_noise_0.05": "#7B4F9E",
    "best_combo": TEXT_DARK,
}
REG_LABELS = {
    "adam_baseline": "Adam baseline",
    "l2_1e-4": r"L2 ($\lambda=10^{-4}$)",
    "l2_1e-3": r"L2 ($\lambda=10^{-3}$)",
    "early_stop_p10": "Early stopping (p=10)",
    "dropout_0.2": "Dropout (p=0.2)",
    "dropout_0.3": "Dropout (p=0.3)",
    "label_smooth_0.1": "Label smooth. (ε=0.1)",
    "input_noise_0.05": "Input noise (σ=0.05)",
    "best_combo": "Best combo",
}

CLASS_NAMES = ["CT1\nSpruce/Fir", "CT2\nLodgepole", "CT3\nPonderosa",
               "CT4\nCottonwood", "CT5\nAspen", "CT6\nDouglas-fir", "CT7\nKrummholz"]


def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(CREAM)
    ax.set_title(title, fontsize=10, color=DENIM, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=8.5, color=TEXT_DARK)
    ax.set_ylabel(ylabel, fontsize=8.5, color=TEXT_DARK)
    ax.tick_params(colors=TEXT_DARK, labelsize=7.5)
    ax.yaxis.grid(True, color=GRID_COL, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(DENIM_PALE)
    return ax


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── FIGURE 1: RO Progress Curves ────────────────────────────────────────────
def fig_ro_progress(log_pattern: str = "rhc|sa|ga"):
    """
    Multi-panel: best-so-far val loss vs. function evaluations for RHC, SA, GA.
    Also shows wall-clock time.
    """
    methods = ["RHC", "SA", "GA"]
    log_files = {
        "RHC": os.path.join(LOGDIR, "rhc.json"),
        "SA":  os.path.join(LOGDIR, "sa.json"),
        "GA":  os.path.join(LOGDIR, "ga.json"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=CREAM)
    fig.suptitle("Part 1: Randomized Optimization — Final 2 Layers\n(Covertype, ~20k instances, seed=42)",
                 fontsize=11, color=DENIM, fontweight="bold")

    ax_fe, ax_wc = axes
    _style(ax_fe, "Best-so-far Val Loss vs. Function Evaluations",
           "Function Evaluations", "Best-so-far Validation Loss")
    _style(ax_wc, "Best-so-far Val Loss vs. Wall-Clock Time",
           "Wall-Clock Time (s)", "Best-so-far Validation Loss")

    for method in methods:
        path = log_files[method]
        if not os.path.exists(path):
            continue
        d = _load_json(path)
        h = d["history"]
        fe  = h["func_evals"]
        bsf = h["best_so_far"]
        wc  = h["wall_clock_s"]
        col = RO_COLORS[method]
        ax_fe.plot(fe, bsf, color=col, lw=1.8, label=f"{method} (best={d['best_val_loss']:.4f})")
        ax_wc.plot(wc, bsf, color=col, lw=1.8, label=method)

        # Mark final val loss with dashed horizontal
        ax_fe.axhline(d["best_val_loss"], color=col, lw=0.8, ls="--", alpha=0.6)

    for ax in axes:
        ax.legend(fontsize=7.5, framealpha=0.85, edgecolor=DENIM_PALE)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIGDIR, "fig_ro_progress.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ── FIGURE 2: Optimizer Loss Trajectories (Part 2) ──────────────────────────
def fig_optimizer_trajectories(seed: int = 42):
    """
    Validation loss vs. gradient evaluations for all 7 optimizer variants.
    Also shows threshold-crossing marker (ℓ).
    """
    # Estimate ℓ from data
    adam_path = os.path.join(LOGDIR, f"part2_adam_seed{seed}.json")
    l_threshold = None
    if os.path.exists(adam_path):
        d = _load_json(adam_path)
        vl = np.array(d["val_loss"])
        # ℓ = 10% improvement over initial val loss
        l_threshold = vl[0] * 0.90

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=CREAM)
    fig.suptitle(f"Part 2: Adam-Family Optimizer Ablations (seed={seed})\n"
                 "Fixed Covertype backbone, 80 epochs, budget-matched",
                 fontsize=11, color=DENIM, fontweight="bold")

    ax_ge, ax_wc = axes
    _style(ax_ge, "Val Loss vs. Gradient Evaluations", "Gradient Evaluations", "Validation Loss")
    _style(ax_wc, "Val Loss vs. Wall-Clock Time", "Wall-Clock Time (s)", "Validation Loss")

    for opt_name, col in OPT_COLORS.items():
        path = os.path.join(LOGDIR, f"part2_{opt_name}_seed{seed}.json")
        if not os.path.exists(path):
            continue
        d = _load_json(path)
        ge = d["grad_evals"]
        vl = d["val_loss"]
        wc = d["wall_clock_s"]
        lbl = OPT_LABELS.get(opt_name, opt_name)
        ax_ge.plot(ge, vl, color=col, lw=1.8, label=lbl)
        ax_wc.plot(wc, vl, color=col, lw=1.8, label=lbl)

    # Draw ℓ threshold
    if l_threshold is not None:
        for ax in axes:
            ax.axhline(l_threshold, color=TEXT_DARK, lw=1.0, ls=":", alpha=0.7,
                       label=f"ℓ = {l_threshold:.3f}")

    for ax in axes:
        ax.legend(fontsize=6.5, framealpha=0.85, edgecolor=DENIM_PALE, ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(FIGDIR, "fig_optimizer_trajectories.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ── FIGURE 3: Sensitivity Heatmap (Part 2) ──────────────────────────────────
def fig_sensitivity_heatmap():
    """Adam (α, β1) sensitivity heatmap — denim diverging colormap."""
    path = os.path.join(LOGDIR, "heatmap_alpha_beta1.json")
    if not os.path.exists(path):
        print(f"  Heatmap data not found: {path}")
        return

    data = _load_json(path)
    rows = list(data.values())
    df = pd.DataFrame(rows)

    pivot = df.pivot(index="beta1", columns="lr", values="val_loss")
    pivot.index   = [f"β₁={v}" for v in pivot.index]
    pivot.columns = [f"α={v:.0e}" for v in pivot.columns]

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=CREAM)
    ax.set_facecolor(CREAM)

    vmin, vmax = df["val_loss"].min(), df["val_loss"].quantile(0.95)
    cmap = sns.color_palette("Blues_r", as_cmap=True)
    sns.heatmap(pivot, ax=ax, cmap=cmap, annot=True, fmt=".4f", linewidths=0.5,
                linecolor=CREAM, vmin=vmin, vmax=vmax,
                annot_kws={"size": 8, "color": TEXT_DARK},
                cbar_kws={"label": "Val Loss (30 epochs)"})
    ax.set_title("Adam Sensitivity: (α, β₁) Heatmap", fontsize=11,
                 color=DENIM, fontweight="bold", pad=8)
    ax.tick_params(labelsize=8, colors=TEXT_DARK)
    ax.figure.axes[-1].tick_params(labelsize=8)

    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_sensitivity_heatmap.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ── FIGURE 4: Seed Stability (Part 2) ───────────────────────────────────────
def fig_seed_stability():
    """
    Median ± IQR band for val loss across 3 seeds per optimizer.
    Shows stability comparison at budget end.
    """
    seeds = [42, 123, 7]
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=CREAM)
    _style(ax, "Part 2: Seed Stability — Val Loss at Epoch Budget",
           "Gradient Evaluations", "Validation Loss")

    for opt_name, col in OPT_COLORS.items():
        all_vl, all_ge = [], None
        for seed in seeds:
            path = os.path.join(LOGDIR, f"part2_{opt_name}_seed{seed}.json")
            if not os.path.exists(path):
                break
            d = _load_json(path)
            all_vl.append(d["val_loss"])
            all_ge = d["grad_evals"]
        if not all_vl or all_ge is None:
            continue

        min_len = min(len(v) for v in all_vl)
        arr = np.array([v[:min_len] for v in all_vl])
        ge  = np.array(all_ge[:min_len])
        med = np.median(arr, axis=0)
        q25 = np.percentile(arr, 25, axis=0)
        q75 = np.percentile(arr, 75, axis=0)

        ax.plot(ge, med, color=col, lw=1.8, label=OPT_LABELS.get(opt_name, opt_name))
        ax.fill_between(ge, q25, q75, color=col, alpha=0.15)

    ax.legend(fontsize=7, framealpha=0.85, edgecolor=DENIM_PALE, ncol=2)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_seed_stability.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ── FIGURE 5: Regularization Comparison (Part 3) ────────────────────────────
def fig_regularization():
    """
    Two-panel: val loss trajectories + final test Macro-F1 bar chart.
    """
    seeds = [42, 123, 7]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=CREAM)
    ax_traj, ax_bar = axes
    _style(ax_traj, "Part 3: Val Loss vs. Epochs (seed=42)", "Epoch", "Validation Loss")
    _style(ax_bar,  "Part 3: Final Test Macro-F1 (median ± IQR)", "Regularization", "Macro-F1")

    # Trajectory panel (seed=42 only for clarity)
    for reg_name, col in REG_COLORS.items():
        path = os.path.join(LOGDIR, f"part3_{reg_name}_seed42.json")
        if not os.path.exists(path):
            continue
        d = _load_json(path)
        epochs = list(range(1, len(d["val_loss"]) + 1))
        ax_traj.plot(epochs, d["val_loss"], color=col, lw=1.5,
                     label=REG_LABELS.get(reg_name, reg_name))
    ax_traj.legend(fontsize=6.5, framealpha=0.85, edgecolor=DENIM_PALE, ncol=2)

    # Bar chart: median Macro-F1 across seeds
    reg_names, medians, iqrs = [], [], []
    for reg_name in REG_COLORS:
        f1s = []
        for seed in seeds:
            path = os.path.join(LOGDIR, f"part3_{reg_name}_seed{seed}.json")
            if os.path.exists(path):
                d = _load_json(path)
                f1s.append(d["test_metrics"]["macro_f1"])
        if f1s:
            reg_names.append(reg_name)
            medians.append(np.median(f1s))
            iqrs.append(np.percentile(f1s, 75) - np.percentile(f1s, 25))

    if reg_names:
        cols  = [REG_COLORS.get(n, DENIM) for n in reg_names]
        lbls  = [REG_LABELS.get(n, n) for n in reg_names]
        x     = np.arange(len(reg_names))
        bars  = ax_bar.bar(x, medians, yerr=iqrs, color=cols, edgecolor=DENIM,
                           linewidth=0.6, capsize=4, error_kw={"elinewidth": 1.2})
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(lbls, rotation=35, ha="right", fontsize=6.5)
        # Annotate median values
        for bar, med in zip(bars, medians):
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"{med:.3f}", ha="center", va="bottom", fontsize=6)

    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_regularization.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ── FIGURE 6: Confusion Matrix (best model) ─────────────────────────────────
def fig_confusion_matrix(best_run: str = "part2_adamw_seed42"):
    """Confusion matrix for best model — denim diverging color map."""
    path = os.path.join(LOGDIR, f"{best_run}.json")
    if not os.path.exists(path):
        print(f"  Confusion matrix data not found: {path}")
        return

    d  = _load_json(path)
    cm = np.array(d["test_metrics"]["confusion_matrix"])

    # Normalise per row (recall)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=CREAM)

    for ax, (data, title, fmt) in zip(axes, [
        (cm,      "Confusion Matrix (counts)",            "d"),
        (cm_norm, "Confusion Matrix (row-normalised)", ".2f"),
    ]):
        ax.set_facecolor(CREAM)
        cmap = sns.color_palette("Blues", as_cmap=True)
        sns.heatmap(data, ax=ax, cmap=cmap, annot=True, fmt=fmt,
                    linewidths=0.4, linecolor=CREAM,
                    xticklabels=[f"CT{i+1}" for i in range(7)],
                    yticklabels=[f"CT{i+1}" for i in range(7)],
                    annot_kws={"size": 8})
        ax.set_title(title, fontsize=10, color=DENIM, fontweight="bold", pad=6)
        ax.set_xlabel("Predicted", fontsize=8.5, color=TEXT_DARK)
        ax.set_ylabel("True", fontsize=8.5, color=TEXT_DARK)
        ax.tick_params(labelsize=7.5)

    fig.suptitle(f"Class-level Diagnostics — {best_run.replace('_', ' ')}",
                 fontsize=11, color=DENIM, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FIGDIR, "fig_confusion_matrix.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ── FIGURE 7: Final Metric Table ────────────────────────────────────────────
def fig_final_metric_table() -> pd.DataFrame:
    """
    Build and save the final summary table (val loss, Acc, Macro-F1, Bal-Acc,
    wall-clock, grad evals, func evals) for all methods × best seed.
    Saved as CSV and as a styled PDF figure.
    """
    rows = []

    # SL Baseline
    path = os.path.join(LOGDIR, "sl_sgd_baseline.json")
    if os.path.exists(path):
        d = _load_json(path)
        rows.append({
            "Method": "SL PyTorch SGD (baseline)",
            "Val Loss": d["best_val_loss"],
            "Accuracy": d["test_metrics"]["accuracy"],
            "Macro-F1": d["test_metrics"]["macro_f1"],
            "Bal. Acc.": d["test_metrics"]["balanced_acc"],
            "Wall-clock (s)": d["wall_clock_s"][-1] if d["wall_clock_s"] else 0,
            "Grad. Evals": d["total_grad_evals"],
            "Func. Evals": "—",
        })

    # Part 1 RO
    for method_name, fname in [("RHC", "rhc"), ("SA", "sa"), ("GA", "ga")]:
        path = os.path.join(LOGDIR, f"{fname}.json")
        if os.path.exists(path):
            d = _load_json(path)
            rows.append({
                "Method": method_name,
                "Val Loss": d["best_val_loss"],
                "Accuracy": d["test_metrics"]["accuracy"],
                "Macro-F1": d["test_metrics"]["macro_f1"],
                "Bal. Acc.": d["test_metrics"]["balanced_acc"],
                "Wall-clock (s)": d["history"]["wall_clock_s"][-1],
                "Grad. Evals": "—",
                "Func. Evals": d["total_func_evals"],
            })

    # Part 2 Optimizers (best seed = 42)
    for opt_name, lbl in OPT_LABELS.items():
        path = os.path.join(LOGDIR, f"part2_{opt_name}_seed42.json")
        if os.path.exists(path):
            d = _load_json(path)
            rows.append({
                "Method": f"Part2: {lbl}",
                "Val Loss": d["best_val_loss"],
                "Accuracy": d["test_metrics"]["accuracy"],
                "Macro-F1": d["test_metrics"]["macro_f1"],
                "Bal. Acc.": d["test_metrics"]["balanced_acc"],
                "Wall-clock (s)": d["wall_clock_s"][-1] if d["wall_clock_s"] else 0,
                "Grad. Evals": d["total_grad_evals"],
                "Func. Evals": "—",
            })

    # Part 3 Regularization (best seed = 42)
    for reg_name, lbl in REG_LABELS.items():
        path = os.path.join(LOGDIR, f"part3_{reg_name}_seed42.json")
        if os.path.exists(path):
            d = _load_json(path)
            rows.append({
                "Method": f"Part3: {lbl}",
                "Val Loss": d["best_val_loss"],
                "Accuracy": d["test_metrics"]["accuracy"],
                "Macro-F1": d["test_metrics"]["macro_f1"],
                "Bal. Acc.": d["test_metrics"]["balanced_acc"],
                "Wall-clock (s)": d["wall_clock_s"][-1] if d["wall_clock_s"] else 0,
                "Grad. Evals": d["total_grad_evals"],
                "Func. Evals": "—",
            })

    if not rows:
        print("  No log files found — skipping final metric table.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Format numerics
    for col in ["Val Loss", "Accuracy", "Macro-F1", "Bal. Acc."]:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)

    # Save CSV
    csv_out = os.path.join(LOGDIR, "final_metric_table.csv")
    df.to_csv(csv_out, index=False)
    print(f"  Saved → {csv_out}")

    # Save PDF figure of table
    fig, ax = plt.subplots(figsize=(14, max(4, 0.35 * len(df) + 1.5)), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.4)
    # Style header
    for j in range(len(df.columns)):
        cell = tbl[(0, j)]
        cell.set_facecolor(DENIM)
        cell.set_text_props(color="white", fontweight="bold")
    # Alternate row shading
    for i in range(1, len(df) + 1):
        bg = DENIM_PALE if i % 2 == 0 else CREAM
        for j in range(len(df.columns)):
            tbl[(i, j)].set_facecolor(bg)
    ax.set_title("Final Metric Table — All Methods", fontsize=12,
                 color=DENIM, fontweight="bold", pad=12)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_final_metric_table.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")
    return df


# ── FIGURE 8: Train-Val Gap (Part 3) ────────────────────────────────────────
def fig_train_val_gap():
    """Bar chart of train-val loss gap at budget end — shows overfitting."""
    seeds = [42]
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=CREAM)
    _style(ax, "Part 3: Train–Validation Loss Gap at Budget End", "Regularization", "Gap (train − val loss)")

    reg_names, gaps = [], []
    for reg_name in REG_COLORS:
        g_list = []
        for seed in seeds:
            path = os.path.join(LOGDIR, f"part3_{reg_name}_seed{seed}.json")
            if os.path.exists(path):
                d = _load_json(path)
                tr = d["train_loss"][-1]
                vl = d["val_loss"][-1]
                g_list.append(tr - vl)
        if g_list:
            reg_names.append(reg_name)
            gaps.append(np.mean(g_list))

    if reg_names:
        cols = [REG_COLORS.get(n, DENIM) for n in reg_names]
        lbls = [REG_LABELS.get(n, n) for n in reg_names]
        x    = np.arange(len(reg_names))
        bars = ax.bar(x, gaps, color=cols, edgecolor=DENIM, linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(lbls, rotation=35, ha="right", fontsize=7)
        ax.axhline(0, color=TEXT_DARK, lw=0.8, ls="--")
        for bar, g in zip(bars, gaps):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    g + 0.0005 if g >= 0 else g - 0.002,
                    f"{g:.4f}", ha="center", va="bottom" if g >= 0 else "top",
                    fontsize=6)

    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_train_val_gap.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ── MAIN ─────────────────────────────────────────────────────────────────────
def generate_all():
    print("=== report_generator.py — Generating all figures ===")
    fig_ro_progress()
    fig_optimizer_trajectories()
    fig_sensitivity_heatmap()
    fig_seed_stability()
    fig_regularization()
    fig_confusion_matrix()
    fig_train_val_gap()
    fig_final_metric_table()
    print("report_generator.py OK ✓")


if __name__ == "__main__":
    generate_all()
