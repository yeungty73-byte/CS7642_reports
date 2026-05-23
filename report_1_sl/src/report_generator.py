"""All denim-themed figure emitters + table writers.

Each function takes pre-computed results and writes a file under
`results/figures/` or `results/tables/`. No experiments here — this module is
deterministic given inputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score,
)

from .cv_optimizer import CurveResult
from .data_loader import CLASS_NAMES
from .styles import (
    AMBER, CLASS_PALETTE, DENIM, DENIM_BRIGHT, DENIM_DARK, DENIM_MID,
    MODEL_PALETTE, PARCHMENT, PLUM, PLUM_COMP, SAGE, SUCCESS, TERRACOTTA,
    apply_style, shaded_band,
)


FIG_DIR = Path("results/figures")
TBL_DIR = Path("results/tables")
LOG_DIR = Path("results/logs")
for _d in (FIG_DIR, TBL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Section II — class balance + scale.
# =============================================================================
def fig_class_balance(
    full_counts: dict, sample_counts: dict, out: str = "fig_class_balance.png",
) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=False)
    classes = sorted(set(full_counts) | set(sample_counts))
    full = [full_counts.get(c, 0) for c in classes]
    samp = [sample_counts.get(c, 0) for c in classes]
    labels = [CLASS_NAMES.get(c, str(c)) for c in classes]
    colors = [CLASS_PALETTE[i % len(CLASS_PALETTE)] for i in range(len(classes))]

    axes[0].bar(range(len(classes)), full, color=colors, edgecolor=DENIM_DARK, linewidth=0.5)
    axes[0].set_title("Full Covertype (n≈581k)", color=DENIM_DARK)
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(len(classes)))
    axes[0].set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    axes[0].set_ylabel("count (log)")

    axes[1].bar(range(len(classes)), samp, color=colors, edgecolor=DENIM_DARK, linewidth=0.5)
    axes[1].set_title(f"Stratified subsample (n={sum(samp)})", color=DENIM_DARK)
    axes[1].set_xticks(range(len(classes)))
    axes[1].set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    for i, v in enumerate(samp):
        axes[1].text(i, v, f"{v}", ha="center", va="bottom",
                     fontsize=6, color=DENIM_DARK)
    fig.suptitle("Class distribution — full vs. stratified subsample",
                 color=DENIM_DARK, fontsize=10, y=1.02)
    fig.tight_layout()
    p = FIG_DIR / out
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p


# =============================================================================
# Section IV — Learning + complexity curves (multi-learner panels).
# =============================================================================
def _color_for(learner_name: str) -> str:
    return MODEL_PALETTE.get(learner_name, DENIM)


def fig_learning_curves(
    curves: dict[str, CurveResult], out: str = "fig_learning_curves.png",
) -> Path:
    apply_style()
    n = len(curves)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.8 * rows),
                             sharey=False, sharex=True, squeeze=False)
    axes = axes.ravel()
    for i, (name, cr) in enumerate(curves.items()):
        ax = axes[i]
        xs = [p.x for p in cr.points]
        val_mean = np.array([p.mean for p in cr.points])
        val_std = np.array([p.std for p in cr.points])
        tr_mean = np.array([p.mean for p in cr.train_points]) if cr.train_points else None
        tr_std = np.array([p.std for p in cr.train_points]) if cr.train_points else None

        c = _color_for(name)
        ax.plot(xs, val_mean, color=c, lw=1.8, marker="o", ms=4, label="val")
        ax.fill_between(xs, val_mean - val_std, val_mean + val_std,
                        color=c, alpha=0.18)
        if tr_mean is not None:
            ax.plot(xs, tr_mean, color=c, lw=1.0, ls="--",
                    marker="s", ms=3, alpha=0.7, label="train")
            ax.fill_between(xs, tr_mean - tr_std, tr_mean + tr_std,
                            color=c, alpha=0.08)
        ax.set_title(cr.learner, color=DENIM_DARK, fontsize=9)
        ax.set_xlabel(cr.x_label, fontsize=8)
        ax.set_ylabel("macro-F1", fontsize=8)
        ax.legend(fontsize=7, loc="lower right", frameon=False)
        ax.grid(True, alpha=0.25)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Learning curves — macro-F1 vs. training-set fraction",
                 color=DENIM_DARK, fontsize=11, y=1.01)
    fig.tight_layout()
    p = FIG_DIR / out
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_complexity_curves(
    curves: dict[str, CurveResult], out: str = "fig_complexity_curves.png",
) -> Path:
    apply_style()
    n = len(curves)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.8 * rows),
                             squeeze=False)
    axes = axes.ravel()
    for i, (name, cr) in enumerate(curves.items()):
        ax = axes[i]
        xs = [str(p.x) for p in cr.points]
        val_mean = np.array([p.mean for p in cr.points])
        val_std = np.array([p.std for p in cr.points])
        tr_mean = np.array([p.mean for p in cr.train_points]) if cr.train_points else None
        tr_std = np.array([p.std for p in cr.train_points]) if cr.train_points else None
        c = _color_for(name)
        ax.errorbar(xs, val_mean, yerr=val_std, color=c, lw=1.6,
                    marker="o", ms=4, capsize=2, label="val")
        if tr_mean is not None:
            ax.errorbar(xs, tr_mean, yerr=tr_std, color=c, lw=1.0, ls="--",
                        marker="s", ms=3, alpha=0.7, capsize=2, label="train")
        ax.set_title(f"{cr.learner} ({cr.x_label})", color=DENIM_DARK, fontsize=9)
        ax.set_xlabel(cr.x_label, fontsize=8)
        ax.set_ylabel("macro-F1", fontsize=8)
        ax.tick_params(axis="x", labelrotation=35, labelsize=7)
        ax.legend(fontsize=7, loc="lower right", frameon=False)
        ax.grid(True, alpha=0.25)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Model-complexity curves — train vs. validation macro-F1",
                 color=DENIM_DARK, fontsize=11, y=1.01)
    fig.tight_layout()
    p = FIG_DIR / out
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p


# =============================================================================
# Section V — confusion matrices + per-class report.
# =============================================================================
def fig_confusion(
    y_true: np.ndarray, y_pred: np.ndarray,
    learner_label: str, out: str,
) -> Path:
    apply_style()
    cm = confusion_matrix(y_true, y_pred, labels=sorted(np.unique(y_true)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", cmap="Blues",
        cbar_kws={"label": "row-normalized"},
        xticklabels=[CLASS_NAMES.get(c, str(c)) for c in sorted(np.unique(y_true))],
        yticklabels=[CLASS_NAMES.get(c, str(c)) for c in sorted(np.unique(y_true))],
        annot_kws={"size": 6}, ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion — {learner_label}", color=DENIM_DARK)
    fig.tight_layout()
    p = FIG_DIR / out
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_all_confusions(
    preds: dict[str, np.ndarray], y_test: np.ndarray,
    out: str = "fig_confusions_all.png",
) -> Path:
    apply_style()
    n = len(preds)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows),
                             squeeze=False)
    axes = axes.ravel()
    classes = sorted(np.unique(y_test))
    labels = [CLASS_NAMES.get(c, str(c)) for c in classes]
    for i, (name, yhat) in enumerate(preds.items()):
        cm = confusion_matrix(y_test, yhat, labels=classes)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        sns.heatmap(
            cm_norm, annot=False, cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=axes[i], cbar=False,
        )
        axes[i].set_title(name, color=DENIM_DARK, fontsize=9)
        axes[i].tick_params(axis="x", labelrotation=35, labelsize=6)
        axes[i].tick_params(axis="y", labelsize=6)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Row-normalized confusion matrices on held-out test",
                 color=DENIM_DARK, fontsize=11, y=1.01)
    fig.tight_layout()
    p = FIG_DIR / out
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p


# =============================================================================
# Runtime — wall-clock per learner.
# =============================================================================
def fig_runtime(
    runtimes: dict[str, dict], out: str = "fig_runtime.png",
) -> Path:
    """`runtimes[name]` is {"fit": s, "predict": s, "tune_total": s}."""
    apply_style()
    names = list(runtimes.keys())
    fit = [runtimes[n].get("fit", 0.0) for n in names]
    pred = [runtimes[n].get("predict", 0.0) for n in names]
    tune = [runtimes[n].get("tune_total", 0.0) for n in names]
    x = np.arange(len(names))
    w = 0.27
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x - w, tune, w, color=DENIM_MID, edgecolor=DENIM_DARK, label="tune (total)")
    ax.bar(x, fit, w, color=DENIM, edgecolor=DENIM_DARK, label="final fit")
    ax.bar(x + w, pred, w, color=AMBER, edgecolor=DENIM_DARK, label="predict test")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("wall-clock seconds (log)")
    ax.set_title("Wall-clock runtime per learner", color=DENIM_DARK)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    p = FIG_DIR / out
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p


# =============================================================================
# Neural network epoch curves (sklearn loss + PyTorch history).
# =============================================================================
def fig_nn_epoch_curves(
    sk_loss_curve: list[float] | None,
    pt_history: dict | None,
    out: str = "fig_nn_epochs.png",
) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    if sk_loss_curve is not None and len(sk_loss_curve):
        axes[0].plot(sk_loss_curve, color=DENIM, lw=1.6, label="train loss")
        axes[0].set_title("sklearn MLP — training loss", color=DENIM_DARK)
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("cross-entropy")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(fontsize=8, frameon=False)
    else:
        axes[0].text(0.5, 0.5, "no sklearn loss curve",
                     transform=axes[0].transAxes, ha="center")
    if pt_history is not None and len(pt_history.get("train_loss", [])):
        ep = range(1, len(pt_history["train_loss"]) + 1)
        axes[1].plot(ep, pt_history["train_loss"], color=DENIM, lw=1.5, label="train")
        axes[1].plot(ep, pt_history["val_loss"], color=AMBER, lw=1.5, label="val")
        axes[1].set_title("PyTorch MLP — loss/epoch", color=DENIM_DARK)
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("cross-entropy")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(fontsize=8, frameon=False)
    else:
        axes[1].text(0.5, 0.5, "no PyTorch history",
                     transform=axes[1].transAxes, ha="center")
    fig.tight_layout()
    p = FIG_DIR / out
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p


# =============================================================================
# Tables.
# =============================================================================
def table_final_scores(
    preds: dict[str, np.ndarray], y_test: np.ndarray,
    runtimes: dict[str, dict], best_params: dict[str, dict],
    out: str = "table_final_scores.csv",
) -> Path:
    rows = []
    for name, yhat in preds.items():
        rows.append({
            "learner": name,
            "accuracy": accuracy_score(y_test, yhat),
            "macro_f1": f1_score(y_test, yhat, average="macro"),
            "balanced_accuracy": balanced_accuracy_score(y_test, yhat),
            "fit_seconds": runtimes.get(name, {}).get("fit", np.nan),
            "predict_seconds": runtimes.get(name, {}).get("predict", np.nan),
            "tune_seconds": runtimes.get(name, {}).get("tune_total", np.nan),
            "best_params": json.dumps(best_params.get(name, {}), default=str),
        })
    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    p = TBL_DIR / out
    df.to_csv(p, index=False)
    return p


def table_per_class(
    name: str, y_true: np.ndarray, y_pred: np.ndarray,
    out: str | None = None,
) -> Path:
    rpt = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(rpt).T.round(4)
    out = out or f"table_per_class_{name}.csv"
    p = TBL_DIR / out
    df.to_csv(p, index=True)
    return p


def table_grid(name: str, grid_df: pd.DataFrame, out: str | None = None) -> Path:
    out = out or f"table_grid_{name}.csv"
    p = TBL_DIR / out
    grid_df.to_csv(p, index=False)
    return p


__all__ = [
    "fig_class_balance", "fig_learning_curves", "fig_complexity_curves",
    "fig_confusion", "fig_all_confusions", "fig_runtime", "fig_nn_epoch_curves",
    "table_final_scores", "table_per_class", "table_grid",
]
