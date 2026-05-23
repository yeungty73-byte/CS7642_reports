"""
style_figures_v2.py — OL figures, SL-matched EXACTLY.

Key fix: SL figures use pure WHITE background (#FFFFFF), seaborn whitegrid.
The parchment (#F4EFE5) is the LaTeX page colour — NOT the figure background.
Figures are white; the document is parchment. They contrast correctly.

Matching SL style (from fig_learning_curves-4.jpg, fig_class_balance.jpg etc.):
  - fig + axes background: white
  - Grid: seaborn whitegrid (light blue-grey horizontal lines, minor off)
  - Font: default sans-serif (matches SL: looks like DejaVu Sans / Helvetica)
  - Titles: bold, denimdark (#1B3A5C), size 13-14 for subplot titles
  - Axis labels: plain (not bold), size 11
  - Legend: white box, thin grey edge, size 9
  - Line thickness: 2.0-2.5
  - Error bands: alpha 0.15-0.20
  - No spines top/right (seaborn whitegrid removes them)
  - Bar edge: same colour as bar (not a contrasting outline)

Colour cycle (SL uses per-learner unique colours — we adapt per-optimizer/method):
  denimdark   #1B3A5C   SGD no-mom / RHC
  denim       #2C6FBB   SGD+mom / Part2 second
  sage/teal   #4C9E82   Nesterov
  amber/gold  #D4A017   Adam / SA
  sienna      #B8552E   Adam no-bias-corr
  plum        #7B2D8E   Adam b1=0
  plumcomp    #408E2D   AdamW / GA
  steel       #5B9BD5   extra
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Apply seaborn whitegrid — EXACT match to SL figures
sns.set_theme(style="whitegrid", font_scale=1.0)
# Override a few defaults to match SL more precisely
plt.rcParams.update({
    "axes.facecolor":   "white",
    "figure.facecolor": "white",
    "axes.edgecolor":   "#cccccc",
    "axes.linewidth":   0.8,
    "grid.color":       "#d0d8e8",
    "grid.linewidth":   0.7,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "legend.framealpha":0.92,
    "legend.edgecolor": "#cccccc",
    "xtick.labelsize":  9.5,
    "ytick.labelsize":  9.5,
})

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
LOGDIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(FIGDIR, exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────
C = dict(
    denimdark  = "#1B3A5C",
    denim      = "#2C6FBB",
    teal       = "#4C9E82",
    amber      = "#D4A017",
    sienna     = "#B8552E",
    plum       = "#7B2D8E",
    green      = "#408E2D",
    steel      = "#5B9BD5",
)

OPT_STYLE = {
    # (color, linestyle, label)
    "sgd_no_momentum": (C["denimdark"], "--",  "SGD (no mom.)"),
    "sgd_momentum":    (C["denim"],     "-",   "SGD + mom."),
    "nesterov":        (C["teal"],      "-.",  "Nesterov"),
    "adam":            (C["amber"],     "-",   "Adam"),
    "adam_no_bias":    (C["sienna"],    "-",   "Adam (no bias-corr.)"),
    "adam_b1_zero":    (C["plum"],      "--",  r"Adam ($\beta_1{=}0$)"),
    "adamw":           (C["green"],     "-",   "AdamW"),
}
RO_STYLE = {
    "RHC": (C["denimdark"], "-",  "RHC"),
    "SA":  (C["amber"],     "--", "SA"),
    "GA":  (C["green"],     "-.", "GA"),
}
REG_STYLE = {
    "adam_baseline":    (C["denimdark"],"-",   "Adam baseline"),
    "l2_1e-4":          (C["denim"],    "-",   r"L2 ($\lambda{=}10^{-4}$)"),
    "l2_1e-3":          (C["steel"],    "--",  r"L2 ($\lambda{=}10^{-3}$)"),
    "early_stop_p10":   (C["amber"],    "-",   "Early stopping"),
    "dropout_0.2":      (C["teal"],     "-",   "Dropout 0.2"),
    "dropout_0.3":      (C["green"],    "--",  "Dropout 0.3"),
    "label_smooth_0.1": (C["plum"],     "-",   r"Label smooth"),
    "input_noise_0.05": (C["sienna"],   "-.",  r"Input noise"),
    "best_combo":       (C["denimdark"],"-",   "Best combo"),
}

def _load(fname):
    p = os.path.join(LOGDIR, fname)
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return None

# ═══════════════════════════════════════════════════════════════════════
# FIG 1: EDA class distribution  (test set note: N/A — EDA only)
# ═══════════════════════════════════════════════════════════════════════
def fig_eda():
    eda = _load("eda_summary.json")
    if eda is None: return
    names = list(eda["class_counts"].keys())
    vals  = list(eda["class_counts"].values())
    pcts  = [v/sum(vals)*100 for v in vals]
    cols  = [C["denimdark"], C["denim"], C["teal"], C["amber"],
             C["sienna"], C["plum"], C["green"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(names)), vals, color=cols, width=0.65, zorder=3)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                f"{pct:.1f}%", ha="center", va="bottom",
                fontsize=8.5, color="#333333")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=22, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax.set_title("Class distribution — stratified 20k subsample (OL)")
    ax.set_xlabel("Cover Type"); ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/eda_class_distribution.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  eda_class_distribution.pdf")

# ═══════════════════════════════════════════════════════════════════════
# FIG 2: RO progress  (val objective — test at end only)
# ═══════════════════════════════════════════════════════════════════════
def fig_ro():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Part 1: Randomized Optimization — final 2 layers",
                 fontsize=14, fontweight="bold", color=C["denimdark"])
    ax_fe, ax_wc = axes

    for meth, fpath in [("RHC","rhc"), ("SA","sa"), ("GA","ga")]:
        d = _load(f"{fpath}.json")
        if d is None: continue
        col, ls, lbl = RO_STYLE[meth]
        h = d["history"]
        lw = 2.5 if meth == "RHC" else 2.0
        ax_fe.plot(h["func_evals"], h["best_so_far"], color=col, ls=ls, lw=lw,
                   label=f"{lbl}  (best val={d['best_val_loss']:.4f})", zorder=3)
        ax_wc.plot(h["wall_clock_s"], h["best_so_far"], color=col, ls=ls, lw=lw,
                   label=lbl, zorder=3)
        ax_fe.axhline(d["best_val_loss"], color=col, lw=0.9, ls=":", alpha=0.5, zorder=2)

    ax_fe.set_title("Best-so-far val loss vs. function evaluations")
    ax_fe.set_xlabel("Function Evaluations"); ax_fe.set_ylabel("Val Loss (best-so-far)")
    ax_fe.legend()
    ax_wc.set_title("Best-so-far val loss vs. wall-clock time")
    ax_wc.set_xlabel("Wall-Clock Time (s)"); ax_wc.set_ylabel("Val Loss (best-so-far)")
    ax_wc.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{FIGDIR}/fig_ro_progress.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig_ro_progress.pdf")

# ═══════════════════════════════════════════════════════════════════════
# FIG 3: Optimizer trajectories  (val loss during training; test in table)
# ═══════════════════════════════════════════════════════════════════════
def fig_opt_traj(seed=42):
    adam = _load(f"part2_adam_seed{seed}.json")
    ell  = (adam["val_loss"][0] * 0.90) if adam else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Part 2: Adam-family optimizer ablations (seed={seed}) — val loss during training\n"
        "(held-out test Macro-F1 in Table III; curves show tuning/training dynamics only)",
        fontsize=12, fontweight="bold", color=C["denimdark"])
    ax_ge, ax_wc = axes

    for opt, (col, ls, lbl) in OPT_STYLE.items():
        d = _load(f"part2_{opt}_seed{seed}.json")
        if d is None: continue
        lw = 2.5 if opt in ("adamw","adam") else 1.9
        ax_ge.plot(d["grad_evals"], d["val_loss"], color=col, ls=ls, lw=lw,
                   label=lbl, zorder=3)
        ax_wc.plot(d["wall_clock_s"], d["val_loss"], color=col, ls=ls, lw=lw,
                   label=lbl, zorder=3)

    if ell:
        for ax in axes:
            ax.axhline(ell, color="#555555", lw=1.2, ls=":", zorder=2,
                       label=f"$\\ell = {ell:.3f}$")

    ax_ge.set_title("Val loss vs. gradient evaluations")
    ax_ge.set_xlabel("Gradient Evaluations"); ax_ge.set_ylabel("Validation Loss")
    ax_ge.legend(ncol=2, fontsize=8)
    ax_wc.set_title("Val loss vs. wall-clock time")
    ax_wc.set_xlabel("Wall-Clock Time (s)"); ax_wc.set_ylabel("Validation Loss")
    ax_wc.legend(ncol=2, fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(f"{FIGDIR}/fig_optimizer_trajectories.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig_optimizer_trajectories.pdf")

# ═══════════════════════════════════════════════════════════════════════
# FIG 4: Sensitivity heatmap  (val loss on 30-epoch probe)
# ═══════════════════════════════════════════════════════════════════════
def fig_heatmap():
    import pandas as pd
    d = _load("heatmap_alpha_beta1.json")
    if d is None: return
    df = pd.DataFrame(list(d.values()))
    pivot = df.pivot(index="beta1", columns="lr", values="val_loss")
    pivot.index   = [f"β₁={v}" for v in pivot.index]
    pivot.columns = [f"lr={v:.0e}" for v in pivot.columns]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    cmap = sns.light_palette(C["denim"], as_cmap=True, reverse=True)
    vmin, vmax = pivot.values.min(), np.percentile(pivot.values, 95)
    norm = (pivot.values - vmin) / (vmax - vmin + 1e-10)

    im = ax.imshow(norm, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            nv = norm[r, c]
            tc = "white" if nv < 0.5 else C["denimdark"]
            ax.text(c, r, f"{pivot.iloc[r,c]:.4f}", ha="center", va="center",
                    fontsize=9, color=tc, fontweight="bold")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title(r"Part 2: Adam $(\alpha, \beta_1)$ sensitivity — val loss (30 epochs)")
    ax.set_xlabel("Learning Rate"); ax.set_ylabel(r"$\beta_1$")

    for i in range(pivot.shape[1]+1): ax.axvline(i-0.5, color="white", lw=1.5)
    for i in range(pivot.shape[0]+1): ax.axhline(i-0.5, color="white", lw=1.5)

    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Val Loss (lower = better)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_sensitivity_heatmap.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig_sensitivity_heatmap.pdf")

# ═══════════════════════════════════════════════════════════════════════
# FIG 5: Seed stability
# ═══════════════════════════════════════════════════════════════════════
def fig_stability():
    seeds = [42, 123, 7]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_title("Part 2: Seed stability — median ± IQR val loss, 3 seeds")
    for opt, (col, ls, lbl) in OPT_STYLE.items():
        runs = [_load(f"part2_{opt}_seed{s}.json") for s in seeds]
        runs = [r for r in runs if r]
        if not runs: continue
        mn = min(len(r["val_loss"]) for r in runs)
        arr = np.array([r["val_loss"][:mn] for r in runs])
        ge  = np.array(runs[0]["grad_evals"][:mn])
        med = np.median(arr, axis=0)
        q25, q75 = np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0)
        lw = 2.5 if opt in ("adamw","adam") else 1.9
        ax.plot(ge, med, color=col, ls=ls, lw=lw, label=lbl, zorder=3)
        ax.fill_between(ge, q25, q75, color=col, alpha=0.13, zorder=2)
    ax.set_xlabel("Gradient Evaluations"); ax.set_ylabel("Validation Loss")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_seed_stability.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig_seed_stability.pdf")

# ═══════════════════════════════════════════════════════════════════════
# FIG 6: Regularization comparison
# Note: bar chart shows TEST Macro-F1 (median across seeds) — this is correct.
# The validation trajectories are val loss during training.
# Both are clearly labelled so there's no confusion.
# ═══════════════════════════════════════════════════════════════════════
def fig_reg():
    seeds = [42, 123, 7]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Part 3: Regularization study\n"
                 "(left: val loss during training, seed=42 | right: test Macro-F1, median ± IQR, 3 seeds)",
                 fontsize=12, fontweight="bold", color=C["denimdark"])
    ax_traj, ax_bar = axes

    for reg, (col, ls, lbl) in REG_STYLE.items():
        d = _load(f"part3_{reg}_seed42.json")
        if d is None: continue
        lw = 2.5 if reg in ("adam_baseline","best_combo") else 1.8
        ax_traj.plot(range(1, len(d["val_loss"])+1), d["val_loss"],
                     color=col, ls=ls, lw=lw, label=lbl, zorder=3)
    ax_traj.set_title("Validation loss vs. epoch (seed=42)")
    ax_traj.set_xlabel("Epoch"); ax_traj.set_ylabel("Validation Loss")
    ax_traj.legend(ncol=2, fontsize=7.5)

    # Bar: TEST Macro-F1 — labelled explicitly
    medians, iqrs, bar_cols, bar_lbls = [], [], [], []
    for reg in REG_STYLE:
        f1s = []
        for s in seeds:
            d = _load(f"part3_{reg}_seed{s}.json")
            if d: f1s.append(d["test_metrics"]["macro_f1"])
        if not f1s: continue
        medians.append(np.median(f1s))
        iqrs.append(np.percentile(f1s,75) - np.percentile(f1s,25))
        bar_cols.append(REG_STYLE[reg][0])
        bar_lbls.append(REG_STYLE[reg][2])

    x = np.arange(len(medians)) * 1.25
    bars = ax_bar.bar(x, medians, width=0.85, color=bar_cols, zorder=3,
                      yerr=iqrs, capsize=4,
                      error_kw={"elinewidth":1.5, "ecolor":"#555555", "capthick":1.5})
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(bar_lbls, rotation=35, ha="right", fontsize=8)
    for bar, med in zip(bars, medians):
        ax_bar.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height() + max(iqrs)*1.2 + 0.003,
                    f"{med:.3f}", ha="center", va="bottom", fontsize=7.5, color="#333333")
    ax_bar.set_ylim(0, max(medians)+max(iqrs)+0.07)
    ax_bar.set_title("Test Macro-F1 (held-out, median ± IQR, 3 seeds)")
    ax_bar.set_xlabel("Regularization"); ax_bar.set_ylabel("Test Macro-F1")
    ax_bar.yaxis.grid(True); ax_bar.set_axisbelow(True)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(f"{FIGDIR}/fig_regularization.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig_regularization.pdf")

# ═══════════════════════════════════════════════════════════════════════
# FIG 7: Train-val gap (Part 3) — val side; test in table
# ═══════════════════════════════════════════════════════════════════════
def fig_gap():
    fig, ax = plt.subplots(figsize=(9.5, 4))
    reg_keys = list(REG_STYLE.keys())
    gaps, cols_g, lbls_g = [], [], []
    for reg in reg_keys:
        d = _load(f"part3_{reg}_seed42.json")
        if d is None: continue
        gaps.append(d["train_loss"][-1] - d["val_loss"][-1])
        cols_g.append(REG_STYLE[reg][0])
        lbls_g.append(REG_STYLE[reg][2])

    x = np.arange(len(gaps)) * 1.25
    ax.bar(x, gaps, width=0.85, color=cols_g, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(lbls_g, rotation=35, ha="right", fontsize=8.5)
    ax.axhline(0, color="#555555", lw=1.2, ls="--", zorder=4)
    for i, g in enumerate(gaps):
        ax.text(x[i], max(g, 0)+0.001, f"{g:.4f}", ha="center",
                va="bottom", fontsize=7, color="#333333")
    ax.set_title("Part 3: Train – validation loss gap at epoch budget (seed=42)")
    ax.set_xlabel("Regularization"); ax.set_ylabel("Gap (train − val loss)")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_train_val_gap.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig_train_val_gap.pdf")

# ═══════════════════════════════════════════════════════════════════════
# FIG 8: Confusion matrix on held-out TEST set — explicitly labelled
# ═══════════════════════════════════════════════════════════════════════
def fig_cm():
    # Use L2 1e-4 (best single regularizer) or fall back to AdamW
    d = _load("part3_l2_1e-4_seed42.json") or _load("part2_adamw_seed42.json")
    if d is None: return
    cm = np.array(d["test_metrics"]["confusion_matrix"])
    cm_n = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    ctlbls = ["Spruce/Fir", "Lodgepole", "Ponderosa", "Cottonwood", "Aspen",
              "Douglas-fir", "Krummholz"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Part 3: Confusion matrices on held-out test set (L2 λ=1e-4, seed=42)\n"
                 "Left: raw counts | Right: row-normalised (= per-class recall)",
                 fontsize=12, fontweight="bold", color=C["denimdark"])

    cmap = sns.light_palette(C["denim"], as_cmap=True, reverse=True)
    for ax, (data, fmt, title) in zip(axes, [
        (cm,   "d",   "Raw counts"),
        (cm_n, ".2f", "Row-normalised recall"),
    ]):
        sns.heatmap(data, ax=ax, cmap=cmap, annot=True, fmt=fmt,
                    linewidths=0.5, linecolor="white",
                    xticklabels=ctlbls, yticklabels=ctlbls,
                    annot_kws={"size": 8})
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.tick_params(labelsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(f"{FIGDIR}/fig_confusion_matrix.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig_confusion_matrix.pdf")

# ═══════════════════════════════════════════════════════════════════════
# FIG 9: Final metric table  (test metrics — all columns clearly labelled)
# ═══════════════════════════════════════════════════════════════════════
def fig_metric_table():
    import pandas as pd
    rows = []

    def _row(lbl, part, d, fe="—"):
        if d is None: return
        m = d.get("test_metrics", {})
        wc = d.get("wall_clock_s", [0])
        rows.append({
            "Part": part, "Method": lbl,
            "Val Loss↓": f"{d.get('best_val_loss',0):.4f}",
            "Test Acc↑":  f"{m.get('accuracy',0):.4f}",
            "Test F1↑":   f"{m.get('macro_f1',0):.4f}",
            "Test BalAcc↑": f"{m.get('balanced_acc',0):.4f}",
            "Wall (s)":   f"{wc[-1]:.1f}" if wc else "—",
            "GEs":        f"{d.get('total_grad_evals',0):,}" if d.get('total_grad_evals') else "—",
            "FEs":        fe,
        })

    _row("SL SGD baseline", "SL Baseline", _load("sl_sgd_baseline.json"))
    for meth, fname in [("RHC","rhc"),("SA","sa"),("GA","ga")]:
        d = _load(f"{fname}.json")
        if d: _row(meth, "Part 1 (RO)", d, fe=f"{d.get('total_func_evals',0):,}")
    for opt, (_c,_ls,lbl) in OPT_STYLE.items():
        _row(lbl, "Part 2 (Opt.)", _load(f"part2_{opt}_seed42.json"))
    for reg, (_c,_ls,lbl) in REG_STYLE.items():
        _row(lbl, "Part 3 (Reg.)", _load(f"part3_{reg}_seed42.json"))

    if not rows: return
    df = pd.DataFrame(rows)

    fig_h = max(4.0, 0.38 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    ax.axis("off")
    ax.set_title("Final metric table — all methods, held-out TEST set (seed=42)\n"
                 "Val Loss = validation loss (used for model selection); "
                 "Test Acc/F1/BalAcc = single held-out evaluation, never seen during training",
                 fontsize=10, fontweight="bold", color=C["denimdark"], pad=10)

    tbl = ax.table(cellText=df.values, colLabels=df.columns,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.5)

    # Header: denim bg, white text
    for j in range(len(df.columns)):
        cell = tbl[(0, j)]
        cell.set_facecolor(C["denimdark"])
        cell.set_text_props(color="white", fontweight="bold")

    # Alternating rows: white / very light blue
    for i in range(1, len(df)+1):
        bg = "#EEF3FA" if i % 2 == 0 else "white"
        for j in range(len(df.columns)):
            tbl[(i,j)].set_facecolor(bg)
            tbl[(i,j)].set_text_props(color="#1a1a1a")

    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_final_metric_table.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    df.to_csv(os.path.join(LOGDIR, "final_metric_table.csv"), index=False)
    print("  fig_final_metric_table.pdf")


if __name__ == "__main__":
    print("=== Regenerating figures (SL-matched: white bg, seaborn whitegrid) ===")
    fig_eda()
    fig_ro()
    fig_opt_traj()
    fig_heatmap()
    fig_stability()
    fig_reg()
    fig_gap()
    fig_cm()
    fig_metric_table()
    print("Done ✓")
