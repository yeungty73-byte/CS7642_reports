"""
run_pipeline.py — CS7641 OL Report (Summer 2026)
Author: Timothy Leung (tleung37)

End-to-end pipeline runner. Reproduces ALL experimental results in sequence:
  1. Data loading & EDA confirmation
  2. SL PyTorch SGD baseline
  3. Part 1: RHC, SA, GA on final 2 layers
  4. Part 2: Adam-family ablations × 3 seeds
  5. Part 2: Sensitivity heatmap
  6. Part 3: Regularization study × 3 seeds
  7. Report figures & tables

Usage (standard Linux machine):
  pip install -r requirements.txt
  python run_pipeline.py

Estimated runtime: 15–40 min on CPU (batch_size=256, 80 epochs).
                   ~5 min on CUDA GPU.

Seeds: [42, 123, 7]  (same as SL Report)
"""

import os
import sys
import time
import copy
import torch

from data_loader import (
    set_global_seed, load_covertype_sample, make_splits, make_dataloaders,
    SEED, BATCH_SIZE,
)
from data_screening import run_eda
from stat_learners import CovertypeNet, unfreeze_all
from optimizer import (
    train_gradient_based, run_rhc, run_sa, run_ga,
    run_part2_ablation, run_sensitivity_heatmap, run_part3_regularization,
    DEVICE,
)
from report_generator import generate_all

SEEDS = [42, 123, 7]


def main():
    t_start = time.perf_counter()
    set_global_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 60)

    # ── Step 1: Data ──────────────────────────────────────────────
    print("\n[1/7] Loading Covertype sample & running EDA …")
    run_eda()
    X, y = load_covertype_sample()
    splits  = make_splits(X, y)
    loaders = make_dataloaders(splits, batch_size=BATCH_SIZE)
    print(f"  Train={splits['split_sizes']['train']}, "
          f"Val={splits['split_sizes']['val']}, "
          f"Test={splits['split_sizes']['test']}")

    # ── Step 2: SL Baseline ───────────────────────────────────────
    print("\n[2/7] Training SL PyTorch SGD baseline …")
    sl_model = CovertypeNet(n_features=54, n_classes=7, dropout_rate=0.0).to(DEVICE)
    sl_opt   = torch.optim.SGD(sl_model.parameters(), lr=0.05, momentum=0.0)
    sl_hist  = train_gradient_based(
        sl_model, loaders, sl_opt, n_epochs=80, seed=SEED, run_name="sl_sgd_baseline"
    )
    print(f"  SL baseline — val_loss: {sl_hist['best_val_loss']:.4f} | "
          f"test_acc: {sl_hist['test_metrics']['accuracy']:.4f} | "
          f"macro_f1: {sl_hist['test_metrics']['macro_f1']:.4f}")

    # ── Step 3: Part 1 — Randomized Optimization ─────────────────
    print("\n[3/7] Part 1: Randomized Optimization on final 2 layers …")
    # Check final-layer param count (must be ≤ 50,000)
    frozen = copy.deepcopy(sl_model)
    from stat_learners import freeze_except_final
    freeze_except_final(frozen, n_final_layers=2)
    n_ro_params = sum(p.numel() for p in frozen.parameters() if p.requires_grad)
    print(f"  RO parameter count (final 2 layers): {n_ro_params:,} (limit: 50,000)")
    assert n_ro_params <= 50_000, f"Exceeds 50k RO param budget: {n_ro_params}"

    rhc_result = run_rhc(sl_model, loaders, n_final_layers=2,
                         n_restarts=3, n_iters_per_restart=400,
                         step_size=0.01, seed=SEED)
    sa_result  = run_sa(sl_model, loaders, n_final_layers=2,
                        n_iters=2000, T_init=1.0, T_decay=0.995,
                        step_size=0.01, seed=SEED)
    ga_result  = run_ga(sl_model, loaders, n_final_layers=2,
                        population_size=20, n_generations=50,
                        mutation_std=0.01, crossover_rate=0.7,
                        elitism_k=2, seed=SEED)

    # ── Step 4: Part 2 — Adam-family ablations ───────────────────
    print("\n[4/7] Part 2: Adam-family optimizer ablations × 3 seeds …")
    part2_results = run_part2_ablation(loaders, seeds=SEEDS)

    # ── Step 5: Part 2 — Sensitivity heatmap ─────────────────────
    print("\n[5/7] Part 2: Sensitivity heatmap (α, β1) …")
    heatmap = run_sensitivity_heatmap(loaders)

    # ── Step 6: Part 3 — Regularization ──────────────────────────
    print("\n[6/7] Part 3: Regularization study × 3 seeds …")
    part3_results = run_part3_regularization(loaders, seeds=SEEDS)

    # ── Step 7: Figures & tables ──────────────────────────────────
    print("\n[7/7] Generating report figures and tables …")
    generate_all()

    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed / 60:.1f} min.")
    print("Outputs: logs/*.json, figures/*.pdf")
    print("Next: compile OL_Report.tex → OL_Report_tleung37.pdf")


if __name__ == "__main__":
    main()
