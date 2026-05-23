"""CS 7641 SL Report — Covertype evaluation harness.

Package layout:
    data_loader     — fetch/cache/sample Covertype; split + CV iterator factory.
    data_screening  — EDA tables/plots + leakage probes (dummy, shuffle, split).
    stat_learners   — Five model families (DT, kNN, SVM, sklearn MLP, PyTorch MLP).
    cv_optimizer    — Stratified k-fold tuning + learning/complexity curves.
    report_generator — Aggregates results into tables/figures/JSON for the .tex.
    styles          — Denim-themed matplotlib stylesheet + palette.

Single source of truth: configs/config.yaml. Seeds, grids, paths all live there.
"""
__version__ = "0.1.0"
