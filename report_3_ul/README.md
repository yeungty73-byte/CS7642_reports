# CS7641 UL Report — Timothy Leung (tleung37)
## Unsupervised Learning | Summer 2026

**Dataset:** UCI Forest Cover Type (Covertype) — same stratified 20k subsample as the SL and OL Reports (seed 7641, `StratifiedShuffleSplit`).  
**Goal of this codebase:** reproduce, from raw data to bound PDF, the UL report's headline numbers — K-Means, GMM/EM, PCA, ICA, Randomized Projections, downstream NN, and extra-credit UMAP — with an R sanity check confirming Python clustering and PCA diagnostics.

---

## Layout

```
report_3_ul/
├── requirements.txt              # pinned Python deps (sklearn 1.5, torch 2.3, umap-learn 0.5)
├── configs/
│   └── config.yaml               # single source of truth: seeds, k/n sweeps, component counts
├── notes/
│   └── HYPOTHESIS_UL.md          # H1–H3 locked before any Part 1–4 experiment ran
├── src/
│   ├── __init__.py
│   ├── styles.py                 # denim palette + orchid partref + Matplotlib/seaborn theme
│   ├── data_loader.py            # Covertype fetch/cache, same 60/20/20 stratified split as SL/OL
│   ├── data_screening.py         # class distribution, CT4 prevalence check
│   ├── clustering.py             # K-Means + GMM sweeps, silhouette/ARI/NMI/DBI, contingency
│   ├── dim_reduction.py          # PCA (EVR, effective rank), ICA (kurtosis/seed sweep),
│   │                             #   RP (JL distortion diagnostics), UMAP (extra credit)
│   ├── nn_eval.py                # downstream MLP [256,128] from OL: re-run on each DR space
│   └── report_generator.py       # all denim-themed figures and tables (PDF output)
├── run_pipeline.py               # main entry; runs Parts 1–4 + UMAP in order
├── data/
│   └── README.md                 # Covertype fetched via sklearn.datasets; see run_pipeline.py
├── results/
│   ├── figures/                  # PDF figures consumed by report/UL_Report.tex
│   ├── tables/                   # CSV ground truth (clustering_metrics.csv, nn_dr_results.csv)
│   └── logs/                     # JSON provenance: one file per run × seed
├── report/
│   ├── UL_Report.tex             # IEEE-conference, denim+orchid aesthetic, hypothesis-first
│   └── UL_Report.pdf             # 8 pages, compiled with pdflatex
├── repro/
│   ├── REPRO_UL.tex              # reproducibility sheet
│   └── REPRO_UL.pdf
└── rsanity/
    └── sanity_check_UL.Rmd       # bookdown::pdf_document2; R kmeans + prcomp vs Python
                                  # confirms sil Δ < 0.002 and PC EVR identical to float precision
```

---

## How to re-run end-to-end

```bash
# 1. Install Python deps
pip install -r requirements.txt

# (optional) install umap-learn only if re-running extra-credit UMAP
pip install umap-learn>=0.5

# 2. Run the full pipeline (~20–30 min on CPU)
python run_pipeline.py

# 3. Re-compile the IEEE report PDF
cd report && pdflatex UL_Report.tex && pdflatex UL_Report.tex

# 4. Re-knit the R sanity check (~3–5 min)
cd ../rsanity && Rscript -e 'bookdown::render_book("sanity_check_UL.Rmd")'
```

For a smoke test (subsample, ~2 min): `python run_pipeline.py --quick`

---

## Reproducibility

- All randomness keyed to `seed: 7641` (course number) in `configs/config.yaml`.
- ICA seed sweep uses seeds `{7641, 42, 123, 7, 99}` — reported aggregate kurtosis is stable across all five (mean |κ| range: 6.01–6.08); individual component orderings vary as expected.
- Hypotheses H1–H3 locked in `notes/HYPOTHESIS_UL.md` before any Part 1–4 experiment ran.
- Covertype sample, split, and preprocessing are **identical** to SL and OL Reports — no changes.
- R sanity check (`rsanity/`) confirms Python K-Means silhouette within |Δ| < 0.002 and PCA EVR to floating-point precision.
- DR transformations in Part 4 (downstream NN) are fit on the training split **only** and applied to val/test — leakage-safe.

---

## What to wire up before submission

1. **Enterprise GT GitHub repo** — push `report_3_ul/` minus `data/covtype_cache.npz` (trivially regenerable via sklearn).
2. **Enterprise GT Overleaf project** — upload `report/UL_Report.tex` plus all PDFs in `results/figures/`. The `.tex` expects figures at `../results/figures/`; mirror the same relative path in Overleaf or update the `\includegraphics` lines if Overleaf flattens the tree.
3. **Fill in the placeholders in `repro/REPRO_UL.tex`:**
   - `PLACEHOLDER_OVERLEAF_LINK` → the read-only Overleaf share URL
   - `PLACEHOLDER_COMMIT_HASH` → the final `git rev-parse HEAD` SHA
4. **AI Use Statement** — already present in the final section of `UL_Report.tex`. Edit if the rubric calls for updated framing.

---

## Headline numbers (test set, n = 4,000, seed 7641)

### Part 1 — Clustering on original features

| Algorithm | Silhouette | DBI | CH | ARI | NMI |
|---|---|---|---|---|---|
| K-Means (k=7) | **0.1306** | **1.792** | **2683** | 0.023 | 0.067 |
| GMM full (n=7) | −0.007 | 4.087 | 513 | **0.052** | **0.131** |

Confirms **H1**: ARI < 0.06 — Euclidean clusters capture elevation/wilderness boundaries, not ecological labels.

### Part 4 — Downstream MLP after DR

| Features | dim | Macro-F1 | Accuracy | Bal. Acc | Train (s) |
|---|---|---|---|---|---|
| Original | 54 | 0.6803 | 0.7940 | 0.6508 | 5.6 |
| **PCA-30** | **30** | **0.6951** | **0.8000** | **0.6642** | 7.0 |
| ICA-20 | 20 | 0.6785 | 0.7875 | 0.6359 | 7.2 |
| RP-30 | 30 | 0.6670 | 0.7883 | 0.6339 | 6.9 |

Confirms **H2**: PCA at n=30 (99.5% variance) improves Macro-F1 +0.015 vs. original — minority classes CT4 (+0.040), CT6 (+0.044), CT3 (+0.023) drive the gain.

R sanity check confirms Python K-Means silhouette (|Δ| = 0.0016) and PCA EVR (identical to float precision); all clustering JSON logs pass range-validity checks.
