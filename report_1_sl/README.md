# CS7641 SL Report — Timothy Leung (tleung37)
## Supervised Learning | Summer 2026

**Dataset:** UCI Forest Cover Type (Covertype) — 7-class, 54 features, ~580k rows; we work on a stratified 20k subsample (seed 7641).  
**Goal of this codebase:** reproduce, from raw data to bound PDF, the SL report's headline numbers — DT, kNN, SVM-RBF, sklearn MLP, PyTorch MLP — with a parallel R sanity check.

---

## Layout

```
report_1_sl/
├── requirements.txt              # all Python deps pinned (sklearn 1.5.1, torch 2.3.1+cu121, etc.)
├── configs/
│   └── config.yaml               # single source of truth for seeds, grids, sample sizes
├── notes/
│   └── HYPOTHESIS.md             # three hypothesis clauses, locked before tuning
├── src/
│   ├── styles.py                 # denim palette + Matplotlib theme
│   ├── data_loader.py            # Covertype download/cache + stratified train/test
│   ├── data_screening.py         # EDA + leakage probes (dummy/shuffled-label/disjoint)
│   ├── stat_learners.py          # DT/kNN/SVM/MLP-sk/MLP-pt unified API
│   ├── cv_optimizer.py           # tune(), learning_curve(), complexity_curve()
│   └── report_generator.py       # all denim-themed figures and tables
├── run_pipeline.py               # main entry; `--quick` flag for a smoke test
├── data/
│   └── covtype_25k.csv.gz        # the stratified pre-sample (also used by R)
├── results/
│   ├── figures/                  # PNGs consumed by SL_Report.tex
│   ├── tables/                   # CSV ground truth for every number in the .tex
│   └── logs/                     # JSON provenance (screening, leakage, tune_summary)
├── report/
│   ├── SL_Report.tex             # IEEE-conference, denim aesthetic, hypothesis-first
│   └── SL_Report.pdf             # 8 pages, compiled with pdflatex
├── repro/
│   ├── REPRO_SL.tex
│   └── REPRO_SL.pdf
└── rsanity/
    ├── sanity_check.Rmd           # bookdown::pdf_document2, denim-themed
    └── sanity_check.pdf           # R vs Python head-to-head
```

---

## How to re-run end-to-end

```bash
# 1. Install Python deps (or use your existing env)
pip install -r requirements.txt

# 2. Re-run the full pipeline (~30 min on 8-core CPU; uses 20k stratified sample)
python run_pipeline.py

# 3. Re-compile the IEEE PDF
cd report && pdflatex SL_Report.tex && pdflatex SL_Report.tex

# 4. Re-knit the R sanity check (~3 min)
cd ../rsanity && Rscript -e 'rmarkdown::render("sanity_check.Rmd", output_format="bookdown::pdf_document2")'
```

For a smoke test (4k sample, ~1 min): `python run_pipeline.py --quick`.

---

## Reproducibility

- All randomness keyed to `seed: 7641` (course number) in `configs/config.yaml`.
- Hypotheses locked in `notes/HYPOTHESIS.md` before any tuning or final experiments ran.
- The stratified 20k sample is drawn from `data/covtype_25k.csv.gz` exactly the same way in Python and R, so the R sanity check is genuinely apples-to-apples.
- Leakage probes (`results/logs/leakage.json`) confirm `dummy_acc=0.372`, `shuffled_label_macro_f1=0.106`, `disjoint=True`.

---

## What to wire up before submission

1. **Enterprise GT GitHub repo** — push `report_1_sl/` minus `data/covtype_cache.npz` (which is 13 MB and trivially regenerable).
2. **Enterprise GT Overleaf project** — upload `report/SL_Report.tex` and the contents of `results/figures/`. The `.tex` expects figures at `../results/figures/`; in Overleaf, mirror the same relative path or update the `\includegraphics` lines.
3. **Fill in the placeholders in `repro/REPRO_SL.tex`:**
   - The read-only Overleaf share URL
   - The final `git rev-parse HEAD` SHA
4. **AI Use Statement** — already present in the final section of the `.tex`. Edit if the rubric calls for a different framing of LLM assistance.

---

## Headline numbers (test set, n=4,000)

| Learner | Acc | Macro-F1 | Bal. Acc | Best hyper-params |
|---|---|---|---|---|
| SVM-RBF | **0.801** | **0.697** | 0.649 | C=10, $\gamma$=0.1 |
| kNN | 0.790 | 0.669 | **0.658** | k=1, uniform |
| MLP-sklearn | 0.782 | 0.645 | 0.604 | (128,64), $\alphta$=1e-3 |
| DT | 0.750 | 0.634 | 0.621 | $\text{ccp}_\alpha$=1e-4, no max-depth cap |
| MLP-PyTorch | 0.756 | 0.538 | 0.495 | (128,64), ReLU, lr=0.1 |

R sanity check confirms DT ($\Delta$=+0.008) and kNN ($\Delta$=−0.028) within 0.03 macro-F1 of Python; SVM ($\Delta$=−0.074) and NN ($\Delta$=−0.108) diverge as expected because `kernlab` defaults differ from libsvm and `nnet` uses BFGS rather than SGD.
