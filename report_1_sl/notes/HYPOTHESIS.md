# SL Report Hypotheses — locked before tuning
# CS7641 Machine Learning, Summer 2026 | Timothy Leung (tleung37)
# Date locked: before any grid-search or final-fit experiments ran.

## Hypothesis clauses (from §II of SL_Report.tex)

On the 20k stratified Covertype subsample, with leakage-safe StandardScaler
fit only on training folds:

**Clause 1 — kNN leads on Macro-F1.**  
k-NN with small k outperforms all other learners on macro-F1. The continuous
geography features form locally-coherent neighbourhoods after standardisation,
and minority classes (4 Cottonwood/Willow, 5 Aspen, 6 Douglas-fir) cluster
tightly in elevation–soil space, which a low-k neighbourhood vote can lift more
readily than a globally-pruned decision rule.

**Clause 2 — RBF-SVM ranks second.**  
RBF-SVM ranks second, edging out an unpruned Decision Tree on balanced accuracy
because the kernel bandwidth implicitly smooths across rare-class neighbourhoods
that the axis-aligned tree splits too coarsely.

**Clause 3 — SGD-only MLPs rank fourth and fifth.**  
The two SGD-only MLPs rank fourth and fifth, separated mainly by scikit-learn's
early-stopping discipline versus the PyTorch model's patience-12 schedule;
both lag the kernel/neighbourhood pair because plain SGD with momentum=0 is
the worst optimiser the rubric allows on an ill-conditioned 54-dimensional input.

## Verdict (from §VII of SL_Report.tex)

- **Clause 1 (kNN dominates): REFUTED** — narrowly. kNN takes the
  balanced-accuracy crown (0.658) but loses macro-F1 by 0.028 to RBF-SVM.
- **Clause 2 (SVM second): REFUTED upward** — SVM led outright rather than
  placed second.
- **Clause 3 (SGD-only MLPs trail): CONFIRMED** — both MLPs finish in the
  bottom three on macro-F1; the persistent train-validation gap is the
  optimiser-deficit fingerprint.
