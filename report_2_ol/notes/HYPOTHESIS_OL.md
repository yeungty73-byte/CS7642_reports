# OL Report Hypotheses — locked before running Parts 2 and 3
# CS7641 Machine Learning, Summer 2026 | Timothy Leung (tleung37)
# Date locked: 2026-05-23 (before any Part 2 or Part 3 experiments)

## H1 — Optimizer speed (locked before Part 2)

Adam-family optimizers will reach the fixed validation-loss threshold ℓ
faster than plain SGD (no momentum) because adaptive second-moment scaling
(Kingma & Ba, 2014) reduces sensitivity to the heterogeneous gradient scales
in Covertype's mix of continuous elevations and binary soil/wilderness indicators.

This speed advantage will NOT uniformly transfer to Macro-F1 improvement,
because majority-class confidence can improve (reducing CE loss) without
correcting minority-class decision boundaries (which determines Macro-F1).

## H2 — Regularization vs. optimizer (locked before Part 3)

Regularization — specifically dropout or early stopping — will improve
Macro-F1 and balanced accuracy more than optimizer choice alone, because
regularization controls the full training trajectory and reduces co-adaptation
in representations that confuse similar cover types (CT1 vs CT7, CT2 vs CT5).

## H3 — RO fine-tuning cost (locked before Part 1)

RO fine-tuning on the final two layers will produce limited Macro-F1 gains
relative to its function-evaluation cost, because gradient descent already
found a well-conditioned basin and the ≤50k-parameter final-layer space is
too well-explored by gradient history for population-based or temperature-
driven search to improve substantially.

## Verdict (to be filled after experiments)

H1: [TBD — will be updated in report §IX]
H2: [TBD — will be updated in report §IX]
H3: [TBD — will be updated in report §IX]
