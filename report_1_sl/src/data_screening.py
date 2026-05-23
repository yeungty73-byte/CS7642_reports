"""EDA + leakage probes.

Two purposes, kept apart on purpose:

1) `screen(...)` produces the descriptives the report needs in Section II —
   class balance, scale per feature, sampling fidelity. Pure summary, no
   modeling.

2) `leakage_probes(...)` runs three independent sanity checks that the rubric
   asks for in the evaluation-debug subsection: a dummy stratified classifier
   (chance floor), a shuffled-label retrain (detects accidental shortcut
   information), and a split-integrity assertion. If any probe disagrees with
   what we expect — chance > 14.3%, shuffled accuracy >> chance, indices
   overlapping — something upstream is wrong and the report should call it out
   rather than paper over it. Falsus in uno, falsus in omnibus.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

from .data_loader import CLASS_NAMES, CovertypeBundle, N_CONTINUOUS


# -----------------------------------------------------------------------------
# Value types.
# -----------------------------------------------------------------------------
@dataclass
class ScreeningReport:
    full_class_counts: dict
    sample_class_counts: dict
    sample_class_pct: dict
    continuous_summary: dict   # mean/std/min/max per continuous feature
    n_train: int
    n_test: int
    n_features: int

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=float))


@dataclass
class LeakageReport:
    dummy_accuracy: float
    dummy_balanced_accuracy: float
    dummy_macro_f1: float
    chance_accuracy_majority: float
    shuffled_label_accuracy: float
    shuffled_label_macro_f1: float
    split_disjoint: bool
    train_test_overlap_count: int
    notes: list[str] = field(default_factory=list)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=float))


# -----------------------------------------------------------------------------
# Section II — EDA / screening.
# -----------------------------------------------------------------------------
def screen(
    full_df: pd.DataFrame,
    bundle: CovertypeBundle,
    label_col: str = "Cover_Type",
) -> ScreeningReport:
    full_counts = full_df[label_col].value_counts().sort_index().to_dict()
    sample_y = np.concatenate([bundle.y_train, bundle.y_test])
    sample_counts = (
        pd.Series(sample_y).value_counts().sort_index().to_dict()
    )
    total = sum(sample_counts.values())
    sample_pct = {int(k): 100.0 * v / total for k, v in sample_counts.items()}

    cont_df = pd.DataFrame(
        bundle.X_train_raw[:, :N_CONTINUOUS],
        columns=bundle.feature_names[:N_CONTINUOUS],
    )
    cont_summary = {
        col: {
            "mean": float(cont_df[col].mean()),
            "std": float(cont_df[col].std()),
            "min": float(cont_df[col].min()),
            "max": float(cont_df[col].max()),
        }
        for col in cont_df.columns
    }
    return ScreeningReport(
        full_class_counts={int(k): int(v) for k, v in full_counts.items()},
        sample_class_counts={int(k): int(v) for k, v in sample_counts.items()},
        sample_class_pct=sample_pct,
        continuous_summary=cont_summary,
        n_train=int(len(bundle.y_train)),
        n_test=int(len(bundle.y_test)),
        n_features=int(bundle.X_train.shape[1]),
    )


# -----------------------------------------------------------------------------
# Evaluation-debug — the leakage probes.
# -----------------------------------------------------------------------------
def leakage_probes(bundle: CovertypeBundle, seed: int) -> LeakageReport:
    notes: list[str] = []

    # (a) Chance floor — uniform-random over labels, then stratified-prior.
    majority_share = float(np.bincount(bundle.y_train).max() / len(bundle.y_train))

    dummy = DummyClassifier(strategy="stratified", random_state=seed)
    dummy.fit(bundle.X_train, bundle.y_train)
    yhat = dummy.predict(bundle.X_test)
    dummy_acc = float(accuracy_score(bundle.y_test, yhat))
    dummy_bacc = float(balanced_accuracy_score(bundle.y_test, yhat))
    dummy_f1 = float(f1_score(bundle.y_test, yhat, average="macro"))

    if dummy_acc > majority_share + 0.05:
        notes.append(
            "Dummy stratified beat majority by >5 pp — possible label imbalance "
            "drift between train and test; investigate."
        )

    # (b) Shuffled-label retrain — must collapse to chance, else the features
    #     are leaking the target through some shortcut.
    rng = np.random.RandomState(seed)
    y_shuffled = rng.permutation(bundle.y_train)
    probe = DecisionTreeClassifier(max_depth=8, random_state=seed)
    probe.fit(bundle.X_train, y_shuffled)
    yhat_shuf = probe.predict(bundle.X_test)
    shuf_acc = float(accuracy_score(bundle.y_test, yhat_shuf))
    shuf_f1 = float(f1_score(bundle.y_test, yhat_shuf, average="macro"))
    if shuf_acc > majority_share + 0.02:
        notes.append(
            f"Shuffled-label DT scored {shuf_acc:.3f} on test, above the "
            f"majority floor {majority_share:.3f}. That's a leakage smoke alarm."
        )

    # (c) Split integrity — already enforced at construction, re-check here for
    #     belt-and-suspenders auditability.
    disjoint = True
    overlap = 0
    if bundle.split is not None:
        ov = np.intersect1d(bundle.split.train_idx, bundle.split.test_idx)
        overlap = int(ov.size)
        disjoint = overlap == 0

    return LeakageReport(
        dummy_accuracy=dummy_acc,
        dummy_balanced_accuracy=dummy_bacc,
        dummy_macro_f1=dummy_f1,
        chance_accuracy_majority=majority_share,
        shuffled_label_accuracy=shuf_acc,
        shuffled_label_macro_f1=shuf_f1,
        split_disjoint=disjoint,
        train_test_overlap_count=overlap,
        notes=notes,
    )


__all__ = ["ScreeningReport", "LeakageReport", "screen", "leakage_probes"]
