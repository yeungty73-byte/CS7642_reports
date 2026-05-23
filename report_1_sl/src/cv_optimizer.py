"""K-fold CV hyperparameter optimizer + learning- and complexity-curve harness.

Three public entry points, each used by `run_pipeline.py`:

- `tune(spec, X, y, cfg)`      → exhaustive grid over `spec.param_grid`, scored
                                 by macro-F1 (rubric-aligned, imbalance-aware).
                                 Returns the best-params dict, a results
                                 DataFrame, and the wall-clock cost in seconds.

- `learning_curve(spec, params, X, y, cfg)`
                                → for each training-size fraction in
                                  cfg.curves.learning_curve_sizes, do a 3-fold
                                  inner CV; report mean ± std macro-F1.

- `complexity_curve(spec, params, X, y, cfg)`
                                → sweep `spec.complexity_param` across its grid
                                  while holding other params at their best; CV
                                  on each value.

Runtime is tracked per fit/predict so the report's runtime table doesn't lie.
SVM gets a smaller subsample for tuning (per config) — that's the
runtime-justified shortcut Section IV-C will own up to."""
from __future__ import annotations

import itertools
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.utils import check_random_state

from .data_loader import stratified_kfold
from .stat_learners import LearnerSpec


# -----------------------------------------------------------------------------
# Result envelopes.
# -----------------------------------------------------------------------------
@dataclass
class TuneResult:
    learner: str
    best_params: dict
    best_macro_f1: float
    best_accuracy: float
    grid_df: pd.DataFrame
    seconds: float


@dataclass
class CurvePoint:
    x: Any
    mean: float
    std: float
    fold_scores: list[float] = field(default_factory=list)
    fit_seconds: float = 0.0


@dataclass
class CurveResult:
    learner: str
    kind: str               # "learning" | "complexity"
    x_label: str
    points: list[CurvePoint]
    train_points: list[CurvePoint] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for p in self.points:
            rows.append({"x": p.x, "split": "val", "mean": p.mean, "std": p.std,
                         "fit_seconds": p.fit_seconds})
        for p in self.train_points:
            rows.append({"x": p.x, "split": "train", "mean": p.mean, "std": p.std,
                         "fit_seconds": p.fit_seconds})
        return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Helpers.
# -----------------------------------------------------------------------------
def _score_fold(estimator, X_tr, y_tr, X_va, y_va) -> dict[str, float]:
    t0 = time.perf_counter()
    estimator.fit(X_tr, y_tr)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    yhat = estimator.predict(X_va)
    t_pred = time.perf_counter() - t0
    return {
        "macro_f1": float(f1_score(y_va, yhat, average="macro")),
        "accuracy": float(accuracy_score(y_va, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(y_va, yhat)),
        "fit_seconds": t_fit,
        "pred_seconds": t_pred,
    }


def _param_combos(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*grid.values())]


# -----------------------------------------------------------------------------
# Tune.
# -----------------------------------------------------------------------------
def tune(
    spec: LearnerSpec,
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    n_splits: int | None = None,
    seed: int = 7641,
    subsample: int | None = None,
) -> TuneResult:
    """Exhaustive k-fold CV over `spec.param_grid`."""
    nfold = n_splits or cfg["data"]["cv_folds"]
    if spec.name == "svm":
        nfold = cfg["models"]["svm"].get("cv_folds_svm", nfold)
        subsample = subsample or cfg["models"]["svm"].get("subsample_for_tuning")

    if subsample is not None and subsample < len(y):
        rng = check_random_state(seed)
        idx = rng.permutation(len(y))[:subsample]
        # Maintain stratification: re-sample with class proportions.
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=subsample, random_state=seed)
        idx, _ = next(sss.split(np.zeros(len(y)), y))
        X_t, y_t = X[idx], y[idx]
    else:
        X_t, y_t = X, y

    rows: list[dict] = []
    combos = _param_combos(spec.param_grid)
    t_total0 = time.perf_counter()
    for combo in combos:
        skf = stratified_kfold(nfold, seed)
        fold_scores = []
        fold_acc = []
        fold_fit_t = []
        for tr_idx, va_idx in skf.split(np.zeros(len(y_t)), y_t):
            est = spec.make(**combo)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = _score_fold(est,
                                X_t[tr_idx], y_t[tr_idx],
                                X_t[va_idx], y_t[va_idx])
            fold_scores.append(m["macro_f1"])
            fold_acc.append(m["accuracy"])
            fold_fit_t.append(m["fit_seconds"])
        row = {
            **{k: (str(v) if isinstance(v, tuple) else v) for k, v in combo.items()},
            "macro_f1_mean": float(np.mean(fold_scores)),
            "macro_f1_std": float(np.std(fold_scores)),
            "accuracy_mean": float(np.mean(fold_acc)),
            "fit_seconds_mean": float(np.mean(fold_fit_t)),
        }
        # Stash the original (typed) combo for re-instantiation later.
        row["_combo"] = combo
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
    best_combo = df.iloc[0]["_combo"]
    return TuneResult(
        learner=spec.name,
        best_params=best_combo,
        best_macro_f1=float(df.iloc[0]["macro_f1_mean"]),
        best_accuracy=float(df.iloc[0]["accuracy_mean"]),
        grid_df=df.drop(columns=["_combo"]),
        seconds=time.perf_counter() - t_total0,
    )


# -----------------------------------------------------------------------------
# Learning curve.
# -----------------------------------------------------------------------------
def learning_curve(
    spec: LearnerSpec,
    params: dict,
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    seed: int = 7641,
) -> CurveResult:
    sizes = cfg["curves"]["learning_curve_sizes"]
    n_splits = cfg["curves"]["cv_folds_lc"]
    skf = stratified_kfold(n_splits, seed)

    points: list[CurvePoint] = []
    train_points: list[CurvePoint] = []

    for frac in sizes:
        val_scores, train_scores, fit_ts = [], [], []
        for tr_idx, va_idx in skf.split(np.zeros(len(y)), y):
            from sklearn.model_selection import StratifiedShuffleSplit
            if frac < 1.0:
                sss = StratifiedShuffleSplit(
                    n_splits=1, train_size=float(frac), random_state=seed,
                )
                sub, _ = next(sss.split(np.zeros(len(tr_idx)), y[tr_idx]))
                use_tr = tr_idx[sub]
            else:
                use_tr = tr_idx
            est = spec.make(**params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t0 = time.perf_counter()
                est.fit(X[use_tr], y[use_tr])
                fit_ts.append(time.perf_counter() - t0)
                val_scores.append(f1_score(y[va_idx], est.predict(X[va_idx]),
                                           average="macro"))
                train_scores.append(f1_score(y[use_tr], est.predict(X[use_tr]),
                                             average="macro"))
        points.append(CurvePoint(
            x=frac, mean=float(np.mean(val_scores)), std=float(np.std(val_scores)),
            fold_scores=val_scores, fit_seconds=float(np.mean(fit_ts)),
        ))
        train_points.append(CurvePoint(
            x=frac, mean=float(np.mean(train_scores)),
            std=float(np.std(train_scores)),
            fold_scores=train_scores, fit_seconds=float(np.mean(fit_ts)),
        ))
    return CurveResult(
        learner=spec.name, kind="learning",
        x_label="training-set fraction",
        points=points, train_points=train_points,
    )


# -----------------------------------------------------------------------------
# Model-complexity curve.
# -----------------------------------------------------------------------------
def complexity_curve(
    spec: LearnerSpec,
    fixed_params: dict,
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    seed: int = 7641,
) -> CurveResult:
    """Sweep `spec.complexity_param` across its grid while holding other
    hyperparams at `fixed_params`. CV with cfg.data.cv_folds (or smaller for
    SVM, per cfg)."""
    cparam = spec.complexity_param
    if cparam not in spec.param_grid:
        raise KeyError(f"{spec.name}: complexity_param {cparam!r} not in grid")
    values = list(spec.param_grid[cparam])

    nfold = cfg["data"]["cv_folds"]
    if spec.name == "svm":
        nfold = cfg["models"]["svm"].get("cv_folds_svm", nfold)
    skf = stratified_kfold(nfold, seed)

    use_X, use_y = X, y
    if spec.name == "svm":
        sub_n = cfg["models"]["svm"].get("subsample_for_tuning")
        if sub_n and sub_n < len(y):
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=sub_n, random_state=seed)
            idx, _ = next(sss.split(np.zeros(len(y)), y))
            use_X, use_y = X[idx], y[idx]

    val_points, train_points = [], []
    for v in values:
        params = dict(fixed_params)
        params[cparam] = v
        val_scores, train_scores, fit_ts = [], [], []
        for tr_idx, va_idx in skf.split(np.zeros(len(use_y)), use_y):
            est = spec.make(**params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t0 = time.perf_counter()
                est.fit(use_X[tr_idx], use_y[tr_idx])
                fit_ts.append(time.perf_counter() - t0)
                val_scores.append(f1_score(use_y[va_idx], est.predict(use_X[va_idx]),
                                           average="macro"))
                train_scores.append(f1_score(use_y[tr_idx], est.predict(use_X[tr_idx]),
                                             average="macro"))
        # x-axis tag — coerce unhashable like tuples to string for plotting.
        xtag = str(v) if isinstance(v, (tuple, list)) else v
        val_points.append(CurvePoint(
            x=xtag, mean=float(np.mean(val_scores)),
            std=float(np.std(val_scores)), fold_scores=val_scores,
            fit_seconds=float(np.mean(fit_ts)),
        ))
        train_points.append(CurvePoint(
            x=xtag, mean=float(np.mean(train_scores)),
            std=float(np.std(train_scores)), fold_scores=train_scores,
            fit_seconds=float(np.mean(fit_ts)),
        ))
    return CurveResult(
        learner=spec.name, kind="complexity", x_label=cparam,
        points=val_points, train_points=train_points,
    )


__all__ = ["TuneResult", "CurveResult", "CurvePoint",
           "tune", "learning_curve", "complexity_curve"]
