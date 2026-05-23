"""Covertype data loader, sampler, and split factory.

Single source of truth for *what data went into each learner*. Everything
downstream — screening, tuning, learning curves — reads from this module so
that swapping the sample seed or size touches exactly one file.

Key invariants enforced here, not in the learners:
- Stratification on Cover_Type at every level (subsample, train/test, k-fold).
- Scaler fit on the train fold only, then applied to the validation/test split.
  Continuous features (cols 0–9) get standardized; binary indicator columns
  (cols 10–53: 4 wilderness + 40 soil) pass through untouched.
- A small `SplitArtifact` dataclass carries the row indices so the report can
  prove no rows leaked across folds.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Constants — locked to the Covertype schema, not user-tunable.
# -----------------------------------------------------------------------------
N_CONTINUOUS = 10  # Elevation, Aspect, Slope, Hor/Vert dist, Hillshades 9/12/3PM, dist-fire.
N_WILDERNESS = 4
N_SOIL = 40
N_FEATURES = N_CONTINUOUS + N_WILDERNESS + N_SOIL  # 54

FEATURE_NAMES: list[str] = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
] + [f"Wilderness_Area_{i+1}" for i in range(N_WILDERNESS)] \
  + [f"Soil_Type_{i+1}" for i in range(N_SOIL)]

CLASS_NAMES = {
    1: "Spruce/Fir", 2: "Lodgepole Pine", 3: "Ponderosa Pine",
    4: "Cottonwood/Willow", 5: "Aspen", 6: "Douglas-fir", 7: "Krummholz",
}


# -----------------------------------------------------------------------------
# Lightweight value types.
# -----------------------------------------------------------------------------
@dataclass
class SplitArtifact:
    """All row indices needed to prove no leakage. Indices reference the
    *post-subsample* DataFrame (length == sample_size)."""
    train_idx: np.ndarray
    test_idx: np.ndarray
    sample_size: int
    seed: int

    def assert_disjoint(self) -> None:
        overlap = np.intersect1d(self.train_idx, self.test_idx)
        if overlap.size:
            raise RuntimeError(
                f"Train/test split leaks {overlap.size} rows. "
                f"Seed={self.seed}; abort the run."
            )


@dataclass
class CovertypeBundle:
    """Everything a learner needs in one envelope."""
    X_train_raw: np.ndarray
    X_test_raw: np.ndarray
    X_train: np.ndarray  # scaled (continuous cols only)
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    class_names: dict = field(default_factory=lambda: dict(CLASS_NAMES))
    split: SplitArtifact | None = None
    scaler: StandardScaler | None = None


# -----------------------------------------------------------------------------
# Loader.
# -----------------------------------------------------------------------------
def fetch_covertype_full(cache_path: str | os.PathLike | None = None) -> pd.DataFrame:
    """Pull the full ~581k Covertype, returns a DataFrame with named columns
    plus a `Cover_Type` integer label (1–7)."""
    if cache_path is not None:
        cp = Path(cache_path)
        if cp.exists():
            with np.load(cp) as npz:
                X, y = npz["X"], npz["y"]
            df = pd.DataFrame(X, columns=FEATURE_NAMES)
            df["Cover_Type"] = y.astype(int)
            return df

    raw = fetch_covtype(as_frame=False)
    X = raw.data.astype(np.float32)
    y = raw.target.astype(np.int64)  # already 1–7
    if cache_path is not None:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cp, X=X, y=y)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["Cover_Type"] = y
    return df


def stratified_subsample(
    df: pd.DataFrame, n: int, seed: int, label_col: str = "Cover_Type"
) -> pd.DataFrame:
    """Draw a class-stratified subsample of *n* rows. The minority classes
    (Cottonwood/Willow ≈ 0.5%, Aspen ≈ 1.6%) are tiny in the full ~581k
    population; stratification keeps their absolute count above ~100 here
    so per-class metrics aren't degenerate."""
    if n >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    keep_idx, _ = next(sss.split(df.index, df[label_col]))
    return df.iloc[keep_idx].reset_index(drop=True)


# -----------------------------------------------------------------------------
# Splitting + scaling.
# -----------------------------------------------------------------------------
def train_test_split_stratified(
    df: pd.DataFrame, test_size: float, seed: int, label_col: str = "Cover_Type"
) -> tuple[pd.DataFrame, pd.DataFrame, SplitArtifact]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(sss.split(df.index, df[label_col]))
    art = SplitArtifact(
        train_idx=train_idx, test_idx=test_idx,
        sample_size=len(df), seed=seed,
    )
    art.assert_disjoint()
    return df.iloc[train_idx].reset_index(drop=True), \
           df.iloc[test_idx].reset_index(drop=True), \
           art


def fit_transform_train_only(
    X_train: np.ndarray, X_test: np.ndarray, n_continuous: int = N_CONTINUOUS
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Standardize continuous columns using train statistics ONLY. Binary
    indicator columns pass through. This is the leakage-safe contract: any
    pipeline that does it differently must say so explicitly."""
    scaler = StandardScaler()
    X_tr_cont = scaler.fit_transform(X_train[:, :n_continuous])
    X_te_cont = scaler.transform(X_test[:, :n_continuous])
    X_tr = np.hstack([X_tr_cont, X_train[:, n_continuous:]]).astype(np.float32)
    X_te = np.hstack([X_te_cont, X_test[:, n_continuous:]]).astype(np.float32)
    return X_tr, X_te, scaler


def build_bundle(
    sample_size: int,
    test_size: float,
    seed: int,
    cache_path: str | None = None,
    label_col: str = "Cover_Type",
) -> CovertypeBundle:
    """End-to-end: full pull -> stratified subsample -> stratified train/test
    -> train-only scaler. Returns one envelope downstream code can rely on."""
    full = fetch_covertype_full(cache_path=cache_path)
    sub = stratified_subsample(full, n=sample_size, seed=seed, label_col=label_col)
    train_df, test_df, art = train_test_split_stratified(
        sub, test_size=test_size, seed=seed, label_col=label_col
    )
    y_train = train_df[label_col].to_numpy()
    y_test = test_df[label_col].to_numpy()
    X_train_raw = train_df.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    X_test_raw = test_df.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    X_train, X_test, scaler = fit_transform_train_only(X_train_raw, X_test_raw)

    return CovertypeBundle(
        X_train_raw=X_train_raw, X_test_raw=X_test_raw,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        split=art, scaler=scaler,
    )


# -----------------------------------------------------------------------------
# CV factory — every learner reaches for this; do NOT instantiate KFold ad hoc.
# -----------------------------------------------------------------------------
def stratified_kfold(n_splits: int, seed: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def cv_indices(
    y: np.ndarray, n_splits: int, seed: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) for each fold. Indices are positional in y."""
    skf = stratified_kfold(n_splits, seed)
    yield from skf.split(np.zeros(len(y)), y)
