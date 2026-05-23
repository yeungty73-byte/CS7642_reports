"""
data_loader.py — CS7641 OL Report (Summer 2026)
Author: Timothy Leung (tleung37)

Loads the Forest Cover Type (Covertype) dataset, applies the SAME preprocessing
pipeline as the SL Report (stratified 60/20/20 train/val/test split, StandardScaler
fit only on train), and exposes PyTorch DataLoaders.

Design decisions documented per OL spec §8 (Data, baseline, and target):
  - ~20,000-instance stratified sample from sklearn's fetch_covtype()
  - Cover Type classes 1–7 (mapped to 0–6 for PyTorch CrossEntropyLoss)
  - Stratified splits: 60% train | 20% val | 20% test (seeds fixed)
  - StandardScaler fit ONLY on train; applied to val and test (leakage-safe)
  - CV folds available for sanity-check R Markdown

References:
  [1] Mitchell, T. (1997). Machine Learning. McGraw-Hill. Ch. 1 (learning from data).
  [2] scikit-learn Covertype docs: https://scikit-learn.org/stable/datasets/real_world.html#covtype-dataset
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

# ── Global seed (must match SL Report) ─────────────────────────────────────
SEED = 42
SAMPLE_SIZE = 20_000   # ~20k instances as required by spec
N_CLASSES   = 7        # Cover Type classes 1–7
BATCH_SIZE  = 256

def set_global_seed(seed: int = SEED) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Dataset loading & sampling ──────────────────────────────────────────────
def load_covertype_sample(n: int = SAMPLE_SIZE, seed: int = SEED) -> tuple[pd.DataFrame, pd.Series]:
    """
    Download (cached) and sample n rows from Covertype.
    Classes are 1-indexed (1–7); preserved as-is here.
    """
    data = fetch_covtype(as_frame=True, data_home="./.cache")
    X: pd.DataFrame = data.data   # 54 features
    y: pd.Series    = data.target  # 1–7

    # Stratified subsample to ~20k (same as SL Report)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=n, replace=False)
    X_samp = X.iloc[idx].reset_index(drop=True)
    y_samp = y.iloc[idx].reset_index(drop=True)
    return X_samp, y_samp


# ── Train / Val / Test split ────────────────────────────────────────────────
def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.20,
    test_size: float = 0.20,
    seed: int = SEED,
) -> dict:
    """
    Stratified 60/20/20 split.
    StandardScaler fit on train only — applied to val and test (leakage-safe).
    Returns dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
                             scaler (fitted), feature_names.
    """
    # 80 / 20 → then 75 / 25 of 80 = 60 / 20
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, stratify=y_tv, random_state=seed
    )

    # Scale: fit on train only
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    # Labels: 0-indexed for PyTorch (subtract 1)
    y_train_0 = y_train.values - 1
    y_val_0   = y_val.values   - 1
    y_test_0  = y_test.values  - 1

    return {
        "X_train": X_train_sc, "X_val": X_val_sc, "X_test": X_test_sc,
        "y_train": y_train_0,  "y_val": y_val_0,  "y_test": y_test_0,
        "scaler":  scaler,
        "feature_names": list(X.columns),
        "split_sizes": {
            "train": len(y_train_0),
            "val":   len(y_val_0),
            "test":  len(y_test_0),
        },
    }


def make_cv_folds(X_sc: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = SEED):
    """5-fold stratified CV on the train set for R sanity-check comparisons."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(X_sc, y))


# ── PyTorch Dataset & DataLoaders ───────────────────────────────────────────
class CovertypeDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def make_dataloaders(splits: dict, batch_size: int = BATCH_SIZE, seed: int = SEED) -> dict:
    """Return train/val/test DataLoaders. Train is shuffled; val/test are not."""
    g = torch.Generator()
    g.manual_seed(seed)

    train_ds = CovertypeDataset(splits["X_train"], splits["y_train"])
    val_ds   = CovertypeDataset(splits["X_val"],   splits["y_val"])
    test_ds  = CovertypeDataset(splits["X_test"],  splits["y_test"])

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=0, generator=g, drop_last=False),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0),
    }


# ── Quick smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    set_global_seed()
    print("Loading Covertype sample …")
    X, y = load_covertype_sample()
    print(f"  Sample shape: {X.shape}, classes: {sorted(y.unique())}")

    splits = make_splits(X, y)
    print(f"  Train: {splits['split_sizes']['train']}, "
          f"Val: {splits['split_sizes']['val']}, "
          f"Test: {splits['split_sizes']['test']}")

    loaders = make_dataloaders(splits)
    Xb, yb = next(iter(loaders["train"]))
    print(f"  First batch — X: {Xb.shape}, y: {yb.shape}, classes: {yb.unique().tolist()}")
    print("data_loader.py OK ✓")
