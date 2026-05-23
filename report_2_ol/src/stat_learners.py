"""
stat_learners.py — CS7641 OL Report (Summer 2026)
Author: Timothy Leung (tleung37)

PyTorch neural network backbone — IDENTICAL to SL Report final architecture.
All OL experiments (Parts 1–3) use this fixed backbone.

Architecture (finalized in SL Report):
  Input → FC(256) → BN → ReLU → Dropout(0.3) → FC(128) → BN → ReLU → FC(7)
  (BN and Dropout are inserted for OL study; removed/modified only in Part 3
   regularization experiments as documented.)

Key design decisions:
  - BatchNorm placed before ReLU (He et al. convention)
  - Dropout default=0.0 so SL backbone is no-dropout baseline;
    Part 3 varies this value
  - He initialization (kaiming_normal_) for ReLU activations
  - Forward passes track compute for gradient-evaluation accounting

References:
  [1] Mitchell, T. (1997). Machine Learning. Ch. 4 — NNs / gradient descent.
  [2] Kingma & Ba (2014). Adam: A Method for Stochastic Optimization.
      arXiv:1412.6980
  [3] Loshchilov & Hutter (2019). Decoupled Weight Decay Regularization.
      ICLR 2019. arXiv:1711.05101
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CovertypeNet(nn.Module):
    """
    Fixed backbone for all OL experiments.

    Architecture: FC(256) → BN → ReLU → FC(128) → BN → ReLU → FC(7)
    Dropout is parameterized so Part 3 can adjust it.

    Parameters
    ----------
    n_features : int
        Number of input features (54 for Covertype).
    n_classes : int
        Number of output classes (7 for Covertype).
    hidden_1 : int
        First hidden layer width.
    hidden_2 : int
        Second hidden layer width.
    dropout_rate : float
        Dropout probability (0.0 = no dropout, as in SL baseline).
    use_batchnorm : bool
        Whether to use Batch Normalisation (True for SL baseline).
    label_smoothing : float
        If > 0, passed to CrossEntropyLoss; used in Part 3.
    input_noise_std : float
        Std of Gaussian input noise during training only (Part 3).
    """

    def __init__(
        self,
        n_features: int = 54,
        n_classes: int = 7,
        hidden_1: int = 256,
        hidden_2: int = 128,
        dropout_rate: float = 0.0,
        use_batchnorm: bool = True,
        label_smoothing: float = 0.0,
        input_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_features    = n_features
        self.n_classes     = n_classes
        self.dropout_rate  = dropout_rate
        self.input_noise_std = input_noise_std

        # Build layers
        self.fc1  = nn.Linear(n_features, hidden_1)
        self.bn1  = nn.BatchNorm1d(hidden_1) if use_batchnorm else nn.Identity()
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2  = nn.Linear(hidden_1, hidden_2)
        self.bn2  = nn.BatchNorm1d(hidden_2) if use_batchnorm else nn.Identity()
        self.drop2 = nn.Dropout(dropout_rate)

        self.fc3  = nn.Linear(hidden_2, n_classes)

        # Loss function (supports label smoothing for Part 3)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Track trainable parameter count for RO budget checks
        self._init_weights()

    def _init_weights(self) -> None:
        """He (kaiming_normal_) initialisation for ReLU layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optional input noise — training only
        if self.training and self.input_noise_std > 0.0:
            x = x + torch.randn_like(x) * self.input_noise_std

        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)

    def trainable_params_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def final_layer_params(self, n_layers: int = 2) -> list[nn.Parameter]:
        """
        Return parameters from the final `n_layers` layers.
        Used for Part 1 RO (freeze all but final 1–3 layers).
        """
        all_layers = [self.fc1, self.bn1, self.fc2, self.bn2, self.fc3]
        rng = all_layers[-n_layers:]
        return [p for layer in rng for p in layer.parameters()]


def freeze_except_final(model: CovertypeNet, n_final_layers: int = 2) -> None:
    """Freeze all parameters except the final n_final_layers."""
    all_layers = [model.fc1, model.bn1, model.fc2, model.bn2, model.fc3]
    for layer in all_layers[:-n_final_layers]:
        for p in layer.parameters():
            p.requires_grad_(False)
    for layer in all_layers[-n_final_layers:]:
        for p in layer.parameters():
            p.requires_grad_(True)


def unfreeze_all(model: CovertypeNet) -> None:
    """Re-enable all gradients (for Part 2 / Part 3 full-network training)."""
    for p in model.parameters():
        p.requires_grad_(True)


# ── Training utilities ───────────────────────────────────────────────────────
def compute_val_loss(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> float:
    """
    Deterministic validation-loss computation.
    Sets model.eval() so dropout is disabled and BN uses running stats.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            total_loss += criterion(logits, y_batch).item() * len(y_batch)
            n += len(y_batch)
    return total_loss / n


def compute_metrics(
    model: nn.Module,
    loader,
    device: torch.device,
) -> dict:
    """
    Compute Accuracy, Macro-F1, Balanced Accuracy, and per-class metrics.
    All evaluation done with model.eval() (no dropout, deterministic BN).
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, balanced_accuracy_score,
        confusion_matrix, classification_report,
    )

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(device))
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)

    return {
        "accuracy":         float(accuracy_score(all_true, all_preds)),
        "macro_f1":         float(f1_score(all_true, all_preds, average="macro", zero_division=0)),
        "balanced_acc":     float(balanced_accuracy_score(all_true, all_preds)),
        "confusion_matrix": confusion_matrix(all_true, all_preds).tolist(),
        "per_class_report": classification_report(
            all_true, all_preds,
            target_names=[f"CT{i+1}" for i in range(7)],
            output_dict=True, zero_division=0
        ),
    }


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = CovertypeNet(n_features=54, n_classes=7, dropout_rate=0.0)
    print(f"  Total trainable params: {model.trainable_params_count():,}")
    x = torch.randn(8, 54)
    out = model(x)
    print(f"  Forward pass output: {out.shape}")  # (8, 7)

    freeze_except_final(model, n_final_layers=2)
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    unfrozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  After freeze: frozen param tensors={frozen}, "
          f"RO-trainable params={unfrozen:,}")
    print("stat_learners.py OK ✓")
