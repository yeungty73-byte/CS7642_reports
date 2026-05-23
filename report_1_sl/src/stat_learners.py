"""Five learner wrappers — one knob-set per learner, all returning the
same `LearnerSpec` envelope so the CV optimizer can iterate uniformly.

Rubric constraints encoded here, NOT in the configs:
- `MLPClassifier` is locked to `solver="sgd"`, `momentum=0.0`,
  `nesterovs_momentum=False`. The user cannot pass anything else through.
- The PyTorch MLP uses plain `torch.optim.SGD` with `momentum=0`. No Adam,
  no Nesterov, no Adagrad/RMSprop. Activation function is a sweep axis for
  the extra-credit ablation (ReLU/GELU/SiLU/tanh).

Every wrapper exposes `make(...)` returning a scikit-learn-compatible
estimator and `complexity_axis()` returning the hyperparameter the report
uses on the x-axis of the model-complexity curve.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted


# =============================================================================
# Shared envelope.
# =============================================================================
@dataclass
class LearnerSpec:
    name: str                    # short tag e.g. "dt", "knn"
    display: str                 # pretty label e.g. "Decision Tree"
    make: Callable[..., Any]     # factory(**hyperparams) -> estimator
    param_grid: dict             # {hyperparam: iterable_of_values}
    complexity_param: str        # name of the hyperparam on MC x-axis
    extra: dict = field(default_factory=dict)


# =============================================================================
# 1) Decision Tree — Mitchell Ch 3. Pruning via ccp_alpha + depth cap.
# =============================================================================
def make_decision_tree(
    ccp_alpha: float = 0.0,
    max_depth: int | None = None,
    criterion: str = "gini",
    random_state: int = 7641,
) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        ccp_alpha=ccp_alpha,
        random_state=random_state,
    )


def dt_spec(cfg: dict, seed: int) -> LearnerSpec:
    g = cfg["decision_tree"]
    return LearnerSpec(
        name="dt", display="Decision Tree",
        make=lambda **kw: make_decision_tree(random_state=seed, **kw),
        param_grid={
            "ccp_alpha": g["ccp_alpha_grid"],
            "max_depth": g["max_depth_grid"],
        },
        complexity_param="ccp_alpha",
        extra={"criterion": g["criterion"]},
    )


# =============================================================================
# 2) k-NN — Mitchell Ch 8. Scaling lives upstream in data_loader.
# =============================================================================
def make_knn(
    n_neighbors: int = 7, weights: str = "uniform", p: int = 2,
) -> KNeighborsClassifier:
    return KNeighborsClassifier(
        n_neighbors=n_neighbors, weights=weights, p=p, n_jobs=-1,
    )


def knn_spec(cfg: dict, seed: int) -> LearnerSpec:
    g = cfg["knn"]
    return LearnerSpec(
        name="knn", display="k-Nearest Neighbours",
        make=lambda **kw: make_knn(**kw),
        param_grid={"n_neighbors": g["k_grid"], "weights": g["weights"]},
        complexity_param="n_neighbors",
        extra={"metric": g["metric"], "p": g["p"]},
    )


# =============================================================================
# 3) SVM — Burges 1998, Schölkopf NIPS slides. Linear + RBF.
# =============================================================================
def make_svm(
    kernel: str = "rbf", C: float = 1.0, gamma: Any = "scale",
    random_state: int = 7641,
) -> SVC:
    return SVC(
        kernel=kernel, C=C, gamma=gamma,
        cache_size=512, random_state=random_state,
    )


def svm_spec(cfg: dict, seed: int) -> LearnerSpec:
    g = cfg["svm"]
    return LearnerSpec(
        name="svm", display="Support Vector Machine",
        make=lambda **kw: make_svm(random_state=seed, **kw),
        param_grid={
            "kernel": g["kernels"],
            "C": g["C_grid"],
            "gamma": g["gamma_grid"],
        },
        complexity_param="C",
        extra={
            "cv_folds_svm": g["cv_folds_svm"],
            "subsample_for_tuning": g["subsample_for_tuning"],
        },
    )


# =============================================================================
# 4) sklearn MLPClassifier — SGD only, momentum=0, no Nesterov. Locked.
# =============================================================================
def make_mlp_sklearn(
    hidden_layer_sizes: tuple = (64, 64),
    alpha: float = 1e-4,
    learning_rate_init: float = 0.01,
    max_iter: int = 200,
    early_stopping: bool = True,
    validation_fraction: float = 0.15,
    n_iter_no_change: int = 15,
    random_state: int = 7641,
) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=tuple(hidden_layer_sizes),
        activation="relu",
        solver="sgd",                     # rubric-locked.
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        momentum=0.0,                     # rubric-locked.
        nesterovs_momentum=False,         # rubric-locked.
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        random_state=random_state,
    )


def mlp_sklearn_spec(cfg: dict, seed: int) -> LearnerSpec:
    g = cfg["mlp_sklearn"]
    return LearnerSpec(
        name="mlp_sk", display="MLP (sklearn, SGD)",
        make=lambda **kw: make_mlp_sklearn(
            random_state=seed,
            learning_rate_init=g["learning_rate_init"],
            max_iter=g["max_iter"],
            early_stopping=g["early_stopping"],
            validation_fraction=g["validation_fraction"],
            n_iter_no_change=g["n_iter_no_change"],
            **kw,
        ),
        param_grid={
            "hidden_layer_sizes": [tuple(h) for h in g["hidden_layer_sizes"]],
            "alpha": g["alpha_grid"],
        },
        complexity_param="hidden_layer_sizes",
    )


# =============================================================================
# 5) PyTorch MLP — plain SGD, momentum=0. Activation sweep is the EC.
# =============================================================================
ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": F.relu, "gelu": F.gelu, "silu": F.silu, "tanh": torch.tanh,
}


class _TorchMLP(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dims: list[int], out_dim: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        dims = [in_dim, *hidden_dims, out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation!r}")
        self.act = ACTIVATIONS[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """A scikit-learn-compatible wrapper around a plain-SGD PyTorch MLP.

    Implements `fit`, `predict`, `predict_proba`, `score` so it slots into the
    same CV/learning-curve harness as the other four learners. Loss curves
    (train/val per epoch) are exposed via `self.history_` for the NN epoch
    figure the rubric asks for.
    """

    def __init__(
        self,
        hidden_dims: tuple = (64, 64),
        activation: str = "relu",
        lr: float = 0.05,
        batch_size: int = 128,
        epochs: int = 80,
        weight_decay: float = 1e-4,
        early_stop_patience: int = 12,
        validation_fraction: float = 0.15,
        random_state: int = 7641,
        device: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.early_stop_patience = early_stop_patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    # ---- sklearn API ----
    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLPClassifier":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        device = torch.device(
            self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        classes_ = np.unique(y)
        self.classes_ = classes_
        # Map labels to 0..K-1 contiguously.
        self._label_to_idx = {c: i for i, c in enumerate(classes_)}
        y_idx = np.vectorize(self._label_to_idx.get)(y).astype(np.int64)

        n = len(X)
        n_val = max(int(n * self.validation_fraction), 64)
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        Xtr = torch.from_numpy(X[tr_idx]).float().to(device)
        ytr = torch.from_numpy(y_idx[tr_idx]).long().to(device)
        Xva = torch.from_numpy(X[val_idx]).float().to(device)
        yva = torch.from_numpy(y_idx[val_idx]).long().to(device)

        self.net_ = _TorchMLP(
            in_dim=X.shape[1],
            hidden_dims=list(self.hidden_dims),
            out_dim=len(classes_),
            activation=self.activation,
        ).to(device)

        opt = torch.optim.SGD(
            self.net_.parameters(),
            lr=self.lr, momentum=0.0,            # rubric-locked.
            weight_decay=self.weight_decay, nesterov=False,
        )
        loss_fn = nn.CrossEntropyLoss()

        history = {"train_loss": [], "val_loss": [],
                   "train_acc": [], "val_acc": []}

        best_val = float("inf")
        bad_epochs = 0
        best_state = None

        for ep in range(self.epochs):
            self.net_.train()
            order = torch.randperm(len(Xtr), device=device)
            tot_loss, tot_correct = 0.0, 0
            for i in range(0, len(Xtr), self.batch_size):
                sel = order[i: i + self.batch_size]
                xb, yb = Xtr[sel], ytr[sel]
                opt.zero_grad()
                logits = self.net_(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                tot_loss += loss.item() * len(xb)
                tot_correct += (logits.argmax(1) == yb).sum().item()
            tr_loss = tot_loss / len(Xtr)
            tr_acc = tot_correct / len(Xtr)

            self.net_.eval()
            with torch.no_grad():
                vlogits = self.net_(Xva)
                vloss = loss_fn(vlogits, yva).item()
                vacc = (vlogits.argmax(1) == yva).float().mean().item()

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(vloss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(vacc)

            if vloss < best_val - 1e-4:
                best_val = vloss
                bad_epochs = 0
                best_state = {k: v.detach().clone()
                              for k, v in self.net_.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.early_stop_patience:
                    if self.verbose:
                        print(f"early stop @ epoch {ep+1}")
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.history_ = history
        self.n_epochs_run_ = len(history["train_loss"])
        self._device = device
        return self

    def _forward_numpy(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "net_")
        self.net_.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X).float().to(self._device)
            logits = self.net_(X_t)
        return logits.detach().cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self._forward_numpy(X)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = self._forward_numpy(X)
        idx = logits.argmax(axis=1)
        return self.classes_[idx]


def mlp_torch_spec(cfg: dict, seed: int) -> LearnerSpec:
    g = cfg["mlp_torch"]
    return LearnerSpec(
        name="mlp_pt", display="MLP (PyTorch, SGD)",
        make=lambda **kw: TorchMLPClassifier(
            random_state=seed,
            lr=g["learning_rate"],
            batch_size=g["batch_size"],
            epochs=g["epochs"],
            weight_decay=g["weight_decay"],
            early_stop_patience=g["early_stop_patience"],
            **kw,
        ),
        param_grid={
            "hidden_dims": [tuple(a) for a in g["architectures"]],
            "activation": g["activations"],
        },
        complexity_param="hidden_dims",
        extra={"epochs": g["epochs"]},
    )


# =============================================================================
# Registry.
# =============================================================================
def all_specs(cfg_models: dict, seed: int) -> dict[str, LearnerSpec]:
    return {
        "dt": dt_spec(cfg_models, seed),
        "knn": knn_spec(cfg_models, seed),
        "svm": svm_spec(cfg_models, seed),
        "mlp_sk": mlp_sklearn_spec(cfg_models, seed),
        "mlp_pt": mlp_torch_spec(cfg_models, seed),
    }


__all__ = [
    "LearnerSpec", "TorchMLPClassifier",
    "make_decision_tree", "make_knn", "make_svm",
    "make_mlp_sklearn", "all_specs",
    "dt_spec", "knn_spec", "svm_spec", "mlp_sklearn_spec", "mlp_torch_spec",
]
