"""
optimizer.py — CS7641 OL Report (Summer 2026)
Author: Timothy Leung (tleung37)

Implements all three experimental parts:

  Part 1 — Randomized Optimization (RHC, SA, GA) on final 1–3 layers
  Part 2 — Adam-family optimizer ablations (7 variants)
  Part 3 — Regularization study (L2, early stopping, dropout, label smoothing,
            input noise) using best Part 2 Adam settings

All runs are budget-matched and reproducible (fixed seeds, gradient/function
evaluation counting, wall-clock timing).

Key references:
  [1] Kingma & Ba (2014). Adam. arXiv:1412.6980
  [2] Loshchilov & Hutter (2019). AdamW. ICLR 2019. arXiv:1711.05101
  [3] Mitchell (1997). Machine Learning. Ch. 9 — Randomized Optimization.
  [4] LaGrow (2025). SGD to AdamW (course reading).
"""

import copy
import time
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from data_loader import set_global_seed, SEED, make_dataloaders, load_covertype_sample, make_splits, BATCH_SIZE
from stat_learners import (
    CovertypeNet, freeze_except_final, unfreeze_all,
    compute_val_loss, compute_metrics,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGDIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOGDIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Adam-family custom optimizers
# ══════════════════════════════════════════════════════════════════════════════

class AdamNoBiasCorrection(torch.optim.Optimizer):
    """
    Adam without bias correction (ablation for Part 2).
    Update rule: θ_t = θ_{t-1} - α * m̂_t / (√v_t + ε)
    where m̂_t = β1*m_{t-1} + (1-β1)*g_t  (no bias correction)
          v_t  = β2*v_{t-1} + (1-β2)*g_t²  (no bias correction)
    This affects early-phase magnitudes; bias correction matters most for
    first several steps. [Kingma & Ba, 2014, §3]
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (β1, β2), eps = group["lr"], group["betas"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                m, v = state["m"], state["v"]
                m.mul_(β1).add_(g, alpha=1 - β1)   # no bias correction
                v.mul_(β2).addcmul_(g, g, value=1 - β2)
                p.add_(-lr * m / (v.sqrt() + eps))
        return loss


def get_optimizer(name: str, model: nn.Module, lr: float = 1e-3,
                  momentum: float = 0.9, weight_decay: float = 0.0,
                  betas=(0.9, 0.999)) -> torch.optim.Optimizer:
    """
    Factory for all 7 Part 2 optimizer variants.
    name options:
      'sgd_no_momentum'    — plain SGD
      'sgd_momentum'       — SGD + momentum
      'nesterov'           — SGD + Nesterov momentum
      'adam'               — standard Adam
      'adam_no_bias'       — Adam without bias correction (custom)
      'adam_b1_zero'       — Adam with β1=0 (RMSProp-like)
      'adamw'              — AdamW (decoupled weight decay)
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "sgd_no_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=weight_decay)
    elif name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, nesterov=False,
                               weight_decay=weight_decay)
    elif name == "nesterov":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, nesterov=True,
                               weight_decay=weight_decay)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif name == "adam_no_bias":
        return AdamNoBiasCorrection(params, lr=lr, betas=betas)
    elif name == "adam_b1_zero":
        return torch.optim.Adam(params, lr=lr, betas=(0.0, betas[1]),
                                weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP — gradient-based (Parts 2 & 3)
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, int]:
    """
    Train for one epoch.
    Returns (mean_train_loss, n_gradient_evaluations).
    One minibatch backward pass = 1 gradient evaluation.
    """
    model.train()
    total_loss, grad_evals = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        grad_evals += 1
    return total_loss / len(loader.dataset), grad_evals


def train_gradient_based(
    model: nn.Module,
    loaders: dict,
    optimizer: torch.optim.Optimizer,
    n_epochs: int = 100,
    device: torch.device = DEVICE,
    early_stopping_patience: Optional[int] = None,
    seed: int = SEED,
    label_smoothing: float = 0.0,
    run_name: str = "run",
) -> dict:
    """
    Full training loop with:
      - per-epoch val loss and metrics
      - gradient evaluation counting
      - wall-clock time tracking
      - optional early stopping (Part 3)

    Returns results dict saved to logs/{run_name}.json
    """
    set_global_seed(seed)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    model.to(device)

    history = {
        "run_name":      run_name,
        "seed":          seed,
        "train_loss":    [],
        "val_loss":      [],
        "grad_evals":    [],
        "wall_clock_s":  [],
        "total_grad_evals": 0,
    }

    best_val  = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    cumulative_grad_evals = 0
    t0 = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        tr_loss, n_grad = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        vl_loss = compute_val_loss(model, loaders["val"], device, criterion)
        cumulative_grad_evals += n_grad
        elapsed = time.perf_counter() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["grad_evals"].append(cumulative_grad_evals)
        history["wall_clock_s"].append(elapsed)

        # Track best
        if vl_loss < best_val:
            best_val = vl_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"    [{run_name}] Epoch {epoch:03d} | "
                  f"Train {tr_loss:.4f} | Val {vl_loss:.4f} | "
                  f"GEs: {cumulative_grad_evals:,} | {elapsed:.1f}s")

        # Early stopping (Part 3)
        if early_stopping_patience and patience_counter >= early_stopping_patience:
            print(f"    Early stopping at epoch {epoch} (patience={early_stopping_patience})")
            break

    history["total_grad_evals"] = cumulative_grad_evals
    history["best_val_loss"] = float(best_val)
    history["n_epochs_run"] = epoch

    # Restore best weights and compute final test metrics
    model.load_state_dict(best_state)
    test_metrics = compute_metrics(model, loaders["test"], device)
    history["test_metrics"] = test_metrics

    # Save log
    out = os.path.join(LOGDIR, f"{run_name}.json")
    with open(out, "w") as f:
        json.dump(history, f, indent=2)
    return history


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Randomized Optimization on final layers
# ══════════════════════════════════════════════════════════════════════════════

def _get_ro_params(model: nn.Module, n_final_layers: int = 2) -> np.ndarray:
    """Extract flat parameter vector from final n_final_layers."""
    all_layers = [model.fc1, model.bn1, model.fc2, model.bn2, model.fc3]
    layers = all_layers[-n_final_layers:]
    return np.concatenate([
        p.detach().cpu().numpy().ravel()
        for layer in layers for p in layer.parameters()
    ])


def _set_ro_params(model: nn.Module, vec: np.ndarray, n_final_layers: int = 2) -> None:
    """Write flat parameter vector back into final n_final_layers."""
    all_layers = [model.fc1, model.bn1, model.fc2, model.bn2, model.fc3]
    layers = all_layers[-n_final_layers:]
    idx = 0
    for layer in layers:
        for p in layer.parameters():
            n = p.numel()
            p.data.copy_(torch.tensor(vec[idx:idx+n], dtype=p.dtype).reshape(p.shape))
            idx += n


def _ro_objective(
    model: nn.Module,
    vec: np.ndarray,
    val_loader,
    n_final_layers: int,
    device: torch.device,
) -> float:
    """Validation loss for a candidate parameter vector (deterministic eval)."""
    _set_ro_params(model, vec, n_final_layers)
    return compute_val_loss(model, val_loader, device)


def run_rhc(
    model_init: nn.Module,
    loaders: dict,
    n_final_layers: int = 2,
    n_restarts: int = 3,
    n_iters_per_restart: int = 500,
    step_size: float = 0.01,
    seed: int = SEED,
    device: torch.device = DEVICE,
    run_name: str = "rhc",
) -> dict:
    """
    Randomized Hill Climbing on final `n_final_layers` layers.
    Disclosure (per spec):
      - Restart policy: n_restarts random restarts
      - Perturbation: Gaussian N(0, step_size²)
      - Stopping: after n_iters_per_restart function evaluations per restart
    """
    rng = np.random.default_rng(seed)
    set_global_seed(seed)
    model = copy.deepcopy(model_init).to(device)
    freeze_except_final(model, n_final_layers)
    val_loader = loaders["val"]

    best_loss = float("inf")
    best_vec  = _get_ro_params(model, n_final_layers).copy()
    func_evals = 0
    history = {"best_so_far": [], "func_evals": [], "wall_clock_s": []}
    t0 = time.perf_counter()

    # Evaluate starting point
    start_loss = _ro_objective(model, best_vec, val_loader, n_final_layers, device)
    best_loss = start_loss
    func_evals += 1
    history["best_so_far"].append(best_loss)
    history["func_evals"].append(func_evals)
    history["wall_clock_s"].append(0.0)

    for restart in range(n_restarts):
        # Random re-initialisation for each restart (except first)
        if restart > 0:
            current_vec = best_vec + rng.normal(0, step_size * 5, size=best_vec.shape)
        else:
            current_vec = best_vec.copy()
        current_loss = _ro_objective(model, current_vec, val_loader, n_final_layers, device)
        func_evals += 1

        for _ in range(n_iters_per_restart):
            candidate = current_vec + rng.normal(0, step_size, size=current_vec.shape)
            cand_loss = _ro_objective(model, candidate, val_loader, n_final_layers, device)
            func_evals += 1

            if cand_loss < current_loss:
                current_vec  = candidate
                current_loss = cand_loss
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_vec  = current_vec.copy()

            history["best_so_far"].append(best_loss)
            history["func_evals"].append(func_evals)
            history["wall_clock_s"].append(time.perf_counter() - t0)

    # Apply best params and compute test metrics
    _set_ro_params(model, best_vec, n_final_layers)
    test_metrics = compute_metrics(model, loaders["test"], device)

    result = {
        "run_name": run_name, "method": "RHC",
        "n_restarts": n_restarts, "step_size": step_size,
        "n_iters_per_restart": n_iters_per_restart,
        "n_final_layers": n_final_layers,
        "n_ro_params": len(best_vec),
        "best_val_loss": float(best_loss),
        "total_func_evals": func_evals,
        "test_metrics": test_metrics,
        "history": history,
    }
    out = os.path.join(LOGDIR, f"{run_name}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  RHC → best_val={best_loss:.5f}, FEs={func_evals}")
    return result


def run_sa(
    model_init: nn.Module,
    loaders: dict,
    n_final_layers: int = 2,
    n_iters: int = 2000,
    T_init: float = 1.0,
    T_decay: float = 0.995,
    step_size: float = 0.01,
    seed: int = SEED,
    device: torch.device = DEVICE,
    run_name: str = "sa",
) -> dict:
    """
    Simulated Annealing on final `n_final_layers` layers.
    Disclosure (per spec):
      - Initial temperature: T_init
      - Cooling schedule: T_t = T_init * T_decay^t (geometric)
      - Perturbation: Gaussian N(0, step_size²)
      - Acceptance: Metropolis criterion p = exp(-(Δloss)/T)
      - Stopping: after n_iters function evaluations
    """
    rng = np.random.default_rng(seed)
    set_global_seed(seed)
    model = copy.deepcopy(model_init).to(device)
    freeze_except_final(model, n_final_layers)
    val_loader = loaders["val"]

    current_vec = _get_ro_params(model, n_final_layers).copy()
    current_loss = _ro_objective(model, current_vec, val_loader, n_final_layers, device)
    best_loss = current_loss
    best_vec  = current_vec.copy()
    func_evals = 1
    history = {"best_so_far": [best_loss], "func_evals": [1],
               "wall_clock_s": [0.0], "temperature": [T_init]}
    T = T_init
    t0 = time.perf_counter()

    for it in range(1, n_iters + 1):
        candidate = current_vec + rng.normal(0, step_size, size=current_vec.shape)
        cand_loss = _ro_objective(model, candidate, val_loader, n_final_layers, device)
        func_evals += 1
        delta = cand_loss - current_loss

        if delta < 0 or (T > 1e-10 and rng.random() < np.exp(-delta / T)):
            current_vec  = candidate
            current_loss = cand_loss
            if current_loss < best_loss:
                best_loss = current_loss
                best_vec  = current_vec.copy()

        T *= T_decay
        history["best_so_far"].append(best_loss)
        history["func_evals"].append(func_evals)
        history["wall_clock_s"].append(time.perf_counter() - t0)
        history["temperature"].append(T)

    _set_ro_params(model, best_vec, n_final_layers)
    test_metrics = compute_metrics(model, loaders["test"], device)

    result = {
        "run_name": run_name, "method": "SA",
        "T_init": T_init, "T_decay": T_decay,
        "step_size": step_size, "n_iters": n_iters,
        "n_final_layers": n_final_layers, "n_ro_params": len(best_vec),
        "best_val_loss": float(best_loss),
        "total_func_evals": func_evals,
        "test_metrics": test_metrics,
        "history": history,
    }
    out = os.path.join(LOGDIR, f"{run_name}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  SA  → best_val={best_loss:.5f}, FEs={func_evals}")
    return result


def run_ga(
    model_init: nn.Module,
    loaders: dict,
    n_final_layers: int = 2,
    population_size: int = 20,
    n_generations: int = 50,
    mutation_std: float = 0.01,
    crossover_rate: float = 0.7,
    elitism_k: int = 2,
    seed: int = SEED,
    device: torch.device = DEVICE,
    run_name: str = "ga",
) -> dict:
    """
    Genetic Algorithm on final `n_final_layers` layers.
    Disclosure (per spec):
      - Population: population_size candidates
      - Selection: tournament (k=3)
      - Crossover: uniform crossover at crossover_rate
      - Mutation: Gaussian N(0, mutation_std²) per gene
      - Elitism: top elitism_k survive unchanged
      - Stopping: n_generations generations
    NOTE: Compare progress by FUNCTION EVALUATIONS, not generations.
          One generation with population_size candidates = ~population_size FEs.
    """
    rng = np.random.default_rng(seed)
    set_global_seed(seed)
    model = copy.deepcopy(model_init).to(device)
    freeze_except_final(model, n_final_layers)
    val_loader = loaders["val"]

    d = len(_get_ro_params(model, n_final_layers))

    def evaluate(pop: np.ndarray) -> np.ndarray:
        return np.array([
            _ro_objective(model, ind, val_loader, n_final_layers, device)
            for ind in pop
        ])

    def tournament(pop, fitness, k=3) -> np.ndarray:
        idx = rng.choice(len(pop), k, replace=False)
        best = idx[np.argmin(fitness[idx])]
        return pop[best].copy()

    # Initialise population around current weights
    init_vec = _get_ro_params(model, n_final_layers).copy()
    pop = init_vec + rng.normal(0, mutation_std * 5, size=(population_size, d))
    fitness = evaluate(pop)
    func_evals = population_size
    best_idx  = np.argmin(fitness)
    best_loss = float(fitness[best_idx])
    best_vec  = pop[best_idx].copy()
    history   = {"best_so_far": [best_loss], "func_evals": [func_evals],
                 "wall_clock_s": [0.0], "generation": [0]}
    t0 = time.perf_counter()

    for gen in range(1, n_generations + 1):
        new_pop = []

        # Elitism
        elite_idx = np.argsort(fitness)[:elitism_k]
        for ei in elite_idx:
            new_pop.append(pop[ei].copy())

        # Fill rest
        while len(new_pop) < population_size:
            p1 = tournament(pop, fitness)
            p2 = tournament(pop, fitness)
            # Uniform crossover
            mask = rng.random(d) < crossover_rate
            child = np.where(mask, p1, p2)
            # Mutation
            child += rng.normal(0, mutation_std, size=d)
            new_pop.append(child)

        pop = np.array(new_pop)
        fitness = evaluate(pop)
        func_evals += population_size

        gen_best = float(np.min(fitness))
        if gen_best < best_loss:
            best_loss = gen_best
            best_vec  = pop[np.argmin(fitness)].copy()

        history["best_so_far"].append(best_loss)
        history["func_evals"].append(func_evals)
        history["wall_clock_s"].append(time.perf_counter() - t0)
        history["generation"].append(gen)

    _set_ro_params(model, best_vec, n_final_layers)
    test_metrics = compute_metrics(model, loaders["test"], device)

    result = {
        "run_name": run_name, "method": "GA",
        "population_size": population_size, "n_generations": n_generations,
        "mutation_std": mutation_std, "crossover_rate": crossover_rate,
        "elitism_k": elitism_k, "n_final_layers": n_final_layers,
        "n_ro_params": d,
        "best_val_loss": float(best_loss),
        "total_func_evals": func_evals,
        "test_metrics": test_metrics,
        "history": history,
    }
    out = os.path.join(LOGDIR, f"{run_name}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  GA  → best_val={best_loss:.5f}, FEs={func_evals}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Full ablation sweep
# ══════════════════════════════════════════════════════════════════════════════

OPTIMIZER_CONFIGS = {
    "sgd_no_momentum": {"lr": 0.05,  "momentum": 0.0},
    "sgd_momentum":    {"lr": 0.05,  "momentum": 0.9},
    "nesterov":        {"lr": 0.05,  "momentum": 0.9},
    "adam":            {"lr": 1e-3,  "betas": (0.9, 0.999)},
    "adam_no_bias":    {"lr": 1e-3,  "betas": (0.9, 0.999)},
    "adam_b1_zero":    {"lr": 1e-3,  "betas": (0.0, 0.999)},
    "adamw":           {"lr": 1e-3,  "betas": (0.9, 0.999), "weight_decay": 1e-4},
}

N_EPOCHS_PART2 = 80


def run_part2_ablation(loaders: dict, seeds: list[int] = [42, 123, 7]) -> dict:
    """
    Run all 7 optimizer variants × 3 seeds.
    Returns dict of results keyed by optimizer name.
    """
    all_results = {}
    for opt_name, cfg in OPTIMIZER_CONFIGS.items():
        seed_results = []
        for seed in seeds:
            set_global_seed(seed)
            model = CovertypeNet(n_features=54, n_classes=7, dropout_rate=0.0).to(DEVICE)
            unfreeze_all(model)
            opt = get_optimizer(opt_name, model,
                                lr=cfg.get("lr", 1e-3),
                                momentum=cfg.get("momentum", 0.9),
                                weight_decay=cfg.get("weight_decay", 0.0),
                                betas=cfg.get("betas", (0.9, 0.999)))
            run_name = f"part2_{opt_name}_seed{seed}"
            print(f"  Running Part 2: {opt_name} (seed={seed}) …")
            hist = train_gradient_based(
                model, loaders, opt,
                n_epochs=N_EPOCHS_PART2,
                seed=seed,
                run_name=run_name,
            )
            seed_results.append(hist)
        all_results[opt_name] = seed_results
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — Regularization study
# ══════════════════════════════════════════════════════════════════════════════

BEST_ADAM_LR    = 1e-3
BEST_ADAM_BETAS = (0.9, 0.999)
N_EPOCHS_PART3  = 80

REGULARIZATION_CONFIGS = {
    "adam_baseline":     {"dropout": 0.0, "l2": 0.0,    "early_stopping": None,
                          "label_smoothing": 0.0, "input_noise_std": 0.0},
    "l2_1e-4":          {"dropout": 0.0, "l2": 1e-4,   "early_stopping": None,
                          "label_smoothing": 0.0, "input_noise_std": 0.0},
    "l2_1e-3":          {"dropout": 0.0, "l2": 1e-3,   "early_stopping": None,
                          "label_smoothing": 0.0, "input_noise_std": 0.0},
    "early_stop_p10":   {"dropout": 0.0, "l2": 0.0,    "early_stopping": 10,
                          "label_smoothing": 0.0, "input_noise_std": 0.0},
    "dropout_0.2":      {"dropout": 0.2, "l2": 0.0,    "early_stopping": None,
                          "label_smoothing": 0.0, "input_noise_std": 0.0},
    "dropout_0.3":      {"dropout": 0.3, "l2": 0.0,    "early_stopping": None,
                          "label_smoothing": 0.0, "input_noise_std": 0.0},
    "label_smooth_0.1": {"dropout": 0.0, "l2": 0.0,    "early_stopping": None,
                          "label_smoothing": 0.1, "input_noise_std": 0.0},
    "input_noise_0.05": {"dropout": 0.0, "l2": 0.0,    "early_stopping": None,
                          "label_smoothing": 0.0, "input_noise_std": 0.05},
    "best_combo":       {"dropout": 0.2, "l2": 1e-4,   "early_stopping": 10,
                          "label_smoothing": 0.1, "input_noise_std": 0.0},
}


def run_part3_regularization(loaders: dict, seeds: list[int] = [42, 123, 7]) -> dict:
    """
    Run all regularization configurations × 3 seeds using best Part 2 Adam settings.
    """
    all_results = {}
    for reg_name, cfg in REGULARIZATION_CONFIGS.items():
        seed_results = []
        for seed in seeds:
            set_global_seed(seed)
            model = CovertypeNet(
                n_features=54, n_classes=7,
                dropout_rate=cfg["dropout"],
                label_smoothing=cfg["label_smoothing"],
                input_noise_std=cfg["input_noise_std"],
            ).to(DEVICE)
            unfreeze_all(model)
            opt = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=BEST_ADAM_LR,
                betas=BEST_ADAM_BETAS,
                weight_decay=cfg["l2"],
            )
            run_name = f"part3_{reg_name}_seed{seed}"
            print(f"  Running Part 3: {reg_name} (seed={seed}) …")
            hist = train_gradient_based(
                model, loaders, opt,
                n_epochs=N_EPOCHS_PART3,
                early_stopping_patience=cfg["early_stopping"],
                label_smoothing=cfg["label_smoothing"],
                seed=seed,
                run_name=run_name,
            )
            seed_results.append(hist)
        all_results[reg_name] = seed_results
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# SENSITIVITY HEATMAPS (Part 2)
# ══════════════════════════════════════════════════════════════════════════════

def run_sensitivity_heatmap(
    loaders: dict,
    lr_grid: list = [1e-4, 5e-4, 1e-3, 5e-3],
    beta1_grid: list = [0.5, 0.8, 0.9, 0.95],
    beta2_grid: list = [0.9, 0.99, 0.999],
    seed: int = SEED,
    n_epochs: int = 30,
) -> dict:
    """
    3×4 coarse sensitivity heatmap for Adam over (alpha, beta1).
    Saved to logs/heatmap_alpha_beta1.json
    """
    results = {}
    for lr in lr_grid:
        for b1 in beta1_grid:
            set_global_seed(seed)
            model = CovertypeNet(54, 7, dropout_rate=0.0).to(DEVICE)
            unfreeze_all(model)
            opt = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=lr, betas=(b1, 0.999),
            )
            criterion = nn.CrossEntropyLoss()
            model.train()
            for epoch in range(n_epochs):
                for Xb, yb in loaders["train"]:
                    opt.zero_grad()
                    loss = criterion(model(Xb.to(DEVICE)), yb.to(DEVICE))
                    loss.backward()
                    opt.step()
            val_loss = compute_val_loss(model, loaders["val"], DEVICE)
            key = f"lr={lr}_b1={b1}"
            results[key] = {"lr": lr, "beta1": b1, "val_loss": float(val_loss)}
            print(f"    Heatmap {key}: val_loss={val_loss:.4f}")

    out = os.path.join(LOGDIR, "heatmap_alpha_beta1.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — run all parts
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== optimizer.py — Running all parts ===")
    set_global_seed()

    # Load data
    X, y = load_covertype_sample()
    splits = make_splits(X, y)
    loaders = make_dataloaders(splits, batch_size=BATCH_SIZE)

    # Establish SL baseline (SGD, no momentum, same architecture as SL Report)
    print("\n-- SL PyTorch SGD baseline --")
    sl_model = CovertypeNet(54, 7, dropout_rate=0.0).to(DEVICE)
    sl_opt   = torch.optim.SGD(sl_model.parameters(), lr=0.05, momentum=0.0)
    sl_hist  = train_gradient_based(sl_model, loaders, sl_opt,
                                    n_epochs=80, run_name="sl_sgd_baseline")
    print(f"  SL baseline: val_loss={sl_hist['best_val_loss']:.4f}, "
          f"test_acc={sl_hist['test_metrics']['accuracy']:.4f}")

    # Part 1 — RO fine-tuning on final 2 layers
    print("\n-- Part 1: Randomized Optimization --")
    frozen_model = copy.deepcopy(sl_model)
    rhc_result = run_rhc(frozen_model, loaders, n_final_layers=2,
                          n_restarts=3, n_iters_per_restart=300,
                          step_size=0.01, seed=SEED)
    sa_result  = run_sa(frozen_model, loaders, n_final_layers=2,
                         n_iters=1500, T_init=1.0, T_decay=0.995,
                         step_size=0.01, seed=SEED)
    ga_result  = run_ga(frozen_model, loaders, n_final_layers=2,
                         population_size=20, n_generations=40,
                         mutation_std=0.01, crossover_rate=0.7,
                         elitism_k=2, seed=SEED)

    # Part 2 — Adam-family ablations
    print("\n-- Part 2: Adam-family ablations --")
    part2_results = run_part2_ablation(loaders, seeds=[42, 123, 7])

    # Part 2 — Sensitivity heatmap
    print("\n-- Part 2: Sensitivity heatmap --")
    heatmap = run_sensitivity_heatmap(loaders)

    # Part 3 — Regularization study
    print("\n-- Part 3: Regularization study --")
    part3_results = run_part3_regularization(loaders, seeds=[42, 123, 7])

    print("\noptimizer.py OK ✓")
