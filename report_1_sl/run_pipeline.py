"""Main entry point — one command, one report's worth of artifacts.

Usage:
    python run_pipeline.py [--config configs/config.yaml] [--quick]

`--quick` shrinks the SVM tuning subsample and grid count to keep dev cycles
short. Production runs (the numbers that go in the PDF) use the default.

What this script does, in order, mirrors the report's section structure:
  II. Data screening + leakage probes        -> results/tables, .json logs
  III. Hyperparameter tuning (k-fold CV)     -> per-learner grid CSV
  IV.A Learning curves                       -> fig_learning_curves.png
  IV.B Model-complexity curves               -> fig_complexity_curves.png
  V.   Final fit + held-out test eval        -> confusion matrices, scores CSV
  VI.  Runtime profile                       -> fig_runtime.png

Each phase prints a progress line and writes its own log under results/logs/
so a failure halfway doesn't strand the partial output.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Make `src` importable as a package even when run from project root.
sys.path.insert(0, str(Path(__file__).parent))

from src.cv_optimizer import complexity_curve, learning_curve, tune
from src.data_loader import (
    CovertypeBundle, build_bundle, fetch_covertype_full,
)
from src.data_screening import leakage_probes, screen
from src.report_generator import (
    fig_all_confusions, fig_class_balance, fig_complexity_curves,
    fig_learning_curves, fig_nn_epoch_curves, fig_runtime,
    table_final_scores, table_grid, table_per_class,
)
from src.stat_learners import all_specs, TorchMLPClassifier
from src.styles import apply_style

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------------------------------------------------------
# Plumbing.
# -----------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(log_path: str) -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sl_report")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="w")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.handlers = [fh, sh]
    return logger


# -----------------------------------------------------------------------------
# Phases.
# -----------------------------------------------------------------------------
def phase_screen(bundle: CovertypeBundle, full_df, seed: int, log) -> dict:
    log.info("II. data screening + leakage probes")
    sr = screen(full_df, bundle)
    sr.to_json("results/logs/screening.json")
    lp = leakage_probes(bundle, seed=seed)
    lp.to_json("results/logs/leakage.json")
    fig_class_balance(sr.full_class_counts, sr.sample_class_counts)
    log.info(f"   dummy_acc={lp.dummy_accuracy:.3f}, "
             f"shuffled_acc={lp.shuffled_label_accuracy:.3f}, "
             f"split_disjoint={lp.split_disjoint}")
    return {"screening": sr, "leakage": lp}


def phase_tune(specs, bundle, cfg, seed, log) -> dict:
    log.info("III. k-fold CV hyperparameter tuning")
    tune_results = {}
    for name, spec in specs.items():
        t0 = time.perf_counter()
        log.info(f"   tuning {name} (display={spec.display})")
        try:
            res = tune(spec, bundle.X_train, bundle.y_train, cfg, seed=seed)
            tune_results[name] = res
            table_grid(name, res.grid_df)
            log.info(f"      done in {res.seconds:.1f}s | "
                     f"best macroF1={res.best_macro_f1:.4f} | "
                     f"params={res.best_params}")
        except Exception as e:
            log.exception(f"   FAILED tuning {name}: {e}")
    return tune_results


def phase_learning_curves(specs, tune_results, bundle, cfg, seed, log) -> dict:
    log.info("IV.A learning curves")
    curves = {}
    for name, spec in specs.items():
        if name not in tune_results:
            continue
        try:
            cr = learning_curve(spec, tune_results[name].best_params,
                                bundle.X_train, bundle.y_train, cfg, seed=seed)
            curves[name] = cr
            cr.to_frame().to_csv(f"results/tables/lc_{name}.csv", index=False)
            log.info(f"   {name} learning curve: "
                     f"final val={cr.points[-1].mean:.4f}")
        except Exception as e:
            log.exception(f"   FAILED learning curve {name}: {e}")
    fig_learning_curves(curves)
    return curves


def phase_complexity_curves(specs, tune_results, bundle, cfg, seed, log) -> dict:
    log.info("IV.B model-complexity curves")
    curves = {}
    for name, spec in specs.items():
        if name not in tune_results:
            continue
        try:
            fixed = dict(tune_results[name].best_params)
            cr = complexity_curve(spec, fixed,
                                  bundle.X_train, bundle.y_train, cfg, seed=seed)
            curves[name] = cr
            cr.to_frame().to_csv(f"results/tables/mc_{name}.csv", index=False)
            log.info(f"   {name} MC: x={cr.x_label} best val="
                     f"{max(p.mean for p in cr.points):.4f}")
        except Exception as e:
            log.exception(f"   FAILED MC {name}: {e}")
    fig_complexity_curves(curves)
    return curves


def phase_final(specs, tune_results, bundle, cfg, seed, log) -> dict:
    log.info("V. final fit + held-out test eval")
    preds = {}
    runtimes = {}
    best_params = {}
    sk_loss_curve = None
    pt_history = None
    for name, spec in specs.items():
        if name not in tune_results:
            continue
        params = dict(tune_results[name].best_params)
        best_params[name] = params
        est = spec.make(**params)
        t0 = time.perf_counter()
        try:
            est.fit(bundle.X_train, bundle.y_train)
        except Exception as e:
            log.exception(f"   FAILED final fit {name}: {e}")
            continue
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        yhat = est.predict(bundle.X_test)
        pred_s = time.perf_counter() - t0
        preds[name] = yhat
        runtimes[name] = {
            "fit": fit_s, "predict": pred_s,
            "tune_total": tune_results[name].seconds,
        }
        if name == "mlp_sk":
            sk_loss_curve = getattr(est, "loss_curve_", None)
        if name == "mlp_pt" and hasattr(est, "history_"):
            pt_history = est.history_
        table_per_class(name, bundle.y_test, yhat)
        log.info(f"   {name}: fit {fit_s:.1f}s, pred {pred_s:.2f}s")

    table_final_scores(preds, bundle.y_test, runtimes, best_params)
    fig_all_confusions(preds, bundle.y_test)
    fig_runtime(runtimes)
    fig_nn_epoch_curves(sk_loss_curve, pt_history)
    return {"preds": preds, "runtimes": runtimes, "best_params": best_params,
            "sk_loss_curve": sk_loss_curve, "pt_history": pt_history}


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------
def main(config_path: str = "configs/config.yaml", quick: bool = False) -> None:
    cfg = load_config(config_path)
    seed = cfg["experiment"]["seed"]
    apply_style()

    if quick:
        # Faster smoke run for development.
        cfg["data"]["sample_size"] = 4000
        cfg["models"]["svm"]["subsample_for_tuning"] = 2000
        cfg["models"]["svm"]["C_grid"] = [1.0, 10.0]
        cfg["models"]["svm"]["gamma_grid"] = ["scale", 0.01]
        cfg["curves"]["learning_curve_sizes"] = [0.10, 0.40, 1.00]
        cfg["models"]["mlp_torch"]["epochs"] = 25
        cfg["models"]["mlp_sklearn"]["max_iter"] = 80
        cfg["models"]["mlp_torch"]["architectures"] = [[64, 64], [128]]
        cfg["models"]["mlp_torch"]["activations"] = ["relu", "tanh"]

    Path("results/logs").mkdir(parents=True, exist_ok=True)
    log = setup_logging("results/logs/pipeline.log")
    log.info(f"config={config_path} | seed={seed} | quick={quick}")
    log.info(f"sample_size={cfg['data']['sample_size']} | "
             f"test_size={cfg['data']['test_size']} | "
             f"cv_folds={cfg['data']['cv_folds']}")

    # ---- I. Data ----
    t0 = time.perf_counter()
    full = fetch_covertype_full(cache_path=cfg["paths"]["data_cache"])
    bundle = build_bundle(
        sample_size=cfg["data"]["sample_size"],
        test_size=cfg["data"]["test_size"],
        seed=seed,
        cache_path=cfg["paths"]["data_cache"],
    )
    log.info(f"   data loaded: train={len(bundle.y_train)} "
             f"test={len(bundle.y_test)} features={bundle.X_train.shape[1]} "
             f"({time.perf_counter() - t0:.1f}s)")

    # ---- II. Screen ----
    phase_screen(bundle, full, seed, log)

    # ---- III. Tune ----
    specs = all_specs(cfg["models"], seed)
    tune_results = phase_tune(specs, bundle, cfg, seed, log)

    # ---- IV. Curves ----
    phase_learning_curves(specs, tune_results, bundle, cfg, seed, log)
    phase_complexity_curves(specs, tune_results, bundle, cfg, seed, log)

    # ---- V/VI. Final + runtime ----
    phase_final(specs, tune_results, bundle, cfg, seed, log)

    # Persist tune-result summary for the R sanity check to read.
    summary = {
        name: {
            "best_params": {k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in tr.best_params.items()},
            "best_macro_f1_cv": tr.best_macro_f1,
            "tune_seconds": tr.seconds,
        }
        for name, tr in tune_results.items()
    }
    Path("results/logs/tune_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    log.info("pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    main(config_path=args.config, quick=args.quick)
