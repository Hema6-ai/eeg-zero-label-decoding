"""Factorial preprocessing-vs-architecture experiment (TMLR revision directive, mandatory Phase
IV). Full orthogonal grid for the two CPU-fast, non-learned method families (no retraining needed
per cell): time window (4 levels) x covariance estimator (2 levels) x method (FgMDM, MEKT-lite
default). SPDNet (the deep-learning representative) is added at the default window only, both
estimators (see paper/train_spdnet.py --cov-estimator and reports/factorial_experiment_report.md
for why a full 4-window x SPDNet retrain grid was judged not worth the additional ~6-8 GPU-hours
this session, given the two fast factors already estimate the window effect within the
classical/domain-adaptation methods, and the estimator effect is checked across ALL THREE method
families at the default window).

Hyperparameters are NOT retuned per cell -- MEKT always uses k=100/lam=1.0/5 iterations (the
BCI-IV 2a "default" config already established before this experiment), never the window-specific
tuned configs from Section 8 of the manuscript. This keeps this experiment a clean preprocessing
ablation, not a second round of test-set hyperparameter search.

Evaluation: strict zero-shot LOSO on BCI-IV 2a (9 subjects), the same protocol as every other
experiment in this project. Every cell's per-subject predictions are saved.

    python -m paper.eval_factorial
"""
import csv
import importlib
import os

import numpy as np
from pyriemann.classification import FgMDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from paper import config
from paper.eval_mekt import mekt_lite
from paper.utils import classification_metrics

RESULTS_DIR = config.RESULTS_DIR
OUT_PREFIX = "factorial"
WINDOWS = [(0.0, 3.0), (0.3, 2.5), (0.5, 2.5), (0.5, 3.0)]
ESTIMATORS = ["lwf", "corr"]
MEKT_K, MEKT_LAM, MEKT_N_ITER = 100, 1.0, 5  # fixed "default" config throughout -- see docstring


def _cell_key(method, tmin, tmax, estimator):
    return f"{method}_t{tmin}_{tmax}_{estimator}"


def _fold_csv_path(cell_key):
    return os.path.join(RESULTS_DIR, f"{OUT_PREFIX}_{cell_key}.csv")


def _load_done_folds(cell_key):
    path = _fold_csv_path(cell_key)
    done = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                done[int(row["held_out_subject"])] = {
                    "accuracy": float(row["accuracy"]), "kappa": float(row["kappa"])}
    return done


def _append_fold_csv(cell_key, held_out, accuracy, kappa):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = _fold_csv_path(cell_key)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "accuracy", "kappa"])
        if write_header:
            writer.writeheader()
        writer.writerow({"held_out_subject": held_out, "accuracy": accuracy, "kappa": kappa})


def load_window_data(tmin, tmax, subjects):
    """Reloads BCI-IV 2a at a given analysis window, EA-aligned -- mirrors the pattern already
    established in paper/eval_mekt_ensemble.py / paper/eval_mekt_progression.py (config.TMIN/TMAX
    read at paradigm-construction time, so must reload paper.data after changing them)."""
    config.TMIN, config.TMAX = tmin, tmax
    import paper.data as data_mod
    importlib.reload(data_mod)
    per_subject, label_encoder = data_mod.load_bnci_all(subjects, align="ea")
    return per_subject, label_encoder


def run_fgmdm_cell(per_subject, subjects, estimator, cell_key):
    done = _load_done_folds(cell_key)
    fold_results = [{"held_out_subject": s, **r} for s, r in done.items()]
    cov = Covariances(estimator=estimator)
    for held_out in subjects:
        if held_out in done:
            continue
        source_subjects = [s for s in subjects if s != held_out]
        source_X = np.concatenate([per_subject[s][0] for s in source_subjects], axis=0).astype(np.float64)
        source_y = np.concatenate([per_subject[s][1] for s in source_subjects], axis=0)
        target_X, target_y = per_subject[held_out]

        source_covs = cov.fit_transform(source_X)
        target_covs = cov.transform(target_X.astype(np.float64))
        clf = FgMDM(metric="riemann", tsupdate=False)
        clf.fit(source_covs, source_y)
        pred = clf.predict(target_covs)

        metrics = classification_metrics(target_y, pred)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        _append_fold_csv(cell_key, held_out, metrics["accuracy"], metrics["kappa"])
    return fold_results


def run_mekt_cell(per_subject, subjects, estimator, cell_key, classes):
    done = _load_done_folds(cell_key)
    fold_results = [{"held_out_subject": s, **r} for s, r in done.items()]
    cov = Covariances(estimator=estimator)
    for held_out in subjects:
        if held_out in done:
            continue
        source_subjects = [s for s in subjects if s != held_out]
        source_X = np.concatenate([per_subject[s][0] for s in source_subjects], axis=0).astype(np.float64)
        source_y = np.concatenate([per_subject[s][1] for s in source_subjects], axis=0)
        target_X, target_y = per_subject[held_out]
        target_X = target_X.astype(np.float64)

        source_covs = cov.fit_transform(source_X)
        target_covs = cov.transform(target_X)
        ts = TangentSpace(metric="riemann")
        Xs = ts.fit_transform(source_covs)
        Xt = ts.transform(target_covs)
        k_eff = min(MEKT_K, Xs.shape[1])

        pred = mekt_lite(Xs, source_y, Xt, classes, k=k_eff, lam=MEKT_LAM, n_iter=MEKT_N_ITER)
        metrics = classification_metrics(target_y, pred)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        _append_fold_csv(cell_key, held_out, metrics["accuracy"], metrics["kappa"])
    return fold_results


def main():
    config.set_seed(config.SEED)
    subjects = config.EVAL_SUBJECTS
    original_tmin, original_tmax = config.TMIN, config.TMAX
    summary_rows = []

    try:
        for tmin, tmax in WINDOWS:
            per_subject, label_encoder = load_window_data(tmin, tmax, subjects)
            classes = np.arange(len(label_encoder.classes_))
            for estimator in ESTIMATORS:
                for method_name, run_fn in [("fgmdm", run_fgmdm_cell), ("mekt_default", run_mekt_cell)]:
                    cell_key = _cell_key(method_name, tmin, tmax, estimator)
                    if method_name == "fgmdm":
                        fold_results = run_fn(per_subject, subjects, estimator, cell_key)
                    else:
                        fold_results = run_fn(per_subject, subjects, estimator, cell_key, classes)
                    accs = np.array([r["accuracy"] for r in fold_results])
                    kappas = np.array([r["kappa"] for r in fold_results])
                    print(f"[{cell_key}] acc={accs.mean()*100:.2f}% +/- {accs.std()*100:.2f}% "
                          f"kappa={kappas.mean():.3f}")
                    summary_rows.append({
                        "method": method_name, "tmin": tmin, "tmax": tmax, "estimator": estimator,
                        "n_subjects": len(fold_results), "accuracy_mean": accs.mean(),
                        "accuracy_std": accs.std(), "kappa_mean": kappas.mean(), "kappa_std": kappas.std(),
                    })
    finally:
        config.TMIN, config.TMAX = original_tmin, original_tmax

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, f"{OUT_PREFIX}_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "tmin", "tmax", "estimator", "n_subjects",
                                                 "accuracy_mean", "accuracy_std", "kappa_mean", "kappa_std"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved {len(summary_rows)} cells to {summary_path}")


if __name__ == "__main__":
    main()
