"""Regenerates per-SUBJECT (not just aggregate) results for every intermediate stage of the
BCI-IV 2a progression FgMDM -> MEKT-default -> +window -> +corr estimator -> 8-member ensemble,
so the paired statistical tests in the manuscript (Section "statanalysis") are computed from
real, traceable per-subject CSVs end-to-end rather than from numbers whose original per-fold
computation was not preserved as a script. Two of the five stages (FgMDM, 8-member ensemble)
already have per-subject CSVs from existing scripts and are reused unmodified; this script fills
in the two missing intermediate per-subject CSVs ("+window" and "+window+corr", single-window
MEKT-lite at tmin=0.3/tmax=2.5 with the tuned k=50/lam=1.0/n_iter=20 hyperparameters, matching
final_results_table.csv rows 28 and 37 respectively).

    python -m paper.eval_mekt_progression
"""
import csv
import importlib
import os

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from paper import config
from paper.eval_mekt import mekt_lite
from paper.utils import classification_metrics

RESULTS_DIR = config.RESULTS_DIR
WINDOW = (0.3, 2.5)
MEKT_K, MEKT_LAM, MEKT_N_ITER = 50, 1.0, 20


def run_single_window_mekt(estimator: str, subjects=None):
    """MEKT-lite at the corrected (0.3, 2.5) window with the tuned hyperparameters, single
    covariance estimator ("lwf" = final_results_table.csv row 28, "corr" = row 37)."""
    config.set_seed(config.SEED)
    subjects = subjects or config.EVAL_SUBJECTS

    config.TMIN, config.TMAX = WINDOW
    import paper.data as data_mod
    importlib.reload(data_mod)  # get_paradigm() reads config.TMIN/TMAX at call time
    per_subject, label_encoder = data_mod.load_bnci_all(subjects, align="ea")
    classes = np.arange(len(label_encoder.classes_))
    cov = Covariances(estimator=estimator)

    fold_results = []
    for held_out in subjects:
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
        print(f"[mekt_window_{estimator}] held-out subject {held_out}: "
              f"acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

    accs = np.array([r["accuracy"] for r in fold_results])
    print(f"[mekt_window_{estimator}] LOSO accuracy: {accs.mean():.4f} +/- {accs.std():.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"zero_shot_mekt_window_{estimator}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "accuracy", "kappa"])
        writer.writeheader()
        for r in fold_results:
            writer.writerow(r)
    return fold_results


def main():
    # Restore config.TMIN/TMAX to the project default afterward so this script has no side
    # effect on any other script run in the same process/session.
    original_tmin, original_tmax = config.TMIN, config.TMAX
    try:
        run_single_window_mekt("lwf")
        run_single_window_mekt("corr")
    finally:
        config.TMIN, config.TMAX = original_tmin, original_tmax


if __name__ == "__main__":
    main()
