"""8-member MEKT-lite ensemble across 4 analysis windows x 2 covariance estimators -- the
headline zero-shot LOSO result on BCI-IV 2a (52.39%, kappa=0.365). Extracted and adapted from a
verified historical Kaggle deployment run (kaggle_kernel/nested_loso_push/run_kernel.py); that
run's saved output (kaggle_kernel/output_nested_loso/results/nested_loso_result.csv) selected
this exact ensemble configuration in all 9 of 9 outer LOSO folds and reproduced these numbers to
the reported precision -- see paper/results/mekt_ensemble_verified_reference.csv for that
historical output. Windows, estimators, and MEKT hyperparameters below are fixed at the verified
values, not re-tuned here.

This is the base, equal-weight ensemble only. The manuscript additionally reports a
confidence-weighted refinement (52.53%); that variant's original implementation has not been
recovered and is not reproduced by this script (see README).

    python -m paper.eval_mekt_ensemble
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

# Verified configuration -- see module docstring for provenance. Do not tune.
WINDOWS = [(0.3, 2.5), (0.5, 2.5), (0.5, 2.0), (0.5, 3.0)]  # seconds, relative to cue onset
ESTIMATORS = ["lwf", "corr"]
MEKT_K = 50
MEKT_LAM = 1.0
MEKT_N_ITER = 10  # BCI-IV 2a value. The separate PhysioNet replication (Section 9) uses 20.


def _precompute_covariances(subjects):
    """Loads BCI-IV 2a once per (window, estimator) and returns per-subject covariance matrices
    keyed by (tmin, tmax, estimator). Euclidean Alignment is per-subject and unsupervised (valid
    for the held-out subject in every LOSO fold, since it uses only that subject's own trials),
    so covariances can be precomputed once across all folds rather than recomputed per fold."""
    per_subject_covs = {}
    per_subject_labels = {}
    label_encoder = None
    for tmin, tmax in WINDOWS:
        config.TMIN, config.TMAX = tmin, tmax
        import paper.data as data_mod
        importlib.reload(data_mod)  # get_paradigm() reads config.TMIN/TMAX at call time
        per_subject, label_encoder = data_mod.load_bnci_all(subjects, align="ea")
        for subj in subjects:
            per_subject_labels[subj] = per_subject[subj][1]
        for est in ESTIMATORS:
            cov = Covariances(estimator=est)
            per_subject_covs[(tmin, tmax, est)] = {
                subj: cov.fit_transform(per_subject[subj][0].astype(np.float64)) for subj in subjects
            }
    classes = np.arange(len(label_encoder.classes_))
    return per_subject_covs, per_subject_labels, classes


def _member_proba(covs_dict, labels, source_subjects, target_subject, classes):
    """One ensemble member's predicted probabilities for one (window, estimator) configuration:
    covariance -> tangent space -> MEKT-lite. Source labels are used to fit; the target subject's
    labels never appear here -- MEKT-lite pseudo-labels the target transductively from its own
    unlabeled features (paper.eval_mekt.mekt_lite)."""
    source_covs = np.concatenate([covs_dict[s] for s in source_subjects], axis=0)
    source_y = np.concatenate([labels[s] for s in source_subjects], axis=0)
    target_covs = covs_dict[target_subject]

    ts = TangentSpace(metric="riemann")
    Xs = ts.fit_transform(source_covs)
    Xt = ts.transform(target_covs)
    k_eff = min(MEKT_K, Xs.shape[1])
    _, proba = mekt_lite(Xs, source_y, Xt, classes, k=k_eff, lam=MEKT_LAM, n_iter=MEKT_N_ITER,
                          return_proba=True)
    return proba


def run(subjects=None):
    """Standard zero-shot LOSO over BCI-IV 2a: for each held-out subject, the other 8 subjects
    form the source pool for every ensemble member. The 8 members' predicted probabilities (4
    windows x 2 estimators) are averaged with equal weight; the held-out subject's true labels
    are used only to score the resulting prediction, never to fit, pseudo-label, or select
    between members."""
    config.set_seed(config.SEED)
    subjects = subjects or config.EVAL_SUBJECTS
    candidate_keys = [(tmin, tmax, est) for tmin, tmax in WINDOWS for est in ESTIMATORS]

    covs, labels, classes = _precompute_covariances(subjects)

    fold_results = []
    for held_out in subjects:
        source_subjects = [s for s in subjects if s != held_out]
        member_probs = [
            _member_proba(covs[key], labels, source_subjects, held_out, classes)
            for key in candidate_keys
        ]
        ensemble_proba = sum(member_probs) / len(member_probs)
        pred = ensemble_proba.argmax(axis=1)

        metrics = classification_metrics(labels[held_out], pred)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        print(f"[mekt-ensemble] held-out subject {held_out}: "
              f"acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

    accs = np.array([r["accuracy"] for r in fold_results])
    kappas = np.array([r["kappa"] for r in fold_results])
    summary = {"accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
               "kappa_mean": kappas.mean(), "kappa_std": kappas.std()}
    print(f"[mekt-ensemble] LOSO accuracy: {summary['accuracy_mean']:.4f} +/- "
          f"{summary['accuracy_std']:.4f}  kappa: {summary['kappa_mean']:.4f} +/- "
          f"{summary['kappa_std']:.4f}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "zero_shot_mekt_ensemble.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "accuracy", "kappa"])
        writer.writeheader()
        for r in fold_results:
            writer.writerow(r)

    return summary, fold_results


if __name__ == "__main__":
    run()
