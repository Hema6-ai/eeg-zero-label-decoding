"""Learned (stacked) ensemble instead of fixed-weight soft-voting, which failed twice (FgMDM+RTv2,
FgMDM+MEKT both underperformed or matched their best single component). Idea: fit a small
meta-classifier (logistic regression) on [FgMDM_probs, MEKT_probs, ...] -> true class, so the
combination weights are LEARNED per-class rather than a single global blend ratio.

To avoid leaking the held-out target subject anywhere, the meta-classifier's training data is
built via an INNER leave-one-source-subject-out loop: for each of the 8 source subjects, FgMDM and
MEKT are trained on the OTHER 7 source subjects and used to predict that one held-out source
subject's probabilities. Stacking those 8 inner out-of-fold predictions (with true labels, since
these are all source subjects) gives a leak-free meta-training set. The meta-classifier is then
applied to FgMDM/MEKT predictions made by models trained on all 8 source subjects, evaluated on
the actual (outer) held-out target subject.

    python -m paper.eval_stacked_ensemble
"""
import csv
import os

import numpy as np
from pyriemann.classification import FgMDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression

from paper import config
from paper.data import load_bnci_all
from paper.eval_mekt import mekt_lite
from paper.utils import classification_metrics

MEKT_KWARGS = dict(k=50, lam=1.0, n_iter=5)


def _fgmdm_probs(source_X, source_y, eval_X, classes):
    cov = Covariances(estimator="lwf")
    source_covs = cov.fit_transform(source_X)
    eval_covs = cov.transform(eval_X)
    clf = FgMDM(metric="riemann", tsupdate=False)
    clf.fit(source_covs, source_y)
    probs_raw = clf.predict_proba(eval_covs)
    probs = np.zeros((len(eval_X), len(classes)))
    for i, c in enumerate(clf.classes_):
        probs[:, c] = probs_raw[:, i]
    return probs


def _mekt_probs(source_X, source_y, eval_X, classes):
    cov = Covariances(estimator="lwf")
    source_covs = cov.fit_transform(source_X)
    eval_covs = cov.transform(eval_X)
    ts = TangentSpace(metric="riemann")
    Xs = ts.fit_transform(source_covs)
    Xe = ts.transform(eval_covs)
    k_eff = min(MEKT_KWARGS["k"], Xs.shape[1])
    _, probs = mekt_lite(Xs, source_y, Xe, classes, k=k_eff, lam=MEKT_KWARGS["lam"],
                          n_iter=MEKT_KWARGS["n_iter"], return_proba=True)
    return probs


def run(subjects=None):
    config.set_seed(config.SEED)
    subjects = subjects or config.EVAL_SUBJECTS
    per_subject, label_encoder = load_bnci_all(subjects, align="ea")
    classes = np.arange(len(label_encoder.classes_))

    fold_results = {"fgmdm": [], "mekt": [], "stacked": []}
    for held_out in subjects:
        source_subjects = [s for s in subjects if s != held_out]

        # --- Inner LOSO over source subjects to build a leak-free meta-training set ---
        meta_X, meta_y = [], []
        for inner_held_out in source_subjects:
            inner_source = [s for s in source_subjects if s != inner_held_out]
            inner_source_X = np.concatenate([per_subject[s][0] for s in inner_source], axis=0).astype(np.float64)
            inner_source_y = np.concatenate([per_subject[s][1] for s in inner_source], axis=0)
            inner_eval_X, inner_eval_y = per_subject[inner_held_out]
            inner_eval_X = inner_eval_X.astype(np.float64)

            p_fgmdm = _fgmdm_probs(inner_source_X, inner_source_y, inner_eval_X, classes)
            p_mekt = _mekt_probs(inner_source_X, inner_source_y, inner_eval_X, classes)
            meta_X.append(np.concatenate([p_fgmdm, p_mekt], axis=1))
            meta_y.append(inner_eval_y)

        meta_X = np.concatenate(meta_X, axis=0)
        meta_y = np.concatenate(meta_y, axis=0)
        meta_clf = LogisticRegression(max_iter=1000, C=1.0)
        meta_clf.fit(meta_X, meta_y)

        # --- Outer fold: FgMDM/MEKT trained on all 8 source subjects, evaluated on target ---
        source_X = np.concatenate([per_subject[s][0] for s in source_subjects], axis=0).astype(np.float64)
        source_y = np.concatenate([per_subject[s][1] for s in source_subjects], axis=0)
        target_X, target_y = per_subject[held_out]
        target_X = target_X.astype(np.float64)

        p_fgmdm_target = _fgmdm_probs(source_X, source_y, target_X, classes)
        p_mekt_target = _mekt_probs(source_X, source_y, target_X, classes)
        meta_features = np.concatenate([p_fgmdm_target, p_mekt_target], axis=1)
        pred_stacked = meta_clf.predict(meta_features)

        pred_fgmdm = p_fgmdm_target.argmax(axis=1)
        pred_mekt = p_mekt_target.argmax(axis=1)

        for key, pred in [("fgmdm", pred_fgmdm), ("mekt", pred_mekt), ("stacked", pred_stacked)]:
            m = classification_metrics(target_y, pred)
            m["held_out_subject"] = held_out
            fold_results[key].append(m)

        print(f"[stacked] held-out subject {held_out}: "
              f"fgmdm={fold_results['fgmdm'][-1]['accuracy']:.4f} "
              f"mekt={fold_results['mekt'][-1]['accuracy']:.4f} "
              f"STACKED={fold_results['stacked'][-1]['accuracy']:.4f}")

    summaries = {}
    for key, results in fold_results.items():
        accs = np.array([r["accuracy"] for r in results])
        kappas = np.array([r["kappa"] for r in results])
        summaries[key] = {"accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
                           "kappa_mean": kappas.mean(), "kappa_std": kappas.std()}
        print(f"[{key}] LOSO accuracy: {summaries[key]['accuracy_mean']:.4f} +/- "
              f"{summaries[key]['accuracy_std']:.4f}  kappa: {summaries[key]['kappa_mean']:.4f} +/- "
              f"{summaries[key]['kappa_std']:.4f}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "zero_shot_stacked_ensemble.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "accuracy", "kappa"])
        writer.writeheader()
        for r in fold_results["stacked"]:
            writer.writerow(r)

    return summaries, fold_results


if __name__ == "__main__":
    run()
