"""Riemannian Procrustes Analysis (Rodrigues et al., 2018) -- genuinely different mechanism from
MEKT (linear tangent-space projection) or EA/RA (covariance whitening only). After EA-whitening
both domains, RPA finds an explicit ORTHOGONAL ROTATION that aligns the target subject's
(pseudo-labeled) per-class means to the source pool's per-class means, refining pseudo-labels
over a few iterations (transductive, same zero-shot-legal structure as MEKT: no true target
labels used anywhere, only its own unlabeled features).

Simplification note: the original RPA paper rotates the actual SPD covariance matrices via
conjugation (X -> U X U^T on the manifold). Here the rotation is solved via orthogonal Procrustes
directly in EUCLIDEAN tangent space (after TangentSpace already linearized the covariances) --
a tractable approximation of the same idea (rigid alignment of class structure) rather than a
manifold-native rotation, implemented this way for speed and to reuse the existing pipeline.

    python -m paper.eval_rpa
"""
import csv
import os

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.linalg import orthogonal_procrustes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from paper import config
from paper.data import load_bnci_all
from paper.utils import classification_metrics


def rpa_lite(Xs, ys, Xt, classes, n_iter=5):
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(Xs, ys)
    pseudo_yt = clf.predict(Xt)
    Xt_current = Xt.copy()

    for _ in range(n_iter):
        source_means = np.array([Xs[ys == c].mean(axis=0) for c in classes])
        target_means = np.array([
            Xt_current[pseudo_yt == c].mean(axis=0) if np.any(pseudo_yt == c) else np.zeros(Xs.shape[1])
            for c in classes
        ])
        R, _ = orthogonal_procrustes(target_means, source_means)
        Xt_current = Xt_current @ R

        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(Xs, ys)
        pseudo_yt = clf.predict(Xt_current)

    return pseudo_yt


def run(subjects=None, cov_estimator="lwf", n_iter=5):
    config.set_seed(config.SEED)
    subjects = subjects or config.EVAL_SUBJECTS
    per_subject, label_encoder = load_bnci_all(subjects, align="ea")
    classes = np.arange(len(label_encoder.classes_))
    cov = Covariances(estimator=cov_estimator)

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

        pred = rpa_lite(Xs, source_y, Xt, classes, n_iter=n_iter)

        metrics = classification_metrics(target_y, pred)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        print(f"[rpa] held-out subject {held_out}: acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

    accs = np.array([r["accuracy"] for r in fold_results])
    kappas = np.array([r["kappa"] for r in fold_results])
    summary = {"accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
               "kappa_mean": kappas.mean(), "kappa_std": kappas.std()}
    print(f"[rpa] LOSO accuracy: {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}  "
          f"kappa: {summary['kappa_mean']:.4f} +/- {summary['kappa_std']:.4f}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "zero_shot_rpa.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "accuracy", "kappa"])
        writer.writeheader()
        for r in fold_results:
            writer.writerow(r)

    return summary, fold_results


if __name__ == "__main__":
    run()
