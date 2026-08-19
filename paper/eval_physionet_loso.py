"""Zero-shot LOSO evaluation WITHIN PhysionetMI itself (109 subjects, vs BCI-IV 2a's 9), to test
whether a larger subject pool averages out BCI illiteracy and yields a higher, more stable mean
accuracy. Same protocol as the BCI-IV 2a experiments: EA per subject (unsupervised), FgMDM and
MEKT-lite (k=50, lam=1.0, 20 iterations -- the tuned best from the BCI-IV 2a grid search).

Uses a 30-subject random sample (seed=42) for a fast run; stops early and reports if either method
crosses 60%, per the requested protocol.

    python -m paper.eval_physionet_loso
"""
import csv
import os

import numpy as np
from pyriemann.classification import FgMDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from paper import config
from paper.data import load_physionet_zeroshot
from paper.eval_mekt import mekt_lite
from paper.utils import classification_metrics

MEKT_BEST = dict(k=50, lam=1.0, n_iter=20)


def run_fgmdm(per_subject, subjects):
    fold_results = []
    for held_out in subjects:
        source_subjects = [s for s in subjects if s != held_out]
        source_X = np.concatenate([per_subject[s][0] for s in source_subjects], axis=0).astype(np.float64)
        source_y = np.concatenate([per_subject[s][1] for s in source_subjects], axis=0)
        target_X, target_y = per_subject[held_out]
        target_X = target_X.astype(np.float64)

        cov = Covariances(estimator="lwf")
        source_covs = cov.fit_transform(source_X)
        target_covs = cov.transform(target_X)
        clf = FgMDM(metric="riemann", tsupdate=False)
        clf.fit(source_covs, source_y)
        pred = clf.predict(target_covs)

        m = classification_metrics(target_y, pred)
        m["held_out_subject"] = held_out
        fold_results.append(m)
        print(f"[physionet-fgmdm] subject {held_out}: acc={m['accuracy']:.4f} kappa={m['kappa']:.4f}")
    return fold_results


def run_mekt(per_subject, subjects, classes):
    fold_results = []
    for held_out in subjects:
        source_subjects = [s for s in subjects if s != held_out]
        source_X = np.concatenate([per_subject[s][0] for s in source_subjects], axis=0).astype(np.float64)
        source_y = np.concatenate([per_subject[s][1] for s in source_subjects], axis=0)
        target_X, target_y = per_subject[held_out]
        target_X = target_X.astype(np.float64)

        cov = Covariances(estimator="lwf")
        source_covs = cov.fit_transform(source_X)
        target_covs = cov.transform(target_X)
        ts = TangentSpace(metric="riemann")
        Xs = ts.fit_transform(source_covs)
        Xt = ts.transform(target_covs)
        k_eff = min(MEKT_BEST["k"], Xs.shape[1])
        pred = mekt_lite(Xs, source_y, Xt, classes, k=k_eff, lam=MEKT_BEST["lam"], n_iter=MEKT_BEST["n_iter"])

        m = classification_metrics(target_y, pred)
        m["held_out_subject"] = held_out
        fold_results.append(m)
        print(f"[physionet-mekt] subject {held_out}: acc={m['accuracy']:.4f} kappa={m['kappa']:.4f}")
    return fold_results


def summarize(fold_results, name):
    accs = np.array([r["accuracy"] for r in fold_results])
    kappas = np.array([r["kappa"] for r in fold_results])
    n_illiterate = int((accs < 0.30).sum())
    summary = {"accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
               "kappa_mean": kappas.mean(), "kappa_std": kappas.std(), "n_illiterate": n_illiterate}
    print(f"[{name}] LOSO accuracy: {summary['accuracy_mean']*100:.2f}% +/- {summary['accuracy_std']*100:.2f}%  "
          f"kappa: {summary['kappa_mean']:.4f} +/- {summary['kappa_std']:.4f}  "
          f"(<30% subjects: {n_illiterate}/{len(fold_results)})")
    return summary


def run(n_subjects=30, seed=42):
    config.set_seed(config.SEED)
    rng = np.random.RandomState(seed)
    candidate_subjects = rng.choice(np.arange(1, 110), n_subjects, replace=False).tolist()

    print(f"Loading {n_subjects} random PhysionetMI subjects (seed={seed})...")
    per_subject, label_encoder, good_subjects = load_physionet_zeroshot(candidate_subjects, align="ea")
    classes = np.arange(len(label_encoder.classes_))
    print(f"Loaded {len(good_subjects)}/{n_subjects} subjects successfully.")
    for s in good_subjects:
        print(f"  subject {s}: {len(per_subject[s][0])} trials")

    print("\n=== Method 1: FgMDM + EA ===")
    fgmdm_folds = run_fgmdm(per_subject, good_subjects)
    fgmdm_summary = summarize(fgmdm_folds, "physionet-fgmdm")

    results = {"fgmdm": (fgmdm_summary, fgmdm_folds)}

    if fgmdm_summary["accuracy_mean"] > 0.60:
        print("\nFgMDM alone crossed 60% -- stopping per protocol (stop if one method hits 60%).")
        return results, good_subjects

    print("\n=== Method 2: MEKT-lite (k=50, lam=1.0, 20 iterations) ===")
    mekt_folds = run_mekt(per_subject, good_subjects, classes)
    mekt_summary = summarize(mekt_folds, "physionet-mekt")
    results["mekt"] = (mekt_summary, mekt_folds)

    if mekt_summary["accuracy_mean"] > 0.60:
        print("\nMEKT alone crossed 60% -- stopping per protocol.")
        return results, good_subjects

    if fgmdm_summary["accuracy_mean"] > 0.55 and mekt_summary["accuracy_mean"] > 0.55:
        print("\nBoth methods > 55% -- a 3-way ensemble would be the next step, BUT the SPDNet/RT-v2 "
              "checkpoints were pretrained on ALL 109 PhysionetMI subjects (including whichever of "
              "these 30 eval subjects overlap with that pretraining set), so reusing them here would "
              "leak pretraining exposure into the \"zero-shot\" eval. Skipping the 3-way ensemble for "
              "this dataset unless/until encoders are retrained excluding the eval subjects.")

    return results, good_subjects


if __name__ == "__main__":
    run()
