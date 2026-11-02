"""TMLR revision round 2, Phase-5 follow-up item 1: the manuscript's alignment claim ("EA
helps") is tested directly only for FgMDM/MEKT on BCI-IV 2a and for EEGNetv4 on BCI-IV
2a/PhysioNet30 (paper/stats_eegnet_comprehensive.py). It was never tested with EA OFF for
FgMDM/MEKT on the full PhysioNet population or on Cho2017 -- this script closes that gap so the
alignment claim is backed everywhere it is stated, not asserted by extension from a different
dataset/method combination.

Runs FgMDM and MEKT-lite (default hyperparameters, matching every other EA-off/on comparison in
this paper) with EA OFF on:
  (a) PhysioNet, all available subjects (same population as physionet_comprehensive_*.csv's
      EA-on rows, so the comparison is apples-to-apples on identical subjects).
  (b) Cho2017, all 52 subjects (same population as e5_cho2017_*.csv's EA-on rows).

    python -m paper.eval_e9_alignment_ablation
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
from paper.eval_e5_third_dataset import load_cho2017
from paper.utils import classification_metrics

RESULTS_DIR = config.RESULTS_DIR
MEKT_DEFAULT_K, MEKT_DEFAULT_LAM, MEKT_DEFAULT_N_ITER = 100, 1.0, 5


def _fold_csv_path(name):
    return os.path.join(RESULTS_DIR, f"{name}.csv")


def _load_done_folds(name):
    path = _fold_csv_path(name)
    done = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                done[int(row["held_out_subject"])] = {
                    "accuracy": float(row["accuracy"]), "kappa": float(row["kappa"])}
    return done


def _append_fold_csv(name, held_out, accuracy, kappa):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = _fold_csv_path(name)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "accuracy", "kappa"])
        if write_header:
            writer.writeheader()
        writer.writerow({"held_out_subject": held_out, "accuracy": accuracy, "kappa": kappa})


def run_fgmdm_noea(per_subject, subjects, name, estimator="lwf"):
    done = _load_done_folds(name)
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
        _append_fold_csv(name, held_out, metrics["accuracy"], metrics["kappa"])
        done[held_out] = metrics
        print(f"[{name}] held_out={held_out} acc={metrics['accuracy']*100:.2f}%", flush=True)
    return done


def run_mekt_noea(per_subject, subjects, classes, name, estimator="lwf"):
    done = _load_done_folds(name)
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
        k_eff = min(MEKT_DEFAULT_K, Xs.shape[1])
        pred = mekt_lite(Xs, source_y, Xt, classes, k=k_eff, lam=MEKT_DEFAULT_LAM,
                          n_iter=MEKT_DEFAULT_N_ITER)
        metrics = classification_metrics(target_y, pred)
        _append_fold_csv(name, held_out, metrics["accuracy"], metrics["kappa"])
        done[held_out] = metrics
        print(f"[{name}] held_out={held_out} acc={metrics['accuracy']*100:.2f}%", flush=True)
    return done


def run_physionet(subjects=None):
    subjects = subjects or config.PRETRAIN_SUBJECTS
    import paper.data as data_mod
    importlib.reload(data_mod)
    per_subject, label_encoder, good_subjects = data_mod.load_physionet_zeroshot(subjects, align="none")
    classes = np.arange(len(label_encoder.classes_))
    print(f"PhysioNet (no EA): {len(good_subjects)}/{len(subjects)} subjects loaded", flush=True)

    fgmdm_done = run_fgmdm_noea(per_subject, good_subjects, "physionet_comprehensive_noea_fgmdm")
    mekt_done = run_mekt_noea(per_subject, good_subjects, classes, "physionet_comprehensive_noea_mekt_default")
    return fgmdm_done, mekt_done


def run_cho2017(subjects=None):
    subjects = subjects or list(range(1, 53))
    per_subject, label_encoder, good_subjects = load_cho2017(subjects)
    classes = np.arange(len(label_encoder.classes_))
    print(f"Cho2017 (no EA): {len(good_subjects)}/{len(subjects)} subjects loaded", flush=True)

    fgmdm_done = run_fgmdm_noea(per_subject, good_subjects, "e5_cho2017_noea_fgmdm")
    mekt_done = run_mekt_noea(per_subject, good_subjects, classes, "e5_cho2017_noea_mekt_default")
    return fgmdm_done, mekt_done


def main():
    config.set_seed(config.SEED)
    print("=== PhysioNet, EA OFF ===", flush=True)
    pn_fgmdm, pn_mekt = run_physionet()
    print("\n=== Cho2017, EA OFF ===", flush=True)
    cho_fgmdm, cho_mekt = run_cho2017()

    for label, done in [("PhysioNet FgMDM noEA", pn_fgmdm), ("PhysioNet MEKT noEA", pn_mekt),
                         ("Cho2017 FgMDM noEA", cho_fgmdm), ("Cho2017 MEKT noEA", cho_mekt)]:
        accs = np.array([v["accuracy"] for v in done.values()])
        print(f"{label}: n={len(accs)} mean={accs.mean()*100:.2f}% std={accs.std()*100:.2f}%")


if __name__ == "__main__":
    main()
