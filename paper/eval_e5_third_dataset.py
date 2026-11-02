"""TMLR revision round 2, task E5 (should-do): a third MOABB motor-imagery dataset beyond
BCI-IV 2a and PhysioNet, to check whether this paper's central comparison generalizes to a THIRD
independent subject population. Uses Cho2017 (52 subjects, 2-class left/right-hand MI, verified
this session to load cleanly under this project's exact canonical pipeline -- 22-channel montage,
128 Hz, default [0,3]s window -- giving (n_trials, 22, 384) with labels {left_hand, right_hand}).

**Chance level here is 50% (2-class), not the 25%/33% of BCI-IV 2a/PhysioNet** -- every accuracy
number from this script must be read against that, not compared as a raw percentage to the other
two datasets' headline numbers.

Runs the same three methods compared everywhere else in this paper, LOSO, zero-shot, EA-aligned:
  - FgMDM (single window, default [0,3]s)
  - MEKT-lite, "default" hyperparameters (k=100, lam=1.0, 5 iterations -- same as paper/eval_factorial.py
    and task E3, NOT the BCI-IV-2a-tuned config)
  - The fixed 8-member ensemble (4 windows x 2 covariance estimators, k=50/lam=1.0/10 iterations --
    the EXACT verified configuration from paper/eval_mekt_ensemble.py, not re-tuned on this dataset)

The "best E1 CNN" row the task also asks for is added separately once paper/run_dl_loso_sweep.py's
results exist (not yet, as of this script's creation) -- this script does not block on that.

Cho2017's per-subject raw files are large (~200MB each); downloading all subjects the first time
this runs is itself a slow, resumable step via MOABB/MNE's own on-disk cache.

    python -m paper.eval_e5_third_dataset
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
from paper.eval_mekt_ensemble import WINDOWS as ENSEMBLE_WINDOWS, ESTIMATORS as ENSEMBLE_ESTIMATORS
from paper.eval_mekt_ensemble import MEKT_K as ENSEMBLE_MEKT_K, MEKT_LAM as ENSEMBLE_MEKT_LAM
from paper.eval_mekt_ensemble import MEKT_N_ITER as ENSEMBLE_MEKT_N_ITER
from paper.utils import classification_metrics

RESULTS_DIR = config.RESULTS_DIR
OUT_PREFIX = "e5_cho2017"
MEKT_DEFAULT_K, MEKT_DEFAULT_LAM, MEKT_DEFAULT_N_ITER = 100, 1.0, 5  # matches E3/eval_factorial


def load_cho2017(subjects):
    """Mirrors paper/data.py::load_physionet_zeroshot's structure (per-subject try/except so one
    bad subject doesn't abort the batch), but for Cho2017 instead of PhysionetMI."""
    from moabb.datasets import Cho2017
    from moabb.paradigms import MotorImagery
    from sklearn.preprocessing import LabelEncoder

    paradigm = MotorImagery(channels=config.COMMON_CHANNELS, resample=config.RESAMPLE_HZ,
                             tmin=config.TMIN, tmax=config.TMAX)
    dataset = Cho2017()
    label_encoder = LabelEncoder()
    label_encoder.fit(["left_hand", "right_hand"])

    per_subject = {}
    skipped = []
    for subject in subjects:
        try:
            X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
        except Exception as e:
            skipped.append((subject, str(e)))
            continue
        if len(X) == 0:
            skipped.append((subject, "no trials"))
            continue
        mean = X.mean(axis=-1, keepdims=True)
        std = X.std(axis=-1, keepdims=True)
        X = ((X - mean) / (std + 1e-6)).astype(np.float32)
        y_encoded = label_encoder.transform(y)
        per_subject[subject] = (X, y_encoded)

    if skipped:
        print(f"load_cho2017: skipped {len(skipped)} subjects: {skipped}", flush=True)
    return per_subject, label_encoder, sorted(per_subject.keys())


def _ea_align_per_subject(per_subject):
    from paper.align import align_per_subject
    X_only = {s: X for s, (X, y) in per_subject.items()}
    aligned = align_per_subject(X_only, "ea")
    return {s: (aligned[s].astype(np.float32), y) for s, (_, y) in per_subject.items()}


def _fold_csv_path(name):
    return os.path.join(RESULTS_DIR, f"{OUT_PREFIX}_{name}.csv")


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


def run_fgmdm(per_subject, subjects, classes, estimator="lwf"):
    name = "fgmdm"
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


def run_mekt_default(per_subject, subjects, classes, estimator="lwf"):
    name = "mekt_default"
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


def run_ensemble(subjects, classes):
    """The fixed, verified 8-member ensemble config from paper/eval_mekt_ensemble.py, applied to
    Cho2017 -- requires its own per-window covariance precomputation since the windows differ
    from the single default window used above."""
    name = "ensemble8"
    done = _load_done_folds(name)
    if len(done) >= len(subjects):
        return done

    per_subject_covs = {}
    per_subject_labels = {}
    original_tmin, original_tmax = config.TMIN, config.TMAX
    try:
        for tmin, tmax in ENSEMBLE_WINDOWS:
            config.TMIN, config.TMAX = tmin, tmax
            per_subject, _, _ = load_cho2017(subjects)
            per_subject = _ea_align_per_subject(per_subject)
            for s, (X, y) in per_subject.items():
                per_subject_labels[s] = y
            for est in ENSEMBLE_ESTIMATORS:
                cov = Covariances(estimator=est)
                per_subject_covs[(tmin, tmax, est)] = {
                    s: cov.fit_transform(X.astype(np.float64)) for s, (X, y) in per_subject.items()
                }
    finally:
        config.TMIN, config.TMAX = original_tmin, original_tmax

    available = sorted(per_subject_labels.keys())
    candidate_keys = [(tmin, tmax, est) for tmin, tmax in ENSEMBLE_WINDOWS for est in ENSEMBLE_ESTIMATORS]

    for held_out in available:
        if held_out in done:
            continue
        source_subjects = [s for s in available if s != held_out]
        member_probs = []
        for key in candidate_keys:
            covs = per_subject_covs[key]
            source_covs = np.concatenate([covs[s] for s in source_subjects], axis=0)
            source_y = np.concatenate([per_subject_labels[s] for s in source_subjects], axis=0)
            target_covs = covs[held_out]
            ts = TangentSpace(metric="riemann")
            Xs = ts.fit_transform(source_covs)
            Xt = ts.transform(target_covs)
            k_eff = min(ENSEMBLE_MEKT_K, Xs.shape[1])
            _, proba = mekt_lite(Xs, source_y, Xt, classes, k=k_eff, lam=ENSEMBLE_MEKT_LAM,
                                  n_iter=ENSEMBLE_MEKT_N_ITER, return_proba=True)
            member_probs.append(proba)
        ensemble_proba = sum(member_probs) / len(member_probs)
        pred = ensemble_proba.argmax(axis=1)
        metrics = classification_metrics(per_subject_labels[held_out], pred)
        _append_fold_csv(name, held_out, metrics["accuracy"], metrics["kappa"])
        done[held_out] = metrics
        print(f"[{name}] held_out={held_out} acc={metrics['accuracy']*100:.2f}%", flush=True)
    return done


def main(subjects=None):
    config.set_seed(config.SEED)
    subjects = subjects or list(range(1, 53))  # Cho2017: 52 subjects
    print(f"E5 (Cho2017, chance=50%): n={len(subjects)} candidate subjects", flush=True)

    per_subject, label_encoder, good_subjects = load_cho2017(subjects)
    per_subject = _ea_align_per_subject(per_subject)
    classes = np.arange(len(label_encoder.classes_))
    print(f"E5: {len(good_subjects)}/{len(subjects)} subjects loaded successfully", flush=True)

    fgmdm_done = run_fgmdm(per_subject, good_subjects, classes)
    mekt_done = run_mekt_default(per_subject, good_subjects, classes)
    ensemble_done = run_ensemble(good_subjects, classes)

    summary_rows = []
    for name, done in [("FgMDM", fgmdm_done), ("MEKT default", mekt_done),
                        ("8-member ensemble", ensemble_done)]:
        accs = np.array([v["accuracy"] for v in done.values()])
        kappas = np.array([v["kappa"] for v in done.values()])
        summary_rows.append({"method": name, "n_subjects": len(accs), "accuracy_mean": accs.mean(),
                              "accuracy_std": accs.std(), "kappa_mean": kappas.mean(),
                              "kappa_std": kappas.std()})
        print(f"=== [Cho2017, chance=50%] {name}: {accs.mean()*100:.2f}% +/- {accs.std()*100:.2f}% "
              f"(n={len(accs)}) ===", flush=True)

    out_path = os.path.join(RESULTS_DIR, f"{OUT_PREFIX}_summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nWrote {len(summary_rows)} summary rows to {out_path}")
    return summary_rows


if __name__ == "__main__":
    main()
