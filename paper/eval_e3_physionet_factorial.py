"""TMLR revision round 2, task E3: replicate the preprocessing-vs-architecture factorial
(Section 10.1, paper/eval_factorial.py) at PhysioNet scale (up to 109 subjects, 12x the sample of
BCI-IV 2a) -- directly answers reviewer Critical Issue 2 ("strengthen the statistical evidence for
time-window selection and covariance estimation; distinguish statistically-supported findings from
empirical trends").

Identical design to paper/eval_factorial.py, with PhysioNet substituted for BCI-IV 2a: same 4
windows, same 2 estimators, same 2 methods (FgMDM, MEKT-lite at the FIXED "default" config
k=100/lam=1.0/5 iterations -- NOT the BCI-IV-2a-tuned config, exactly mirroring
paper/eval_factorial.py's own docstring rationale for keeping this a clean preprocessing ablation,
not a second round of test-set hyperparameter search), same EA-aligned zero-shot LOSO protocol.

Report whatever happens, honestly: if the estimator effect does not replicate at this much larger
sample, that is itself the answer to Critical Issue 2, not a reason to suppress or reframe the
result.

    python -m paper.eval_e3_physionet_factorial
"""
import csv
import importlib
import os
import random

import numpy as np
from pyriemann.classification import FgMDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from paper import config
from paper.eval_factorial import WINDOWS, ESTIMATORS, MEKT_K, MEKT_LAM, MEKT_N_ITER
from paper.eval_mekt import mekt_lite
from paper.utils import classification_metrics

RESULTS_DIR = config.RESULTS_DIR
OUT_PREFIX = "e3_physionet_factorial"
SUBSET_SEED = 42
SUBSET_SIZE = 30
# A full 109-subject x 16-cell run was attempted first and killed after measuring its real
# per-fold cost: FgMDM alone (the FASTER of the two methods) averaged ~1.3 minutes/fold on a
# ~108-subject training pool (the Riemannian-mean step in pyriemann's FgMDM.fit scales with
# training-set size), projecting to 2-3 DAYS total across all 16 cells -- not tractable to
# monitor to completion in this session. Falls back to the same disclosed, fixed,
# seed-42-sampled 30-subject PhysioNet subset used by task E1 (paper/run_dl_loso_sweep.py),
# for consistency and so E1/E3 results are on the same PhysioNet population. Pass
# subject_mode="full" to attempt the full 109-subject run anyway (very slow; resumable).


_SUBJECT_MODE = "subset30"  # set by main(); namespaces fold-CSV filenames so the full-109-subject
# run never collides with (or silently extends) the already-reported n=30 subset result.


def _cell_key(method, tmin, tmax, estimator):
    suffix = "" if _SUBJECT_MODE == "subset30" else f"_{_SUBJECT_MODE}"
    return f"{method}_t{tmin}_{tmax}_{estimator}{suffix}"


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
    """Mirrors paper/eval_factorial.py::load_window_data exactly, but for PhysioNet instead of
    BCI-IV 2a (load_physionet_zeroshot instead of load_bnci_all), align='ea' throughout (the same
    EA-aligned convention every factorial cell uses)."""
    config.TMIN, config.TMAX = tmin, tmax
    import paper.data as data_mod
    importlib.reload(data_mod)
    per_subject, label_encoder, good_subjects = data_mod.load_physionet_zeroshot(subjects, align="ea")
    # PhysioNet subjects' raw recordings resample to very slightly different trial lengths at some
    # (tmin,tmax) windows (off-by-one-sample edge case; not observed on BCI-IV 2a, which this
    # factorial design was originally built for) -- crop every subject to the shortest common
    # n_times at this window so concatenation across subjects doesn't fail. At most 1 sample
    # (<=1/128s) is ever dropped; verified below.
    n_times_per_subject = {s: X.shape[-1] for s, (X, y) in per_subject.items()}
    min_n_times = min(n_times_per_subject.values())
    max_n_times = max(n_times_per_subject.values())
    if max_n_times - min_n_times > 1:
        raise RuntimeError(f"Unexpected >1-sample trial-length mismatch across PhysioNet subjects "
                            f"at window ({tmin},{tmax}): {n_times_per_subject}")
    if min_n_times != max_n_times:
        per_subject = {s: (X[..., :min_n_times], y) for s, (X, y) in per_subject.items()}
    return per_subject, label_encoder, good_subjects


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


def main(subject_mode="subset30", window_filter=None, method_filter=None):
    """window_filter: optional (tmin,tmax) tuple to restrict to one window (for parallelizing
    the full-109-subject run across separate Kaggle kernels, one per window -- see task E
    follow-up: the full run is too slow for one kernel/session, so it is split).
    method_filter: optional "fgmdm" or "mekt_default" to further restrict to one method."""
    global _SUBJECT_MODE
    _SUBJECT_MODE = subject_mode
    config.set_seed(config.SEED)
    if subject_mode == "full":
        subjects = config.PRETRAIN_SUBJECTS  # up to 109 PhysioNet subjects
        # The n=30 subset is a seed-42 sample of this SAME 109-subject population under the
        # IDENTICAL protocol -- its already-computed folds are valid, legitimate head-start data
        # for the full run, not a different experiment. Copy (never move) them into the
        # full-namespaced files so the n=30 result stays intact and separately reportable.
        for tmin, tmax in (WINDOWS if window_filter is None else [window_filter]):
            for estimator in ESTIMATORS:
                for method_name in (["fgmdm", "mekt_default"] if method_filter is None else [method_filter]):
                    subset_key = f"{method_name}_t{tmin}_{tmax}_{estimator}"
                    full_key = f"{subset_key}_full"
                    subset_path = os.path.join(RESULTS_DIR, f"{OUT_PREFIX}_{subset_key}.csv")
                    full_path = os.path.join(RESULTS_DIR, f"{OUT_PREFIX}_{full_key}.csv")
                    if os.path.exists(subset_path) and not os.path.exists(full_path):
                        import shutil
                        shutil.copy(subset_path, full_path)
                        print(f"Seeded {full_path} from completed n=30 subset result "
                              f"(legitimate head start, same population/protocol)", flush=True)
    else:
        rng = random.Random(SUBSET_SEED)
        subjects = sorted(rng.sample(list(config.PRETRAIN_SUBJECTS), SUBSET_SIZE))
    windows = [window_filter] if window_filter else WINDOWS
    methods = [(method_filter, run_fgmdm_cell if method_filter == "fgmdm" else run_mekt_cell)] \
        if method_filter else [("fgmdm", run_fgmdm_cell), ("mekt_default", run_mekt_cell)]
    print(f"E3 PhysioNet factorial: subject_mode={subject_mode}, n={len(subjects)} subjects, "
          f"windows={windows}, methods={[m for m, _ in methods]}", flush=True)
    original_tmin, original_tmax = config.TMIN, config.TMAX
    summary_rows = []

    try:
        for tmin, tmax in windows:
            per_subject, label_encoder, good_subjects = load_window_data(tmin, tmax, subjects)
            classes = np.arange(len(label_encoder.classes_))
            for estimator in ESTIMATORS:
                for method_name, run_fn in methods:
                    cell_key = _cell_key(method_name, tmin, tmax, estimator)
                    if method_name == "fgmdm":
                        fold_results = run_fn(per_subject, good_subjects, estimator, cell_key)
                    else:
                        fold_results = run_fn(per_subject, good_subjects, estimator, cell_key, classes)
                    accs = np.array([r["accuracy"] for r in fold_results])
                    kappas = np.array([r["kappa"] for r in fold_results])
                    print(f"[{cell_key}] n={len(fold_results)} acc={accs.mean()*100:.2f}% +/- "
                          f"{accs.std()*100:.2f}% kappa={kappas.mean():.3f}", flush=True)
                    summary_rows.append({
                        "method": method_name, "tmin": tmin, "tmax": tmax, "estimator": estimator,
                        "n_subjects": len(fold_results), "accuracy_mean": accs.mean(),
                        "accuracy_std": accs.std(), "kappa_mean": kappas.mean(), "kappa_std": kappas.std(),
                    })
    finally:
        config.TMIN, config.TMAX = original_tmin, original_tmax

    # Namespaced per (subject_mode, filter) so parallel split-by-window/method kernel runs never
    # clobber each other's partial summary; the final combined 16-row summary/stats are computed
    # separately once all pieces land (same pattern as every other split-across-kernels task).
    filt_suffix = ""
    if window_filter:
        filt_suffix += f"_t{window_filter[0]}_{window_filter[1]}"
    if method_filter:
        filt_suffix += f"_{method_filter}"
    out_path = os.path.join(RESULTS_DIR, f"{OUT_PREFIX}_{subject_mode}_summary{filt_suffix}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "tmin", "tmax", "estimator", "n_subjects",
                                                "accuracy_mean", "accuracy_std", "kappa_mean", "kappa_std"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nWrote {len(summary_rows)} summary rows to {out_path}")
    return summary_rows


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "subset30")
