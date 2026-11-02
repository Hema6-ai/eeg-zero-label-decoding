"""TMLR revision round 2, task E4: truly nested model selection, replacing the existing nested
check (paper/eval_mekt_ensemble.py's nested-LOSO validation, Section~\\ref{sec:nestedloso}) which
chose among only 9 pre-built candidates (the 8 individual (window,estimator) configs already
selected using all 9 BCI-IV 2a test subjects, plus the one specific 8-member ensemble already
built from those same windows/estimators) -- a real leakage concern the task calls out explicitly,
since the CANDIDATE SET ITSELF was defined by looking at all 9 test subjects, even though the
final choice among those 9 candidates was then nested correctly.

This script fixes that: per OUTER fold, the full hyperparameter sweep itself (not just the choice
among a few pre-selected finalists) is re-run on an INNER loop over the remaining (source) subjects
ONLY -- the outer held-out subject is never touched until the single final evaluation.

  - Window sweep: tmin in {0, 0.3, 0.5, 1.0} x tmax in {2.0, 2.5, 3.0, 3.5, 4.0} (20 combinations;
    every tmin < every tmax here, so all 20 are "valid").
  - Estimator sweep: {lwf, corr, oas}.
  - Method: {FgMDM, MEKT default (k=100, lam=1.0, 5 iterations -- the SAME untuned "default"
    config used throughout paper/eval_factorial.py and task E3, not the BCI-IV-2a-tuned config)}.
  - 20 x 3 x 2 = 120 inner configs per outer fold.

Per outer fold: for each of the |source subjects| inner folds, evaluate all 120 configs (train on
the other source subjects, validate on the held-out inner subject); average each config's inner
accuracy across inner folds; pick (a) the single best config by inner accuracy, and (b) the
top-8-by-inner-accuracy configs for a soft-vote ensemble (summed/averaged predicted-class
probabilities). Both are then applied EXACTLY ONCE to the true outer held-out subject (trained on
ALL source subjects at the winning config(s)). This nested accuracy may legitimately be below the
52.39% headline -- report whatever happens.

Datasets: BCI-IV 2a (9 subjects, full LOSO) and a fixed, seed-42-sampled 30-subject PhysioNet
subset (same subset and same disclosed rationale as paper/run_dl_loso_sweep.py's task E1 --
full-109-subject nested selection would multiply an already-heavy inner sweep by >3x).

The OLD candidate-set check (paper/eval_mekt_ensemble.py's nested-LOSO) is kept, unmodified and
separately labeled, with its own limitation (candidate set itself was pre-selected using all 9
test subjects) restated rather than silently superseded.

    python -m paper.eval_e4_nested_selection
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
FULL_WINDOWS_E4 = [(tmin, tmax) for tmin in (0.0, 0.3, 0.5, 1.0) for tmax in (2.0, 2.5, 3.0, 3.5, 4.0)]
FULL_ESTIMATORS_E4 = ["lwf", "corr", "oas"]
# Compute-cost finding (measured directly, not estimated): a first attempt at the full 20-window
# x 3-estimator x 2-method (120-config) inner sweep used 2.6 CPU-hours without finishing even the
# FIRST of 9 outer folds (960 inner fits/fold; MEKT's JDA iterations dominate the cost at this
# scale). Projected total: ~20+ hours, not tractable to run to completion in this session. Per the
# same disclosed-fallback principle used for tasks E1 (fixed 30-subject PhysioNet subset) and E3
# (same subset, after an analogous measured-cost finding), this reduces the inner grid to 8 windows
# (all 4 tmin values x a representative tmax subset {2.5, 3.5} spanning short/long, instead of all
# 5 tmax values) x 2 estimators (lwf, corr -- dropping oas, which this paper uses nowhere else as a
# primary lever) x 2 methods = 32 configs, a 3.75x reduction, projecting to ~5-6 hours. Pass
# full_grid=True to run the complete, literal 120-config sweep instead (very slow; resumable).
REDUCED_WINDOWS_E4 = [(tmin, tmax) for tmin in (0.0, 0.3, 0.5, 1.0) for tmax in (2.5, 3.5)]
REDUCED_ESTIMATORS_E4 = ["lwf", "corr"]
WINDOWS_E4 = REDUCED_WINDOWS_E4
ESTIMATORS_E4 = REDUCED_ESTIMATORS_E4
METHODS_E4 = ["fgmdm", "mekt_default"]
MEKT_K, MEKT_LAM, MEKT_N_ITER = 100, 1.0, 5  # "default" -- matches eval_factorial.py / task E3
TOP_K_ENSEMBLE = 8


def precompute_covariances(subjects):
    """Returns ({(tmin,tmax,estimator): {subject: covs}}, {subject: labels}, label_encoder).
    Covariances depend only on the (window, estimator) choice, not on which subjects end up in
    the inner-train/inner-val/outer split for a given fold -- so each is computed exactly once
    and reused across every fold and config that needs it (mirrors
    paper/eval_mekt_ensemble.py::_precompute_covariances' precomputation pattern)."""
    per_subject_covs = {}
    per_subject_labels = {}
    label_encoder = None
    for tmin, tmax in WINDOWS_E4:
        config.TMIN, config.TMAX = tmin, tmax
        import paper.data as data_mod
        importlib.reload(data_mod)
        per_subject, label_encoder = data_mod.load_bnci_all(subjects, align="ea") if _IS_BCI \
            else _load_physionet(data_mod, subjects)
        for subj in subjects:
            if subj in per_subject:
                per_subject_labels[subj] = per_subject[subj][1]
        for est in ESTIMATORS_E4:
            cov = Covariances(estimator=est)
            per_subject_covs[(tmin, tmax, est)] = {
                subj: cov.fit_transform(per_subject[subj][0].astype(np.float64))
                for subj in subjects if subj in per_subject
            }
    return per_subject_covs, per_subject_labels, label_encoder


_IS_BCI = True  # toggled by main() per dataset


def _load_physionet(data_mod, subjects):
    per_subject, label_encoder, good_subjects = data_mod.load_physionet_zeroshot(subjects, align="ea")
    # Same off-by-one-sample edge case as paper/eval_e3_physionet_factorial.py -- crop every
    # subject to the shortest common n_times at this window before concatenation.
    n_times_per_subject = {s: X.shape[-1] for s, (X, y) in per_subject.items()}
    min_n_times, max_n_times = min(n_times_per_subject.values()), max(n_times_per_subject.values())
    if max_n_times - min_n_times > 1:
        raise RuntimeError(f"Unexpected >1-sample trial-length mismatch across PhysioNet subjects: "
                            f"{n_times_per_subject}")
    if min_n_times != max_n_times:
        per_subject = {s: (X[..., :min_n_times], y) for s, (X, y) in per_subject.items()}
    return per_subject, label_encoder


def fit_predict_proba(method, train_covs, train_y, val_covs, classes):
    if method == "fgmdm":
        clf = FgMDM(metric="riemann", tsupdate=False)
        clf.fit(train_covs, train_y)
        proba_raw = clf.predict_proba(val_covs)
        proba = np.zeros((len(val_covs), len(classes)))
        for i, c in enumerate(clf.classes_):
            proba[:, c] = proba_raw[:, i]
        return proba
    ts = TangentSpace(metric="riemann")
    Zs = ts.fit_transform(train_covs)
    Zt = ts.transform(val_covs)
    k_eff = min(MEKT_K, Zs.shape[1])
    _, proba = mekt_lite(Zs, train_y, Zt, classes, k=k_eff, lam=MEKT_LAM, n_iter=MEKT_N_ITER,
                          return_proba=True)
    return proba


def run_outer_fold(held_out, available_subjects, covs, labels, classes, max_inner_folds=None):
    """max_inner_folds: if set, subsamples (seed=42, deterministic) which source subjects serve
    as the inner-validation fold, without reducing the inner-TRAIN pool itself -- controls cost
    on large subject pools (e.g. PhysioNet) where an all-29-inner-folds sweep is prohibitively
    slow (see docstring compute-cost note)."""
    source_subjects = [s for s in available_subjects if s != held_out]
    configs = [(w, e, m) for w in WINDOWS_E4 for e in ESTIMATORS_E4 for m in METHODS_E4]

    inner_val_subjects = source_subjects
    if max_inner_folds is not None and len(source_subjects) > max_inner_folds:
        import random
        rng = random.Random(42)
        inner_val_subjects = sorted(rng.sample(source_subjects, max_inner_folds))

    # --- Inner loop: full sweep, source subjects only ---
    inner_scores = {cfg: [] for cfg in configs}
    for inner_val in inner_val_subjects:
        inner_train = [s for s in source_subjects if s != inner_val]
        for (tmin, tmax), est, method in configs:
            c = covs[(tmin, tmax, est)]
            train_covs = np.concatenate([c[s] for s in inner_train], axis=0)
            train_y = np.concatenate([labels[s] for s in inner_train], axis=0)
            val_covs = c[inner_val]
            val_y = labels[inner_val]
            try:
                proba = fit_predict_proba(method, train_covs, train_y, val_covs, classes)
            except Exception as e:
                # Isolates which exact (window, estimator, method, inner_val subject) combo
                # failed, rather than crashing the whole outer fold on one numerical edge case
                # (e.g. a degenerate/near-singular covariance for a specific subject at a
                # specific window+estimator -- more likely with larger, previously-untested
                # training pools like PhysioNet's). That config's inner score for this inner fold
                # is simply skipped; its mean is still over whichever inner folds succeeded.
                import traceback
                print(f"[E4] inner fit FAILED: window=({tmin},{tmax}) est={est} method={method} "
                      f"inner_val={inner_val} train_n={len(inner_train)} -- {e}", flush=True)
                traceback.print_exc()
                continue
            pred = proba.argmax(axis=1)
            acc = (pred == val_y).mean()
            inner_scores[(tmin, tmax), est, method].append(acc)

    inner_scores = {cfg: accs for cfg, accs in inner_scores.items() if len(accs) > 0}
    if not inner_scores:
        raise RuntimeError("Every inner config failed for this outer fold -- see printed "
                            "tracebacks above for the underlying errors.")
    mean_inner = {cfg: float(np.mean(accs)) for cfg, accs in inner_scores.items()}
    ranked = sorted(mean_inner.items(), key=lambda kv: kv[1], reverse=True)
    best_cfg = ranked[0][0]
    top_k_cfgs = [cfg for cfg, _ in ranked[:TOP_K_ENSEMBLE]]

    # --- Outer application: fit on ALL source subjects at the winning config(s), predict once ---
    def fit_on_all_source(cfg):
        (tmin, tmax), est, method = cfg
        c = covs[(tmin, tmax, est)]
        train_covs = np.concatenate([c[s] for s in source_subjects], axis=0)
        train_y = np.concatenate([labels[s] for s in source_subjects], axis=0)
        val_covs = c[held_out]
        return fit_predict_proba(method, train_covs, train_y, val_covs, classes)

    target_y = labels[held_out]

    best_proba = fit_on_all_source(best_cfg)
    best_pred = best_proba.argmax(axis=1)
    best_metrics = classification_metrics(target_y, best_pred)

    ensemble_proba = np.mean([fit_on_all_source(cfg) for cfg in top_k_cfgs], axis=0)
    ensemble_pred = ensemble_proba.argmax(axis=1)
    ensemble_metrics = classification_metrics(target_y, ensemble_pred)

    return {
        "held_out_subject": held_out,
        "best_config": f"{best_cfg[0]}_{best_cfg[1]}_{best_cfg[2]}",
        "best_inner_acc": ranked[0][1],
        "best_nested_accuracy": best_metrics["accuracy"],
        "best_nested_kappa": best_metrics["kappa"],
        "top8_configs": ";".join(f"{c[0]}_{c[1]}_{c[2]}" for c in top_k_cfgs),
        "ensemble_nested_accuracy": ensemble_metrics["accuracy"],
        "ensemble_nested_kappa": ensemble_metrics["kappa"],
    }


def main(dataset="bci2a", full_grid=False, max_inner_folds=None):
    global _IS_BCI, WINDOWS_E4, ESTIMATORS_E4
    _IS_BCI = (dataset == "bci2a")
    WINDOWS_E4 = FULL_WINDOWS_E4 if full_grid else REDUCED_WINDOWS_E4
    ESTIMATORS_E4 = FULL_ESTIMATORS_E4 if full_grid else REDUCED_ESTIMATORS_E4
    # PhysioNet's source pool (~29 subjects/outer fold, vs ~7-8 for BCI-IV 2a) makes both the
    # per-fit cost (larger training sets) AND the inner-fold count much higher than BCI-IV 2a's
    # already-measured-and-reduced grid. Default to 10 inner-validation folds (deterministic,
    # seed 42 subsample) instead of all ~29 for any dataset other than bci2a, unless the caller
    # overrides it -- same disclosed-fallback principle as the grid reduction above.
    if max_inner_folds is None and dataset != "bci2a":
        max_inner_folds = 10
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if dataset == "bci2a":
        subjects = config.EVAL_SUBJECTS
    else:
        import random
        rng = random.Random(42)
        subjects = sorted(rng.sample(list(config.PRETRAIN_SUBJECTS), 30))
    print(f"E4 nested selection: dataset={dataset}, subjects={subjects}", flush=True)

    grid_suffix = "fullgrid" if full_grid else "reducedgrid"
    out_path = os.path.join(RESULTS_DIR, f"e4_nested_selection_{dataset}_{grid_suffix}.csv")
    done = {}
    if os.path.exists(out_path):
        with open(out_path, newline="") as f:
            for row in csv.DictReader(f):
                done[int(row["held_out_subject"])] = row

    original_tmin, original_tmax = config.TMIN, config.TMAX
    try:
        covs, labels, label_encoder = precompute_covariances(subjects)
        available_subjects = sorted(labels.keys())
        classes = np.arange(len(label_encoder.classes_))

        fieldnames = ["held_out_subject", "best_config", "best_inner_acc", "best_nested_accuracy",
                      "best_nested_kappa", "top8_configs", "ensemble_nested_accuracy",
                      "ensemble_nested_kappa"]
        write_header = not os.path.exists(out_path)
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for held_out in available_subjects:
                if held_out in done:
                    continue
                try:
                    row = run_outer_fold(held_out, available_subjects, covs, labels, classes,
                                          max_inner_folds=max_inner_folds)
                except Exception as e:
                    # Fault-tolerant: one problematic outer fold (e.g. a numerical edge case in
                    # MEKT/FgMDM for a specific subject/config on a dataset this script was not
                    # originally built for) should not lose every other fold's results on a
                    # long-running Kaggle kernel with no interactive debugging. Print full
                    # diagnostics and continue; the skipped fold is simply absent from the output
                    # CSV (visible as a gap, not silently imputed).
                    import traceback
                    print(f"[E4] held_out={held_out} FAILED -- {e}", flush=True)
                    traceback.print_exc()
                    continue
                writer.writerow(row)
                f.flush()
                print(f"[{dataset}] held_out={held_out} best={row['best_config']} "
                      f"(inner_acc={row['best_inner_acc']*100:.2f}%) "
                      f"nested_acc={row['best_nested_accuracy']*100:.2f}% "
                      f"ensemble_nested_acc={row['ensemble_nested_accuracy']*100:.2f}%", flush=True)
    finally:
        config.TMIN, config.TMAX = original_tmin, original_tmax

    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))
    best_accs = np.array([float(r["best_nested_accuracy"]) for r in rows])
    ens_accs = np.array([float(r["ensemble_nested_accuracy"]) for r in rows])
    print(f"\n=== [{dataset}] E4 truly-nested result: best-single-config "
          f"{best_accs.mean()*100:.2f}% +/- {best_accs.std()*100:.2f}%, "
          f"top-{TOP_K_ENSEMBLE}-ensemble {ens_accs.mean()*100:.2f}% +/- {ens_accs.std()*100:.2f}% "
          f"(n={len(rows)}) ===", flush=True)
    return rows


if __name__ == "__main__":
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "bci2a"
    main(dataset)
