"""Comprehensive classical/Riemannian method-family comparison, zero-shot LOSO WITHIN PhysionetMI's
full subject population (up to 109 subjects), addressing TMLR review Critical Issues A/B: the
original method-family comparison was run almost entirely on BCI-IV 2a (9 subjects); only the
final winning pipeline (8-member MEKT ensemble) and two individual methods (FgMDM+EA, MEKT-lite
default) had previously been replicated on PhysioNet (see paper/results/final_results_table.csv
rows 55-56, 79). This script fills that gap for every CPU-feasible classical method family tried
on BCI-IV 2a, using the exact same method implementations (reused directly from eval_csp_lda_svm,
eval_riemannian_advanced, eval_rpa, eval_mekt) with PhysioNet substituted as the eval population.

Methods NOT included here: the raw-EEG self-supervised / geometric-deep-learning families (MAE*,
SPDNet, Riemannian Transformer, SupCon). Those encoders were pretrained on ALL 109 PhysioNet
subjects (see README / config.PRETRAIN_SUBJECTS); evaluating them zero-shot WITHIN PhysioNet would
leak each held-out subject's own unlabeled data into its own encoder's pretraining corpus. The
original project explicitly identified and avoided this (final_results_table.csv row 27 notes a
3-way ensemble step was "skipped per protocol...would have been invalid anyway: SPDNet/RT-v2 were
pretrained on all 109 PhysionetMI subjects, overlapping with this eval set"). See
reports/data_leakage_audit.md for the full discussion and the reversed-pretraining-direction fix
(pretrain on BCI-IV 2a instead) used for the deep-learning PhysioNet comparison on Kaggle.

    python -m paper.eval_physionet_comprehensive
    python -m paper.eval_physionet_comprehensive --subjects 1-10   # smoke test
"""
import argparse
import csv
import os
import time

import numpy as np
from mne.decoding import CSP
from mne.filter import filter_data
from pyriemann.classification import MDM, FgMDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.linalg import orthogonal_procrustes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

from paper import config
from paper.data import load_physionet_zeroshot
from paper.eval_mekt import mekt_lite
from paper.eval_rpa import rpa_lite
from paper.utils import classification_metrics

FBCSP_BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]
RESULTS_DIR = config.RESULTS_DIR
OUT_PREFIX = "physionet_comprehensive"


def _summarize(fold_results, name):
    accs = np.array([r["accuracy"] for r in fold_results])
    kappas = np.array([r["kappa"] for r in fold_results])
    summary = {
        "name": name,
        "n_subjects": len(fold_results),
        "accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
        "accuracy_median": np.median(accs),
        "kappa_mean": kappas.mean(), "kappa_std": kappas.std(),
    }
    print(f"[{name}] n={summary['n_subjects']} LOSO accuracy: {summary['accuracy_mean']:.4f} +/- "
          f"{summary['accuracy_std']:.4f}  kappa: {summary['kappa_mean']:.4f} +/- {summary['kappa_std']:.4f}")
    return summary


def _fold_csv_path(method_key):
    return os.path.join(RESULTS_DIR, f"{OUT_PREFIX}_{method_key}.csv")


def _load_done_folds(method_key):
    """Resume support: this environment has had background jobs killed mid-run by unrelated
    environment restarts more than once during this revision -- returns
    {held_out_subject: {accuracy, kappa}} already written for this method, so a restart only
    loses the in-progress fold, not the whole method (same convention as
    kaggle_kernel/physionet_full_pipeline_push/run_kernel.py and
    paper/eval_physionet_loso_deep.py)."""
    path = _fold_csv_path(method_key)
    done = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                done[int(row["held_out_subject"])] = {
                    "accuracy": float(row["accuracy"]), "kappa": float(row["kappa"])}
    return done


def _append_fold_csv(method_key, held_out, accuracy, kappa):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = _fold_csv_path(method_key)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "accuracy", "kappa"])
        if write_header:
            writer.writeheader()
        writer.writerow({"held_out_subject": held_out, "accuracy": accuracy, "kappa": kappa})


# --- classical pipelines, generalized from BCI-IV 2a scripts to an arbitrary (per_subject, subjects) pool ---

def run_csp_classifier(per_subject, subjects, classifier_name, n_components=8, method_key=None):
    done = _load_done_folds(method_key) if method_key else {}
    fold_results = [{"held_out_subject": s, **r} for s, r in done.items()]
    for held_out in subjects:
        if held_out in done:
            continue
        source_subjects = [s for s in subjects if s != held_out]
        source_X = np.concatenate([per_subject[s][0] for s in source_subjects], axis=0)
        source_y = np.concatenate([per_subject[s][1] for s in source_subjects], axis=0)
        target_X, target_y = per_subject[held_out]

        csp = CSP(n_components=n_components, reg="ledoit_wolf", log=True)
        source_feats = csp.fit_transform(source_X.astype(np.float64), source_y)
        target_feats = csp.transform(target_X.astype(np.float64))

        clf = (LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto") if classifier_name == "lda"
               else SVC(kernel="rbf", C=1.0, gamma="scale"))
        clf.fit(source_feats, source_y)
        pred = clf.predict(target_feats)
        metrics = classification_metrics(target_y, pred)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        if method_key:
            _append_fold_csv(method_key, held_out, metrics["accuracy"], metrics["kappa"])
    return fold_results


def run_fbcsp_svm(per_subject, subjects, n_components=4, method_key=None):
    sfreq = config.RESAMPLE_HZ
    done = _load_done_folds(method_key) if method_key else {}
    fold_results = [{"held_out_subject": s, **r} for s, r in done.items()]
    for held_out in subjects:
        if held_out in done:
            continue
        source_subjects = [s for s in subjects if s != held_out]
        source_X = np.concatenate([per_subject[s][0] for s in source_subjects], axis=0)
        source_y = np.concatenate([per_subject[s][1] for s in source_subjects], axis=0)
        target_X, target_y = per_subject[held_out]

        source_feats_bands, target_feats_bands = [], []
        for l_freq, h_freq in FBCSP_BANDS:
            source_band = filter_data(source_X.astype(np.float64), sfreq, l_freq, h_freq, verbose=False)
            target_band = filter_data(target_X.astype(np.float64), sfreq, l_freq, h_freq, verbose=False)
            csp = CSP(n_components=n_components, reg="ledoit_wolf", log=True)
            source_feats_bands.append(csp.fit_transform(source_band, source_y))
            target_feats_bands.append(csp.transform(target_band))

        source_feats = np.concatenate(source_feats_bands, axis=1)
        target_feats = np.concatenate(target_feats_bands, axis=1)
        clf = SVC(kernel="rbf", C=1.0, gamma="scale")
        clf.fit(source_feats, source_y)
        pred = clf.predict(target_feats)
        metrics = classification_metrics(target_y, pred)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        if method_key:
            _append_fold_csv(method_key, held_out, metrics["accuracy"], metrics["kappa"])
    return fold_results


def run_mdm_family(per_subject, subjects, classifier_name, method_key=None):
    done = _load_done_folds(method_key) if method_key else {}
    fold_results = [{"held_out_subject": s, **r} for s, r in done.items()]
    for held_out in subjects:
        if held_out in done:
            continue
        source_subjects = [s for s in subjects if s != held_out]
        source_X = np.concatenate([per_subject[s][0] for s in source_subjects], axis=0)
        source_y = np.concatenate([per_subject[s][1] for s in source_subjects], axis=0)
        target_X, target_y = per_subject[held_out]

        cov = Covariances(estimator="lwf")
        source_covs = cov.fit_transform(source_X.astype(np.float64))
        target_covs = cov.transform(target_X.astype(np.float64))
        clf = MDM(metric="riemann") if classifier_name == "mdm" else FgMDM(metric="riemann", tsupdate=False)
        clf.fit(source_covs, source_y)
        pred = clf.predict(target_covs)
        metrics = classification_metrics(target_y, pred)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        if method_key:
            _append_fold_csv(method_key, held_out, metrics["accuracy"], metrics["kappa"])
    return fold_results


def run_rpa(per_subject, subjects, cov_estimator="lwf", n_iter=5, classes=None, method_key=None):
    cov = Covariances(estimator=cov_estimator)
    done = _load_done_folds(method_key) if method_key else {}
    fold_results = [{"held_out_subject": s, **r} for s, r in done.items()]
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

        pred = rpa_lite(Xs, source_y, Xt, classes, n_iter=n_iter)
        metrics = classification_metrics(target_y, pred)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        if method_key:
            _append_fold_csv(method_key, held_out, metrics["accuracy"], metrics["kappa"])
    return fold_results


def run_mekt(per_subject, subjects, k=100, lam=1.0, n_iter=5, cov_estimator="lwf", classes=None, method_key=None):
    cov = Covariances(estimator=cov_estimator)
    done = _load_done_folds(method_key) if method_key else {}
    fold_results = [{"held_out_subject": s, **r} for s, r in done.items()]
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
        k_eff = min(k, Xs.shape[1])

        pred = mekt_lite(Xs, source_y, Xt, classes, k=k_eff, lam=lam, n_iter=n_iter)
        metrics = classification_metrics(target_y, pred)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        if method_key:
            _append_fold_csv(method_key, held_out, metrics["accuracy"], metrics["kappa"])
    return fold_results


METHOD_SPECS = [
    ("ea_csp8_lda", "EA + CSP(8) + shrinkage LDA",
     lambda ps, subj, classes, key: run_csp_classifier(ps, subj, "lda", method_key=key)),
    ("ea_csp8_svm", "EA + CSP(8) + SVM (RBF)",
     lambda ps, subj, classes, key: run_csp_classifier(ps, subj, "svm", method_key=key)),
    ("ea_fbcsp_svm", "EA + FBCSP(5 bands, 4 comp each) + SVM (RBF)",
     lambda ps, subj, classes, key: run_fbcsp_svm(ps, subj, method_key=key)),
    ("ea_mdm", "EA + Covariances(lwf) + MDM",
     lambda ps, subj, classes, key: run_mdm_family(ps, subj, "mdm", method_key=key)),
    ("ea_fgmdm", "EA + Covariances(lwf) + FgMDM",
     lambda ps, subj, classes, key: run_mdm_family(ps, subj, "fgmdm", method_key=key)),
    ("rpa", "EA + Covariances(lwf) + TangentSpace + RPA (Procrustes rotation, 5 iter)",
     lambda ps, subj, classes, key: run_rpa(ps, subj, n_iter=5, classes=classes, method_key=key)),
    ("mekt_default", "EA + MEKT-lite (k=100, lam=1.0, 5 iterations) -- BCI-IV 2a default config",
     lambda ps, subj, classes, key: run_mekt(ps, subj, k=100, lam=1.0, n_iter=5, classes=classes, method_key=key)),
    ("mekt_tuned", "EA + MEKT-lite (k=50, lam=1.0, 20 iterations) -- BCI-IV 2a tuned/best-single config",
     lambda ps, subj, classes, key: run_mekt(ps, subj, k=50, lam=1.0, n_iter=20, classes=classes, method_key=key)),
]


def parse_subject_range(spec):
    if spec is None:
        return None
    lo, hi = spec.split("-")
    return list(range(int(lo), int(hi) + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=str, default=None,
                         help="e.g. '1-10' to restrict to a subject-ID range for a smoke test; "
                              "default is the full available PhysioNet population.")
    parser.add_argument("--methods", type=str, default=None,
                         help="Comma-separated subset of method keys to run (default: all).")
    args = parser.parse_args()

    requested_subjects = parse_subject_range(args.subjects) or list(range(1, 110))
    method_keys = args.methods.split(",") if args.methods else [m[0] for m in METHOD_SPECS]

    print(f"=== Loading PhysioNet zero-shot data (EA-aligned), requested subjects: "
          f"{requested_subjects[0]}-{requested_subjects[-1]} (n={len(requested_subjects)}) ===")
    per_subject, label_encoder, good_subjects = load_physionet_zeroshot(
        requested_subjects, align="ea")
    classes = np.arange(len(label_encoder.classes_))
    print(f"=== {len(good_subjects)}/{len(requested_subjects)} subjects loaded successfully "
          f"(classes: {list(label_encoder.classes_)}) ===")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, f"{OUT_PREFIX}_summary.csv")

    # Resume support: if this process was interrupted earlier (e.g. an environment restart killed
    # a prior background run -- happened once during this revision), the per-method fold CSVs for
    # already-completed methods still exist on disk even though this invocation's method_keys only
    # lists the remaining ones. Rebuild their summary rows from those files first so re-running a
    # subset never drops previously-completed methods from the summary CSV.
    summaries_by_key = {}
    for key, label, _ in METHOD_SPECS:
        fold_path = os.path.join(RESULTS_DIR, f"{OUT_PREFIX}_{key}.csv")
        if not os.path.exists(fold_path):
            continue
        with open(fold_path, newline="") as f:
            rows = list(csv.DictReader(f))
        fold_results = [{"accuracy": float(r["accuracy"]), "kappa": float(r["kappa"])} for r in rows]
        summary = _summarize(fold_results, label)
        summary["method_key"] = key
        summary["elapsed_seconds"] = ""  # unknown for a resumed/pre-existing method
        summaries_by_key[key] = summary

    def _flush_summary():
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method_key", "name", "n_subjects",
                                                     "accuracy_mean", "accuracy_std", "accuracy_median",
                                                     "kappa_mean", "kappa_std", "elapsed_seconds"])
            writer.writeheader()
            for key, _, _ in METHOD_SPECS:
                if key in summaries_by_key:
                    writer.writerow(summaries_by_key[key])

    _flush_summary()  # persist the resumed state immediately, before (re)computing anything new

    for key, label, fn in METHOD_SPECS:
        if key not in method_keys:
            continue
        print(f"\n--- Running {label} ({key}) across {len(good_subjects)} subjects ---")
        t0 = time.time()
        fold_results = fn(per_subject, good_subjects, classes, key)
        elapsed = time.time() - t0
        summary = _summarize(fold_results, label)
        summary["method_key"] = key
        summary["elapsed_seconds"] = elapsed
        summaries_by_key[key] = summary
        print(f"  elapsed: {elapsed:.1f}s ({elapsed / max(len(good_subjects), 1):.2f}s/fold)")

        # Write/overwrite the summary CSV incrementally so partial progress is never lost.
        _flush_summary()

    summaries = [summaries_by_key[key] for key, _, _ in METHOD_SPECS if key in summaries_by_key]

    print("\n=== Summary: comprehensive classical-method PhysioNet comparison ===")
    for s in summaries:
        print(f"{s['name']}: acc={s['accuracy_mean']*100:.2f}% +/- {s['accuracy_std']*100:.2f}% "
              f"(n={s['n_subjects']}) kappa={s['kappa_mean']:.3f} +/- {s['kappa_std']:.3f}")


if __name__ == "__main__":
    main()
