"""EA + classical BCI pipelines, zero-shot LOSO on BCI-IV 2a (BNCI2014_001):

    1. EA + CSP(8) + shrinkage LDA
    2. EA + CSP(8) + SVM (RBF)
    3. EA + FBCSP (5 bands, CSP(4) each) + SVM (RBF)

Euclidean Alignment is applied per-subject (unsupervised, that subject's own trial covariances
only) to both the pooled source subjects and the held-out subject -- valid with no label leakage.
A single CSP/FBCSP+classifier is then fit on the pooled aligned source subjects and evaluated
zero-shot on the aligned held-out subject.

    python -m paper.eval_csp_lda_svm
"""
import csv
import os

import numpy as np
from mne.decoding import CSP
from mne.filter import filter_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

from paper import config
from paper.align import align_per_subject
from paper.data import load_bnci_all
from paper.utils import classification_metrics

FBCSP_BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]


def _ea_aligned(per_subject, subjects):
    X_only = {s: per_subject[s][0] for s in subjects}
    aligned_X = align_per_subject(X_only, "ea")
    return {s: (aligned_X[s].astype(np.float32), per_subject[s][1]) for s in subjects}


def _loso_folds(aligned, subjects):
    for held_out in subjects:
        source_subjects = [s for s in subjects if s != held_out]
        source_X = np.concatenate([aligned[s][0] for s in source_subjects], axis=0)
        source_y = np.concatenate([aligned[s][1] for s in source_subjects], axis=0)
        target_X, target_y = aligned[held_out]
        yield held_out, source_X, source_y, target_X, target_y


def run_csp_classifier(classifier_name, subjects=None, n_components=8):
    subjects = subjects or config.EVAL_SUBJECTS
    per_subject, label_encoder = load_bnci_all(subjects)
    aligned = _ea_aligned(per_subject, subjects)

    fold_results = []
    for held_out, source_X, source_y, target_X, target_y in _loso_folds(aligned, subjects):
        csp = CSP(n_components=n_components, reg="ledoit_wolf", log=True)
        source_feats = csp.fit_transform(source_X.astype(np.float64), source_y)
        target_feats = csp.transform(target_X.astype(np.float64))

        if classifier_name == "lda":
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        elif classifier_name == "svm":
            clf = SVC(kernel="rbf", C=1.0, gamma="scale")
        else:
            raise ValueError(classifier_name)

        clf.fit(source_feats, source_y)
        pred = clf.predict(target_feats)
        metrics = classification_metrics(target_y, pred)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        print(f"[EA+CSP+{classifier_name}] held-out subject {held_out}: "
              f"acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

    return _summarize(fold_results, f"EA+CSP+{classifier_name}")


def run_fbcsp_svm(subjects=None, n_components=4):
    subjects = subjects or config.EVAL_SUBJECTS
    per_subject, label_encoder = load_bnci_all(subjects)
    aligned = _ea_aligned(per_subject, subjects)
    sfreq = config.RESAMPLE_HZ

    fold_results = []
    for held_out, source_X, source_y, target_X, target_y in _loso_folds(aligned, subjects):
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
        print(f"[EA+FBCSP+svm] held-out subject {held_out}: "
              f"acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

    return _summarize(fold_results, "EA+FBCSP+svm")


def _summarize(fold_results, name):
    accs = np.array([r["accuracy"] for r in fold_results])
    kappas = np.array([r["kappa"] for r in fold_results])
    summary = {"name": name, "accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
               "kappa_mean": kappas.mean(), "kappa_std": kappas.std()}
    print(f"[{name}] LOSO accuracy: {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}  "
          f"kappa: {summary['kappa_mean']:.4f} +/- {summary['kappa_std']:.4f}\n")
    return summary, fold_results


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    summaries = []

    summary, folds = run_csp_classifier("lda")
    summaries.append((summary, folds, "EA + CSP(8) + shrinkage LDA"))

    summary, folds = run_csp_classifier("svm")
    summaries.append((summary, folds, "EA + CSP(8) + SVM (RBF)"))

    summary, folds = run_fbcsp_svm()
    summaries.append((summary, folds, "EA + FBCSP(5 bands, 4 comp each) + SVM (RBF)"))

    path = os.path.join(config.RESULTS_DIR, "final_results_table.csv")
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy_mean", "accuracy_std",
                                                "kappa_mean", "kappa_std", "subject_probe_acc", "notes"])
        for summary, folds, label in summaries:
            writer.writerow({
                "model": label,
                "accuracy_mean": summary["accuracy_mean"],
                "accuracy_std": summary["accuracy_std"],
                "kappa_mean": summary["kappa_mean"],
                "kappa_std": summary["kappa_std"],
                "subject_probe_acc": "",
                "notes": "Zero-shot LOSO on BCI-IV 2a; Euclidean Alignment per-subject "
                         "(unsupervised) applied to both source pool and held-out subject",
            })

    with open(os.path.join(config.RESULTS_DIR, "csp_lda_svm_experiment.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "accuracy_mean", "accuracy_std", "kappa_mean", "kappa_std"])
        for summary, folds, label in summaries:
            writer.writerow([label, summary["accuracy_mean"], summary["accuracy_std"],
                              summary["kappa_mean"], summary["kappa_std"]])

    print("=== Summary ===")
    for summary, folds, label in summaries:
        crossed_60 = summary["accuracy_mean"] > 0.60
        print(f"{label}: acc={summary['accuracy_mean']*100:.2f}% +/- {summary['accuracy_std']*100:.2f}%  "
              f"kappa={summary['kappa_mean']:.3f} +/- {summary['kappa_std']:.3f}  "
              f"CROSSED 60%? {'YES' if crossed_60 else 'NO'}")


if __name__ == "__main__":
    main()
