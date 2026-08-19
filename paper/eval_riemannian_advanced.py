"""Four pyriemann-based EA pipelines, zero-shot LOSO on BCI-IV 2a (BNCI2014_001):

    1. EA + Covariances(lwf) + MDM(riemann)               -- direct Riemannian-distance classifier
    2. EA + Covariances(lwf) + FgMDM(riemann)              -- geodesic-filtered MDM, built for transfer
    3. Per-class EA (source only) + TangentSpace + shrinkage LDA
    4. Frequency-band ensemble (mu 8-12Hz, beta 16-24Hz) TangentSpace + LogisticRegression + EA,
       softmax-averaged across bands

    python -m paper.eval_riemannian_advanced

IMPORTANT PROTOCOL NOTE on (3): "align each class independently" requires trial labels to know
which class-specific reference to align each trial to. That's valid for the 8 SOURCE subjects
(their labels are used for training anyway), but NOT valid for the held-out TARGET subject in a
genuine zero-shot setting -- using its labels to pick a per-class alignment would be calibration-
data leakage. So per-class EA is applied to the source pool only; the held-out subject still gets
the standard label-free (global) EA, which is the only valid choice without seeing its labels.
"""
import csv
import os

import numpy as np
from mne.filter import filter_data
from pyriemann.classification import MDM, FgMDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from paper import config
from paper.align import align_per_subject, euclidean_alignment
from paper.data import load_bnci_all
from paper.utils import classification_metrics


def _ea_aligned(per_subject, subjects):
    X_only = {s: per_subject[s][0] for s in subjects}
    aligned_X = align_per_subject(X_only, "ea")
    return {s: (aligned_X[s].astype(np.float32), per_subject[s][1]) for s in subjects}


def _loso_split(per_subject, subjects, held_out):
    source_subjects = [s for s in subjects if s != held_out]
    return source_subjects, per_subject[held_out]


def _summarize(fold_results, name):
    accs = np.array([r["accuracy"] for r in fold_results])
    kappas = np.array([r["kappa"] for r in fold_results])
    summary = {"name": name, "accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
               "kappa_mean": kappas.mean(), "kappa_std": kappas.std()}
    print(f"[{name}] LOSO accuracy: {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}  "
          f"kappa: {summary['kappa_mean']:.4f} +/- {summary['kappa_std']:.4f}\n")
    return summary, fold_results


# --- 1 & 2: MDM / FgMDM ---
def run_mdm_family(classifier_name, subjects=None):
    subjects = subjects or config.EVAL_SUBJECTS
    per_subject, label_encoder = load_bnci_all(subjects)
    aligned = _ea_aligned(per_subject, subjects)

    fold_results = []
    for held_out in subjects:
        source_subjects, (target_X, target_y) = _loso_split(aligned, subjects, held_out)
        source_X = np.concatenate([aligned[s][0] for s in source_subjects], axis=0)
        source_y = np.concatenate([aligned[s][1] for s in source_subjects], axis=0)

        cov = Covariances(estimator="lwf")
        source_covs = cov.fit_transform(source_X.astype(np.float64))
        target_covs = cov.transform(target_X.astype(np.float64))

        clf = MDM(metric="riemann") if classifier_name == "mdm" else FgMDM(metric="riemann", tsupdate=False)
        clf.fit(source_covs, source_y)
        pred = clf.predict(target_covs)

        metrics = classification_metrics(target_y, pred)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        print(f"[EA+{classifier_name}] held-out subject {held_out}: "
              f"acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

    return _summarize(fold_results, f"EA+{classifier_name}")


# --- 3: per-class EA (source only) + TS + shrinkage LDA ---
def _per_class_ea(X, y):
    aligned = np.zeros_like(X)
    for c in np.unique(y):
        mask = y == c
        aligned[mask] = euclidean_alignment(X[mask])
    return aligned


def run_tslda_per_class_ea(subjects=None):
    subjects = subjects or config.EVAL_SUBJECTS
    per_subject, label_encoder = load_bnci_all(subjects)  # raw, pre-alignment

    fold_results = []
    for held_out in subjects:
        source_subjects, (target_X_raw, target_y) = _loso_split(per_subject, subjects, held_out)

        source_X_parts, source_y_parts = [], []
        for s in source_subjects:
            Xs, ys = per_subject[s]
            source_X_parts.append(_per_class_ea(Xs.astype(np.float32), ys))
            source_y_parts.append(ys)
        source_X = np.concatenate(source_X_parts, axis=0)
        source_y = np.concatenate(source_y_parts, axis=0)
        # Target: standard label-free (global) EA only -- see module docstring for why.
        target_X = euclidean_alignment(target_X_raw.astype(np.float32))

        cov = Covariances(estimator="lwf")
        source_covs = cov.fit_transform(source_X.astype(np.float64))
        target_covs = cov.transform(target_X.astype(np.float64))

        ts = TangentSpace(metric="riemann")
        source_feats = ts.fit_transform(source_covs)
        target_feats = ts.transform(target_covs)

        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(source_feats, source_y)
        pred = clf.predict(target_feats)

        metrics = classification_metrics(target_y, pred)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        print(f"[TSLDA+per-class-EA] held-out subject {held_out}: "
              f"acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

    return _summarize(fold_results, "TSLDA+per-class-EA")


# --- 4: frequency-band ensemble TS+LR+EA ---
def run_band_ensemble(subjects=None, bands=((8, 12), (16, 24))):
    subjects = subjects or config.EVAL_SUBJECTS
    per_subject, label_encoder = load_bnci_all(subjects)  # raw, pre-filter
    sfreq = config.RESAMPLE_HZ

    fold_results = []
    for held_out in subjects:
        source_subjects = [s for s in subjects if s != held_out]
        target_y = per_subject[held_out][1]
        avg_probs = None

        for l_freq, h_freq in bands:
            per_subject_band = {
                s: filter_data(per_subject[s][0].astype(np.float64), sfreq, l_freq, h_freq, verbose=False)
                for s in subjects
            }
            aligned_band = align_per_subject(per_subject_band, "ea")

            source_X = np.concatenate([aligned_band[s] for s in source_subjects], axis=0)
            source_y = np.concatenate([per_subject[s][1] for s in source_subjects], axis=0)
            target_X = aligned_band[held_out]

            cov = Covariances(estimator="lwf")
            source_covs = cov.fit_transform(source_X)
            target_covs = cov.transform(target_X)

            ts = TangentSpace(metric="riemann")
            source_feats = ts.fit_transform(source_covs)
            target_feats = ts.transform(target_covs)

            clf = LogisticRegression(C=1.0, max_iter=1000)
            clf.fit(source_feats, source_y)
            probs = clf.predict_proba(target_feats)
            avg_probs = probs if avg_probs is None else avg_probs + probs

        avg_probs /= len(bands)
        pred = np.argmax(avg_probs, axis=1)

        metrics = classification_metrics(target_y, pred)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        print(f"[band-ensemble-TS+LR+EA] held-out subject {held_out}: "
              f"acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

    return _summarize(fold_results, "band-ensemble-TS+LR+EA")


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    runs = [
        (run_mdm_family("mdm"), "EA + Covariances(lwf) + MDM"),
        (run_mdm_family("fgmdm"), "EA + Covariances(lwf) + FgMDM"),
        (run_tslda_per_class_ea(), "Per-class EA (source) + TS + shrinkage LDA"),
        (run_band_ensemble(), "Band ensemble (mu+beta) TS+LR+EA"),
    ]

    path = os.path.join(config.RESULTS_DIR, "final_results_table.csv")
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy_mean", "accuracy_std",
                                                "kappa_mean", "kappa_std", "subject_probe_acc", "notes"])
        for (summary, folds), label in runs:
            writer.writerow({
                "model": label,
                "accuracy_mean": summary["accuracy_mean"],
                "accuracy_std": summary["accuracy_std"],
                "kappa_mean": summary["kappa_mean"],
                "kappa_std": summary["kappa_std"],
                "subject_probe_acc": "",
                "notes": "Zero-shot LOSO on BCI-IV 2a, pyriemann-based, EA per-subject",
            })

    with open(os.path.join(config.RESULTS_DIR, "riemannian_advanced_experiment.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "accuracy_mean", "accuracy_std", "kappa_mean", "kappa_std"])
        for (summary, folds), label in runs:
            writer.writerow([label, summary["accuracy_mean"], summary["accuracy_std"],
                              summary["kappa_mean"], summary["kappa_std"]])
        # per-fold breakdown for whichever is best, appended below
        best = max(runs, key=lambda r: r[0][0]["accuracy_mean"])
        writer.writerow([])
        writer.writerow([f"per-fold breakdown for best method: {best[1]}"])
        writer.writerow(["held_out_subject", "accuracy", "kappa"])
        for r in best[0][1]:
            writer.writerow([r["held_out_subject"], r["accuracy"], r["kappa"]])

    print("=== Summary ===")
    for (summary, folds), label in runs:
        crossed_60 = summary["accuracy_mean"] > 0.60
        print(f"{label}: acc={summary['accuracy_mean']*100:.2f}% +/- {summary['accuracy_std']*100:.2f}%  "
              f"kappa={summary['kappa_mean']:.3f} +/- {summary['kappa_std']:.3f}  "
              f"CROSSED 60%? {'YES' if crossed_60 else 'NO'}")

    best = max(runs, key=lambda r: r[0][0]["accuracy_mean"])
    print(f"\nBest method: {best[1]} at {best[0][0]['accuracy_mean']*100:.2f}%")
    print("Per-fold breakdown for best method:")
    for r in best[0][1]:
        print(f"  subject {r['held_out_subject']}: acc={r['accuracy']:.4f} kappa={r['kappa']:.4f}")


if __name__ == "__main__":
    main()
