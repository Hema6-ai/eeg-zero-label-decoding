"""Tests whether cross-subject distribution shift, not representation quality, is the zero-label
bottleneck: a classical Riemannian-geometry classifier (Tangent Space + Logistic Regression) run
LOSO on BCI-IV 2a with no alignment, Euclidean Alignment, and Riemannian Alignment. This is fully
independent of the MAE/CLUB/GRL pipeline -- it isolates the "does alignment help at all on this
dataset" question from any confound in our deep model.

Phase-5 follow-up (item 1): this script originally only printed per-subject results and never
saved them to a per-subject CSV or ran a significance test -- the manuscript's "+11.24pp" EA
effect for tangent-space LR was therefore a raw accuracy delta, not a Holm/Wilcoxon-tested claim
like every other number in this paper. Fixed: per-subject CSVs are now saved and a paired
Wilcoxon (none -> ea) is computed and printed/saved.

    python -m paper.eval_alignment_baseline
"""
import csv
import os

import numpy as np
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression

from paper import config
from paper.align import align_per_subject, _trial_covariances
from paper.data import load_bnci_all
from paper.stats_analysis import paired_comparison_from_dicts, save_results
from paper.utils import classification_metrics

RESULTS_DIR = config.RESULTS_DIR


def run(subjects=None):
    subjects = subjects or config.EVAL_SUBJECTS
    per_subject, label_encoder = load_bnci_all(subjects)

    results = {}
    per_subject_acc = {}
    for method in ("none", "ea", "ra"):
        X_only = {s: X for s, (X, y) in per_subject.items()}
        aligned_X = align_per_subject(X_only, method)
        aligned = {s: (aligned_X[s], per_subject[s][1]) for s in per_subject}

        fold_results = []
        for held_out in subjects:
            source_subjects = [s for s in subjects if s != held_out]
            source_X = np.concatenate([aligned[s][0] for s in source_subjects], axis=0)
            source_y = np.concatenate([per_subject[s][1] for s in source_subjects], axis=0)
            target_X, target_y = aligned[held_out][0], per_subject[held_out][1]

            source_covs = _trial_covariances(source_X)
            target_covs = _trial_covariances(target_X)

            ts = TangentSpace(metric="riemann")
            source_feats = ts.fit_transform(source_covs)
            target_feats = ts.transform(target_covs)

            clf = LogisticRegression(max_iter=1000)
            clf.fit(source_feats, source_y)
            pred = clf.predict(target_feats)

            metrics = classification_metrics(target_y, pred)
            metrics["held_out_subject"] = held_out
            fold_results.append(metrics)
            print(f"[{method}] held-out subject {held_out}: "
                  f"acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

        accs = np.array([r["accuracy"] for r in fold_results])
        kappas = np.array([r["kappa"] for r in fold_results])
        results[method] = {
            "accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
            "kappa_mean": kappas.mean(), "kappa_std": kappas.std(),
        }
        per_subject_acc[method] = {r["held_out_subject"]: r["accuracy"] for r in fold_results}
        print(f"[{method}] LOSO accuracy: {accs.mean():.4f} +/- {accs.std():.4f}  "
              f"kappa: {kappas.mean():.4f} +/- {kappas.std():.4f}\n")

        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, f"alignment_baseline_tslr_{method}.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["held_out_subject", "accuracy", "kappa"])
            writer.writeheader()
            for r in fold_results:
                writer.writerow({"held_out_subject": r["held_out_subject"],
                                  "accuracy": r["accuracy"], "kappa": r["kappa"]})

    print("=== Summary ===")
    for method, r in results.items():
        print(f"{method:>4}: acc={r['accuracy_mean']*100:.2f}% +/- {r['accuracy_std']*100:.2f}%  "
              f"kappa={r['kappa_mean']:.3f} +/- {r['kappa_std']:.3f}")

    print("\n=== Paired Wilcoxon: none -> ea (the manuscript's cited comparison) ===")
    stat_result = paired_comparison_from_dicts("TS+LR (BCI-2a): no-EA -> EA",
                                                 per_subject_acc["none"], per_subject_acc["ea"])
    if stat_result is not None:
        save_results([stat_result], "stats_alignment_baseline_tslr.csv")

    return results


if __name__ == "__main__":
    run()
