"""TMLR revision round 2, task E7: reproduce the confidence-weighted fusion refinement (reported
in the manuscript as 52.53%, Section~\\ref{sec:negative}) from its specification, since the
original implementation was not recovered (disclosed in 3 places in the manuscript as not
independently reproducible).

Spec, exactly as the manuscript and task describe it: starting from the validated 8-member
MEKT-lite ensemble (paper/eval_mekt_ensemble.py, 4 windows x 2 covariance estimators, 52.39%),
replace equal-weight averaging with a label-free, PER-TRIAL confidence-weighted fusion -- each
member's vote for a given trial is weighted by that member's own max-softmax probability on that
specific trial, using no target-subject labels anywhere (the weights come only from each member's
own predicted-probability vector for that trial, never from ground truth).

If this reproduces the reported 52.53% within 0.05pp: keep it, with this script as the citable
implementation, and the manuscript's 3 "not reproducible" disclosures can be updated. If it does
not: report the reproduced value honestly everywhere 52.53% currently appears, and drop 52.53% as
a specific number (the manuscript already discloses the attempt and its outcome either way).

    python -m paper.eval_e7_confidence_weighted_fusion
"""
import csv
import os

import numpy as np

from paper import config
from paper.eval_mekt_ensemble import WINDOWS, ESTIMATORS, _precompute_covariances, _member_proba
from paper.utils import classification_metrics

REPORTED_VALUE = 0.5253
REPRODUCTION_TOLERANCE = 0.0005  # 0.05pp


def confidence_weighted_fusion(member_probs):
    """member_probs: list of (n_trials, n_classes) arrays, one per ensemble member. Per trial,
    each member's weight is its OWN max-softmax probability for that trial (label-free); the
    fused probability is the per-trial weighted average across members."""
    stacked = np.stack(member_probs, axis=0)  # (n_members, n_trials, n_classes)
    weights = stacked.max(axis=-1)  # (n_members, n_trials) -- each member's own confidence per trial
    weights_sum = weights.sum(axis=0, keepdims=True)  # (1, n_trials)
    weights_sum = np.where(weights_sum == 0, 1.0, weights_sum)  # guard divide-by-zero
    normalized = weights / weights_sum  # (n_members, n_trials)
    fused = (stacked * normalized[:, :, None]).sum(axis=0)  # (n_trials, n_classes)
    return fused


def run(subjects=None):
    config.set_seed(config.SEED)
    subjects = subjects or config.EVAL_SUBJECTS
    candidate_keys = [(tmin, tmax, est) for tmin, tmax in WINDOWS for est in ESTIMATORS]

    covs, labels, classes = _precompute_covariances(subjects)

    fold_results_equal = []
    fold_results_confweighted = []
    for held_out in subjects:
        source_subjects = [s for s in subjects if s != held_out]
        member_probs = [
            _member_proba(covs[key], labels, source_subjects, held_out, classes)
            for key in candidate_keys
        ]

        equal_proba = sum(member_probs) / len(member_probs)
        equal_pred = equal_proba.argmax(axis=1)
        equal_metrics = classification_metrics(labels[held_out], equal_pred)
        equal_metrics["held_out_subject"] = held_out
        fold_results_equal.append(equal_metrics)

        conf_proba = confidence_weighted_fusion(member_probs)
        conf_pred = conf_proba.argmax(axis=1)
        conf_metrics = classification_metrics(labels[held_out], conf_pred)
        conf_metrics["held_out_subject"] = held_out
        fold_results_confweighted.append(conf_metrics)

        print(f"[e7] held_out={held_out} equal_weight_acc={equal_metrics['accuracy']*100:.2f}% "
              f"conf_weighted_acc={conf_metrics['accuracy']*100:.2f}%", flush=True)

    equal_accs = np.array([r["accuracy"] for r in fold_results_equal])
    conf_accs = np.array([r["accuracy"] for r in fold_results_confweighted])

    print(f"\n[e7] Equal-weight ensemble (reference): {equal_accs.mean()*100:.2f}% +/- "
          f"{equal_accs.std()*100:.2f}%")
    print(f"[e7] Confidence-weighted fusion (reproduced): {conf_accs.mean()*100:.2f}% +/- "
          f"{conf_accs.std()*100:.2f}%")
    print(f"[e7] Manuscript's reported confidence-weighted value: {REPORTED_VALUE*100:.2f}%")
    diff_from_reported = conf_accs.mean() - REPORTED_VALUE
    reproduces = abs(diff_from_reported) <= REPRODUCTION_TOLERANCE
    print(f"[e7] Difference from reported: {diff_from_reported*100:+.3f}pp "
          f"({'REPRODUCES within 0.05pp' if reproduces else 'DOES NOT reproduce within 0.05pp'})")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, "e7_confidence_weighted_fusion.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "accuracy", "kappa"])
        writer.writeheader()
        for r in fold_results_confweighted:
            writer.writerow(r)

    summary_path = os.path.join(config.RESULTS_DIR, "e7_confidence_weighted_fusion_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerow({"metric": "equal_weight_accuracy_mean", "value": equal_accs.mean()})
        writer.writerow({"metric": "equal_weight_accuracy_std", "value": equal_accs.std()})
        writer.writerow({"metric": "conf_weighted_accuracy_mean", "value": conf_accs.mean()})
        writer.writerow({"metric": "conf_weighted_accuracy_std", "value": conf_accs.std()})
        writer.writerow({"metric": "manuscript_reported_value", "value": REPORTED_VALUE})
        writer.writerow({"metric": "diff_from_reported_pp", "value": diff_from_reported * 100})
        writer.writerow({"metric": "reproduces_within_0.05pp", "value": reproduces})

    print(f"\nWrote per-fold results to {out_path}, summary to {summary_path}")
    return fold_results_confweighted, conf_accs.mean(), reproduces


if __name__ == "__main__":
    run()
