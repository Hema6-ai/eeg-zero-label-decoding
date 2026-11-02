"""TMLR revision round 2, task E6: strict inductive vs. transductive target-data access.

Directly answers reviewer Suggestion 1 ("clarify strict zero-shot decoding vs. transductive
adaptation using unlabeled target-subject data") with a quantitative comparison, rather than only
the qualitative terminology note already in Section 3.1 (sec:canon). For FgMDM and MDM (BCI-IV 2a,
LOSO, default window [0,3]s -- window held fixed since this task isolates TARGET-DATA ACCESS, not
preprocessing), compares five target-side regimes that differ only in how much of the held-out
subject's own unlabeled data each is allowed to see before producing a prediction, from least to
most target-data-hungry:

  1. none              -- no target-side (or source-side) preprocessing at all. Strictest:
                           every target trial predicted independently, zero bytes of any OTHER
                           target trial ever touched.
  2. online_causal_ea  -- running reference covariance, updated trial-by-trial, STARTING FROM THE
                           SOURCE-MEAN REFERENCE (pooled mean covariance of all source trials,
                           pseudo-weight 1); trial t's prediction uses only target trials 1..t-1 --
                           a strictly causal/inductive online setting, never batch access.
  3. partial_ea_N      -- batch EA, but the reference covariance is computed from only the first
                           N target trials (recording order); N in {8,16,32,64}. Still "batch" in
                           that the reference is fixed before any prediction and then applied to
                           ALL target trials (including the first N), but with limited data access.
  4. batch_ea          -- this paper's standard EA: reference from ALL target trials collectively.
                           The "current" condition used everywhere else in this paper.
  5. mekt_transductive -- MEKT-lite (JDA, tangent-space pseudo-labeling), which iterates over the
                           FULL unlabeled target set at once. The most target-data-hungry
                           condition tested, and the only genuinely transductive (not just aligned)
                           one.

Source subjects always get this paper's standard batch EA from their own complete recording
(unambiguous -- source labels are available, and this matches the convention used everywhere else
in this paper) for every row except "none", where source is also left unaligned, matching this
paper's existing FgMDM (no EA) vs. EA+FgMDM baseline convention. Only the TARGET subject's access
regime varies across rows 1-5.

    python -m paper.eval_e6_transductive_inductive
"""
import csv
import os

import numpy as np
from pyriemann.classification import FgMDM, MDM
from pyriemann.estimation import Covariances
from pyriemann.geometry.base import invsqrtm
from pyriemann.tangentspace import TangentSpace

from paper import config
from paper.align import _trial_covariances
from paper.data import load_bnci_all
from paper.eval_mekt import mekt_lite
from paper.utils import classification_metrics

RESULTS_DIR = config.RESULTS_DIR
PARTIAL_NS = [8, 16, 32, 64]
MEKT_K, MEKT_LAM, MEKT_N_ITER = 50, 1.0, 20  # this paper's headline-tuned MEKT hyperparameters


def _ea_whiten(X, ref_cov):
    whitening = invsqrtm(ref_cov)
    return np.array([whitening @ x for x in X])


def batch_ea(X):
    covs = _trial_covariances(X)
    return _ea_whiten(X, covs.mean(axis=0)), covs.mean(axis=0)


def partial_ea(X, n):
    n = min(n, len(X))
    covs = _trial_covariances(X[:n])
    ref = covs.mean(axis=0)
    return _ea_whiten(X, ref), ref


def online_causal_ea(X, source_mean_ref):
    """Causal online EA: predicts each trial using only the reference built from strictly earlier
    target trials (starting from the source-mean reference), then updates the running reference
    with that trial's own covariance. Returns the list of per-trial ALIGNED trials (trial t aligned
    using the reference available BEFORE trial t was incorporated), for downstream classification."""
    covs = _trial_covariances(X)
    running_ref = source_mean_ref.copy()
    count = 1  # source-mean reference counts as one pseudo-trial of prior weight
    aligned = []
    for i in range(len(X)):
        whitening = invsqrtm(running_ref)
        aligned.append(whitening @ X[i])
        count += 1
        running_ref = running_ref + (covs[i] - running_ref) / count
    return np.array(aligned)


def run_fold(method_name, clf_builder, source_X, source_y, target_X, target_y, classes, source_mean_ref):
    cov = Covariances(estimator="lwf")
    results = {}

    # 1. none -- no alignment on either side
    src_covs = cov.fit_transform(source_X.astype(np.float64))
    tgt_covs = cov.transform(target_X.astype(np.float64))
    clf = clf_builder()
    clf.fit(src_covs, source_y)
    pred = clf.predict(tgt_covs)
    results["none"] = classification_metrics(target_y, pred)

    # Source-side: standard batch EA from here on (every remaining row)
    source_aligned, _ = batch_ea(source_X.astype(np.float64))
    src_covs_ea = cov.fit_transform(source_aligned)
    clf_ea = clf_builder()
    clf_ea.fit(src_covs_ea, source_y)

    # 2. online/causal EA
    target_aligned_online = online_causal_ea(target_X.astype(np.float64), source_mean_ref)
    tgt_covs_online = cov.transform(target_aligned_online)
    pred = clf_ea.predict(tgt_covs_online)
    results["online_causal_ea"] = classification_metrics(target_y, pred)

    # 3. partial EA at each N
    for n in PARTIAL_NS:
        target_aligned_n, _ = partial_ea(target_X.astype(np.float64), n)
        tgt_covs_n = cov.transform(target_aligned_n)
        pred = clf_ea.predict(tgt_covs_n)
        results[f"partial_ea_{n}"] = classification_metrics(target_y, pred)

    # 4. batch EA (all target trials)
    target_aligned_batch, _ = batch_ea(target_X.astype(np.float64))
    tgt_covs_batch = cov.transform(target_aligned_batch)
    pred = clf_ea.predict(tgt_covs_batch)
    results["batch_ea"] = classification_metrics(target_y, pred)

    # 5. MEKT transductive (N=all), on top of the same source/target batch-EA covariances
    ts = TangentSpace(metric="riemann")
    Xs = ts.fit_transform(src_covs_ea)
    Xt = ts.transform(tgt_covs_batch)
    k_eff = min(MEKT_K, Xs.shape[1])
    pred = mekt_lite(Xs, source_y, Xt, classes, k=k_eff, lam=MEKT_LAM, n_iter=MEKT_N_ITER)
    results["mekt_transductive"] = classification_metrics(target_y, pred)

    return {f"{method_name}__{cond}": metrics for cond, metrics in results.items()}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    subjects = config.EVAL_SUBJECTS
    per_subject, label_encoder = load_bnci_all(subjects, align="none")
    classes = np.arange(len(label_encoder.classes_))

    fold_rows = []
    for held_out in subjects:
        source_subjects = [s for s in subjects if s != held_out]
        source_X = np.concatenate([per_subject[s][0] for s in source_subjects], axis=0)
        source_y = np.concatenate([per_subject[s][1] for s in source_subjects], axis=0)
        target_X, target_y = per_subject[held_out]

        source_mean_ref = _trial_covariances(source_X.astype(np.float64)).mean(axis=0)

        for method_name, clf_builder in [
            ("fgmdm", lambda: FgMDM(metric="riemann", tsupdate=False)),
            ("mdm", lambda: MDM(metric="riemann")),
        ]:
            cond_results = run_fold(method_name, clf_builder, source_X, source_y, target_X, target_y,
                                     classes, source_mean_ref)
            for key, metrics in cond_results.items():
                row = {"held_out_subject": held_out, "method_condition": key,
                       "accuracy": metrics["accuracy"], "kappa": metrics["kappa"]}
                fold_rows.append(row)
                print(f"[{key}] held_out={held_out} acc={metrics['accuracy']*100:.2f}% "
                      f"kappa={metrics['kappa']:.3f}", flush=True)

    fold_path = os.path.join(RESULTS_DIR, "e6_transductive_inductive_folds.csv")
    with open(fold_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "method_condition", "accuracy", "kappa"])
        writer.writeheader()
        writer.writerows(fold_rows)

    summary_rows = []
    conditions = ["none", "online_causal_ea"] + [f"partial_ea_{n}" for n in PARTIAL_NS] + \
                 ["batch_ea", "mekt_transductive"]
    for method_name in ["fgmdm", "mdm"]:
        for cond in conditions:
            key = f"{method_name}__{cond}"
            accs = np.array([r["accuracy"] for r in fold_rows if r["method_condition"] == key])
            summary_rows.append({
                "method": method_name, "condition": cond, "n_subjects": len(accs),
                "accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
            })
            print(f"=== {key}: {accs.mean()*100:.2f}% +/- {accs.std()*100:.2f}% ===", flush=True)

    summary_path = os.path.join(RESULTS_DIR, "e6_transductive_inductive_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "condition", "n_subjects",
                                                "accuracy_mean", "accuracy_std"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nWrote {len(fold_rows)} fold rows to {fold_path}")
    print(f"Wrote {len(summary_rows)} summary rows to {summary_path}")
    return summary_rows


if __name__ == "__main__":
    main()
