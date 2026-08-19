"""MEKT-lite: Manifold Embedded Knowledge Transfer (Zhang & Wu, 2020), built on Joint
Distribution Adaptation (Long et al., 2013). Unlike subject-similarity weighting (which just
reweights samples) or EA/RA (which whitens each subject's own covariance independently), this
learns a linear transform of the tangent-space features themselves that jointly minimizes:
  - marginal MMD between source and target feature distributions, and
  - per-class conditional MMD between source classes and target's PSEUDO-labeled classes,
refining the pseudo-labels (and thus the transform) over several iterations. It's transductive: the
final iteration's pseudo-labels ARE the zero-shot predictions for the held-out subject -- no true
target labels are ever used, only the target's own unlabeled tangent-space features, so this stays
zero-shot-legal (matches the transductive unsupervised-domain-adaptation literature).

    python -m paper.eval_mekt
"""
import csv
import os

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from paper import config
from paper.data import load_bnci_all
from paper.utils import classification_metrics


def _mmd_vector(n, idx_pos, idx_neg, weights_pos=None, weights_neg=None):
    """weights_pos/weights_neg: optional per-index weights (e.g. source-subject Fisher-score
    weights) for the pos/neg groups; None means uniform (the original MEKT/JDA behavior)."""
    e = np.zeros(n)
    if weights_pos is None:
        e[idx_pos] = 1.0 / len(idx_pos)
    else:
        w = weights_pos / weights_pos.sum()
        e[idx_pos] = w
    if weights_neg is None:
        e[idx_neg] = -1.0 / len(idx_neg)
    else:
        w = weights_neg / weights_neg.sum()
        e[idx_neg] = -w
    return e


def jda_transform(Xs, ys, Xt, pseudo_yt, classes, k, lam, confident_mask=None, source_weights=None):
    """Xs: (n_s, d) source tangent vectors, Xt: (n_t, d) target. Returns (Zs, Zt) projected into
    a k-dim domain-invariant, class-discriminative subspace (top-k smallest-eigenvalue directions
    of the generalized eigenproblem (X M X^T + lam*I) a = eig * (X H X^T) a).

    confident_mask: optional (n_t,) boolean array -- if given, only CONFIDENT target pseudo-labels
    contribute to the per-class (conditional) MMD terms, so low-confidence/likely-wrong pseudo-
    labels don't corrupt the class-conditional adaptation. The marginal term (M0) always uses all
    target trials regardless, since it doesn't depend on labels at all.

    source_weights: optional (n_s,) per-SOURCE-TRIAL weight (e.g. that trial's source subject's
    Fisher-discriminability score) -- source trials from subjects with cleaner class separability
    contribute more to both the marginal and per-class MMD terms. Target-side weighting is always
    uniform (a single held-out subject has no internal "which-source-subject" structure)."""
    n_s, d = Xs.shape
    n_t = Xt.shape[0]
    n = n_s + n_t
    X = np.vstack([Xs, Xt])  # (n, d)
    if confident_mask is None:
        confident_mask = np.ones(n_t, dtype=bool)

    idx_s_all = np.arange(n_s)
    w_s_all = source_weights if source_weights is not None else None
    e0 = _mmd_vector(n, idx_s_all, np.arange(n_s, n), weights_pos=w_s_all)
    v0 = X.T @ e0
    XMXt = np.outer(v0, v0)

    for c in classes:
        idx_s = np.where(ys == c)[0]
        idx_t = np.where((pseudo_yt == c) & confident_mask)[0] + n_s
        if len(idx_s) == 0 or len(idx_t) == 0:
            continue
        w_s_c = source_weights[idx_s] if source_weights is not None else None
        ec = _mmd_vector(n, idx_s, idx_t, weights_pos=w_s_c)
        vc = X.T @ ec
        XMXt += np.outer(vc, vc)

    Xsum = X.sum(axis=0)
    XHXt = X.T @ X - (1.0 / n) * np.outer(Xsum, Xsum)

    lhs = XMXt + lam * np.eye(d)
    rhs = XHXt + 1e-6 * np.eye(d)
    eigvals, eigvecs = eigh(lhs, rhs)  # ascending order
    A = eigvecs[:, :k]

    return Xs @ A, Xt @ A


def _balanced_assignment(proba_full, n_classes, quotas):
    """Greedy quota-constrained assignment: sort all (trial, class) probabilities descending,
    assign each trial to its best available class, respecting per-class quotas as they fill up.
    This uses ONLY n_t and n_classes (an equal-repetitions-per-class experimental design is public
    protocol knowledge, not the target subject's actual labels), so it stays zero-shot-legal."""
    n_t = proba_full.shape[0]
    order = np.dstack(np.unravel_index(np.argsort(-proba_full, axis=None), proba_full.shape))[0]
    assigned = -np.ones(n_t, dtype=int)
    remaining = list(quotas)
    n_assigned = 0
    for i, c in order:
        if assigned[i] == -1 and remaining[c] > 0:
            assigned[i] = c
            remaining[c] -= 1
            n_assigned += 1
            if n_assigned == n_t:
                break
    return assigned


def per_subject_fisher_score(X, y, classes):
    """Fisher discriminability ratio = tr(between-class scatter) / tr(within-class scatter) for a
    single subject's own tangent-space trials + own labels (matches how Table tab:feature-stats'
    Fisher ratio is computed, just per-subject instead of pooled)."""
    overall_mean = X.mean(axis=0)
    between, within = 0.0, 0.0
    for c in classes:
        Xc = X[y == c]
        if len(Xc) == 0:
            continue
        class_mean = Xc.mean(axis=0)
        between += len(Xc) * np.sum((class_mean - overall_mean) ** 2)
        within += np.sum((Xc - class_mean) ** 2)
    return between / (within + 1e-8)


def compute_source_subject_weights(Xs, ys, source_subject_ids, classes, source_subjects):
    """Per-source-TRIAL weight array (n_s,): each source subject's Fisher score (computed from
    its OWN tangent-space trials and OWN known labels -- no target information at all) is
    softmax-weighted across the source-subject pool, then broadcast to that subject's trials."""
    fisher_scores = {}
    for s in source_subjects:
        mask = source_subject_ids == s
        fisher_scores[s] = per_subject_fisher_score(Xs[mask], ys[mask], classes)
    scores = np.array([fisher_scores[s] for s in source_subjects])
    temperature = scores.std() + 1e-8
    weights = np.exp(scores / temperature)
    weights /= weights.sum()
    weight_per_subject = dict(zip(source_subjects, weights))
    return np.array([weight_per_subject[s] for s in source_subject_ids]), weight_per_subject


def mekt_lite(Xs, ys, Xt, classes, k=100, lam=1.0, n_iter=5, return_proba=False, confidence_threshold=None,
              balanced_pseudo_labels=False, source_weights=None):
    """confidence_threshold: if set (e.g. 0.5), only target trials whose current-iteration
    predicted class probability exceeds this threshold count toward the per-class MMD terms in
    the NEXT JDA iteration -- low-confidence pseudo-labels (likely wrong, especially early on)
    are excluded from shaping the class-conditional adaptation, though every trial still gets
    re-classified every iteration and all trials contribute to the label-free marginal term.

    balanced_pseudo_labels: if True, pseudo-labels are assigned via a quota-constrained greedy
    matching (each class gets exactly n_t/n_classes trials, the most-confident ones) instead of
    free per-trial argmax. Motivated by a real diagnostic finding: MEKT's raw argmax pseudo-labels
    are meaningfully skewed for some subjects (e.g. one class collapsing to <15% instead of the
    known 25% balanced rate) -- a classic transductive self-training failure mode. Using the
    dataset's known equal-class-count design (public protocol knowledge, not this subject's actual
    labels) as a prior stays zero-shot-legal."""
    n_classes = len(classes)
    n_t = Xt.shape[0]
    base_quota = n_t // n_classes
    quotas = [base_quota + (1 if i < n_t % n_classes else 0) for i in range(n_classes)]

    def assign(proba):
        if balanced_pseudo_labels:
            return _balanced_assignment(proba, n_classes, quotas)
        return proba.argmax(axis=1)

    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(Xs, ys)
    proba0 = np.zeros((n_t, n_classes))
    for i, c in enumerate(clf.classes_):
        proba0[:, c] = clf.predict_proba(Xt)[:, i]
    pseudo_yt = assign(proba0)
    confident_mask = None
    if confidence_threshold is not None:
        confident_mask = proba0.max(axis=1) >= confidence_threshold

    for _ in range(n_iter):
        Zs, Zt = jda_transform(Xs, ys, Xt, pseudo_yt, classes, k, lam, confident_mask=confident_mask,
                                source_weights=source_weights)
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(Zs, ys)
        proba_raw = clf.predict_proba(Zt)
        proba = np.zeros((len(Zt), n_classes))
        for i, c in enumerate(clf.classes_):
            proba[:, c] = proba_raw[:, i]
        pseudo_yt = assign(proba)
        if confidence_threshold is not None:
            confident_mask = proba.max(axis=1) >= confidence_threshold

    if return_proba:
        return pseudo_yt, proba
    return pseudo_yt


def run(subjects=None, k=100, lam=1.0, n_iter=5):
    config.set_seed(config.SEED)
    subjects = subjects or config.EVAL_SUBJECTS
    per_subject, label_encoder = load_bnci_all(subjects, align="ea")
    classes = np.arange(len(label_encoder.classes_))
    cov = Covariances(estimator="lwf")

    fold_results = []
    for held_out in subjects:
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
        print(f"[mekt] held-out subject {held_out}: acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

    accs = np.array([r["accuracy"] for r in fold_results])
    kappas = np.array([r["kappa"] for r in fold_results])
    summary = {"accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
               "kappa_mean": kappas.mean(), "kappa_std": kappas.std()}
    print(f"[mekt] LOSO accuracy: {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}  "
          f"kappa: {summary['kappa_mean']:.4f} +/- {summary['kappa_std']:.4f}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "zero_shot_mekt.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "accuracy", "kappa"])
        writer.writeheader()
        for r in fold_results:
            writer.writerow(r)

    return summary, fold_results


if __name__ == "__main__":
    run()
