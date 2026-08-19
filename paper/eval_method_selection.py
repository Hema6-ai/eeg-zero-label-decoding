"""Unsupervised per-target method selection: for each held-out subject, find the geometrically
closest OTHER subject (Riemannian distance between EA-aligned mean covariances -- unsupervised,
no labels), look up which method scored best when THAT subject was itself the LOSO target, and use
that method's (already-computed) accuracy for the actual target. Zero label leakage: the target's
own labels are never used for selection, only its unlabeled mean covariance.

Candidate methods: FgMDM, MEKT-lite (k=50, lam=1.0, 20 iterations -- the tuned best from the grid
search), SPDNet, Riemannian Transformer v2. CSP+LDA and TS+LR+EA are NOT included: this project
only ever saved aggregate mean/std for those, not per-subject accuracy, and fabricating per-fold
numbers for them would undermine the whole point of this experiment.

Also reports the oracle upper bound: best method per subject using that subject's OWN true
accuracy (impossible to know without labels -- a cherry-picking ceiling, not a deployable result).

    python -m paper.eval_method_selection
"""
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.geometry.distance import distance_riemann
from pyriemann.geometry.mean import mean_riemann

from paper import config
from paper.data import load_bnci_all
from paper.eval_ensemble import rt_v2_probs
from paper.eval_mekt import mekt_lite
from paper.eval_stacked_ensemble import _fgmdm_probs
from paper.eval_stacked_ensemble_3way import _spdnet_probs
from paper.eval_zero_shot_riemannian_transformer_v2 import load_frozen_encoder as load_rt_encoder
from paper.eval_zero_shot_spdnet import load_frozen_encoder as load_spdnet_encoder
from paper.utils import classification_metrics

METHOD_NAMES = ["fgmdm", "mekt", "spdnet", "rt_v2"]
MEKT_BEST = dict(k=50, lam=1.0, n_iter=20)  # tuned via the 36-config grid search


def _mekt_probs_best(source_X, source_y, target_X, classes):
    cov = Covariances(estimator="lwf")
    source_covs = cov.fit_transform(source_X)
    target_covs = cov.transform(target_X)
    from pyriemann.tangentspace import TangentSpace
    ts = TangentSpace(metric="riemann")
    Xs = ts.fit_transform(source_covs)
    Xt = ts.transform(target_covs)
    k_eff = min(MEKT_BEST["k"], Xs.shape[1])
    _, probs = mekt_lite(Xs, source_y, Xt, classes, k=k_eff, lam=MEKT_BEST["lam"],
                          n_iter=MEKT_BEST["n_iter"], return_proba=True)
    return probs


def compute_all_method_results(subjects, device="cpu"):
    """For every subject as LOSO target, get every method's accuracy/kappa. This IS each method's
    genuine per-fold result (no shortcuts) -- needed both to build the closest-subject lookup
    table and to report the oracle ceiling."""
    per_subject, label_encoder = load_bnci_all(subjects, align="ea")
    classes = np.arange(len(label_encoder.classes_))
    rt_encoder = load_rt_encoder(device)
    spdnet_encoder = load_spdnet_encoder(device)

    results = {}
    for held_out in subjects:
        source_subjects = [s for s in subjects if s != held_out]
        source_X = np.concatenate([per_subject[s][0] for s in source_subjects], axis=0).astype(np.float64)
        source_y = np.concatenate([per_subject[s][1] for s in source_subjects], axis=0)
        target_X, target_y = per_subject[held_out]
        target_X = target_X.astype(np.float64)

        p_fgmdm = _fgmdm_probs(source_X, source_y, target_X, classes)
        p_mekt = _mekt_probs_best(source_X, source_y, target_X, classes)
        p_spd = _spdnet_probs(source_X, source_y, target_X, classes, spdnet_encoder, device)
        p_rt = rt_v2_probs(source_X, source_y, target_X, len(classes), rt_encoder, device)

        results[held_out] = {}
        for name, probs in zip(METHOD_NAMES, [p_fgmdm, p_mekt, p_spd, p_rt]):
            pred = probs.argmax(axis=1)
            m = classification_metrics(target_y, pred)
            results[held_out][name] = {"accuracy": m["accuracy"], "kappa": m["kappa"]}
        print(f"[method-selection] computed subject {held_out}: " +
              " ".join(f"{n}={results[held_out][n]['accuracy']:.4f}" for n in METHOD_NAMES))

    return results, per_subject


def run(subjects=None, device: str = "cpu"):
    config.set_seed(config.SEED)
    subjects = subjects or config.EVAL_SUBJECTS

    print("Computing every method's genuine per-fold result (needed for the lookup table)...")
    results, per_subject = compute_all_method_results(subjects, device)

    cov = Covariances(estimator="lwf")
    mean_covs = {s: mean_riemann(cov.fit_transform(per_subject[s][0].astype(np.float64))) for s in subjects}

    selection_results, oracle_results = [], []
    for target in subjects:
        candidates = [s for s in subjects if s != target]
        dists = {s: distance_riemann(mean_covs[target], mean_covs[s]) for s in candidates}
        closest = min(dists, key=dists.get)

        best_method_for_closest = max(METHOD_NAMES, key=lambda m: results[closest][m]["accuracy"])
        chosen_acc = results[target][best_method_for_closest]["accuracy"]
        chosen_kappa = results[target][best_method_for_closest]["kappa"]
        selection_results.append({"subject": target, "closest_source": closest, "distance": dists[closest],
                                   "chosen_method": best_method_for_closest,
                                   "accuracy": chosen_acc, "kappa": chosen_kappa})

        oracle_method = max(METHOD_NAMES, key=lambda m: results[target][m]["accuracy"])
        oracle_results.append({"subject": target, "method": oracle_method,
                                "accuracy": results[target][oracle_method]["accuracy"],
                                "kappa": results[target][oracle_method]["kappa"]})

        print(f"subject {target}: closest={closest} (d={dists[closest]:.4f}, best-for-closest={best_method_for_closest}) "
              f"-> selected={best_method_for_closest} acc={chosen_acc:.4f}   "
              f"[oracle: {oracle_method} acc={oracle_results[-1]['accuracy']:.4f}]")

    sel_accs = np.array([r["accuracy"] for r in selection_results])
    sel_kappas = np.array([r["kappa"] for r in selection_results])
    oracle_accs = np.array([r["accuracy"] for r in oracle_results])
    oracle_kappas = np.array([r["kappa"] for r in oracle_results])

    print(f"\nUnsupervised selection: {sel_accs.mean()*100:.2f}% +/- {sel_accs.std()*100:.2f}%  "
          f"kappa={sel_kappas.mean():.4f} +/- {sel_kappas.std():.4f}")
    print(f"Oracle upper bound:     {oracle_accs.mean()*100:.2f}% +/- {oracle_accs.std()*100:.2f}%  "
          f"kappa={oracle_kappas.mean():.4f} +/- {oracle_kappas.std():.4f}")
    print(f"\nBeat FgMDM (47.45%)? {'YES' if sel_accs.mean() > 0.4745 else 'NO'}")
    print(f"Beat MEKT-tuned (48.48%)? {'YES' if sel_accs.mean() > 0.4848 else 'NO'}")
    print(f"Crossed 50%? {'YES' if sel_accs.mean() > 0.50 else 'NO'}")
    print(f"Crossed 55%? {'YES' if sel_accs.mean() > 0.55 else 'NO'}")

    return selection_results, oracle_results, results


if __name__ == "__main__":
    run()
