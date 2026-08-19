"""Soft-voting ensemble across the three most mechanistically different zero-shot methods built
in this project: FgMDM (classical, non-learned, our best single method at 47.45%), Riemannian
Transformer v2 (tangent-space features + learned trial-attention), and SPDNet (stays on the SPD
manifold end-to-end, never flattens to tangent space until the final layer). Ensembling across
genuinely different modeling approaches is the standard lever for gains when individual models'
errors are only partially correlated -- untried in this project until now, and cheap (CPU-only,
no new training beyond what already exists as frozen checkpoints).

    python -m paper.eval_ensemble
"""
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from pyriemann.classification import FgMDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from torch.utils.data import DataLoader, TensorDataset

from paper import config
from paper.data import load_bnci_all
from paper.eval_zero_shot_riemannian_transformer_v2 import embed_all_trials
from paper.eval_zero_shot_riemannian_transformer_v2 import load_frozen_encoder as load_rt_encoder
from paper.eval_zero_shot_spdnet import embed_covs
from paper.eval_zero_shot_spdnet import load_frozen_encoder as load_spdnet_encoder
from paper.models.heads import MLPProbe
from paper.utils import classification_metrics


def train_mlp_head_probs(feats, labels, n_classes, hidden, dropout, epochs, device):
    head = MLPProbe(feats.shape[1], n_classes, hidden, dropout).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=config.PROBE_LR)
    loader = DataLoader(TensorDataset(feats, labels), batch_size=64, shuffle=True)
    head.train()
    for _ in range(epochs):
        for feat, label in loader:
            feat, label = feat.to(device), label.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(head(feat), label)
            loss.backward()
            optimizer.step()
    return head


def fgmdm_probs(source_X, source_y, target_X, classes):
    cov = Covariances(estimator="lwf")
    source_covs = cov.fit_transform(source_X.astype(np.float64))
    target_covs = cov.transform(target_X.astype(np.float64))
    clf = FgMDM(metric="riemann", tsupdate=False)
    clf.fit(source_covs, source_y)
    probs = clf.predict_proba(target_covs)
    # FgMDM's class order is clf.classes_ -- reindex to the canonical 0..n_classes-1 order.
    out = np.zeros((len(target_X), len(classes)))
    for i, c in enumerate(clf.classes_):
        out[:, c] = probs[:, i]
    return out


def rt_v2_probs(source_X, source_y, target_X, n_classes, encoder, device):
    cov = Covariances(estimator="lwf")
    source_covs = cov.fit_transform(source_X.astype(np.float64))
    target_covs = cov.transform(target_X.astype(np.float64))
    ts = TangentSpace(metric="riemann")
    source_vecs = ts.fit_transform(source_covs).astype(np.float32)
    target_vecs = ts.transform(target_covs).astype(np.float32)

    source_feats = torch.from_numpy(embed_all_trials(encoder, source_vecs, config.RTV2_N_TRIALS, device))
    target_feats = torch.from_numpy(embed_all_trials(encoder, target_vecs, config.RTV2_N_TRIALS, device))

    head = train_mlp_head_probs(source_feats, torch.from_numpy(source_y).long(), n_classes,
                                 config.RT_HEAD_HIDDEN, config.RT_HEAD_DROPOUT, config.RT_HEAD_EPOCHS, device)
    head.eval()
    with torch.no_grad():
        probs = F.softmax(head(target_feats.to(device)), dim=-1).cpu().numpy()
    return probs


def spdnet_probs(source_X, source_y, target_X, n_classes, encoder, device):
    cov = Covariances(estimator="lwf")
    source_covs = cov.fit_transform(source_X.astype(np.float64)).astype(np.float32)
    target_covs = cov.transform(target_X.astype(np.float64)).astype(np.float32)

    source_feats = embed_covs(encoder, source_covs, device)
    target_feats = embed_covs(encoder, target_covs, device)

    head = train_mlp_head_probs(source_feats, torch.from_numpy(source_y).long(), n_classes,
                                 config.SPDNET_HEAD_HIDDEN, config.SPDNET_HEAD_DROPOUT,
                                 config.SPDNET_HEAD_EPOCHS, device)
    head.eval()
    with torch.no_grad():
        probs = F.softmax(head(target_feats.to(device)), dim=-1).cpu().numpy()
    return probs


def run(subjects=None, device: str = None, use_spdnet: bool = True, weights: dict = None):
    """weights: optional {"fgmdm": w, "rt_v2": w, "spdnet": w} to weight the soft-vote instead of
    averaging equally (equal-weighting a strong and a weak method just drags the strong one down)."""
    config.set_seed(config.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    subjects = subjects or config.EVAL_SUBJECTS
    weights = weights or {"fgmdm": 1.0, "rt_v2": 1.0, "spdnet": 1.0}

    per_subject, label_encoder = load_bnci_all(subjects, align="ea")
    classes = np.arange(len(label_encoder.classes_))
    n_classes = len(classes)

    rt_encoder = load_rt_encoder(device)
    spdnet_encoder = load_spdnet_encoder(device) if use_spdnet else None

    fold_results = {"fgmdm": [], "rt_v2": [], "spdnet": [], "ensemble": []}
    for held_out in subjects:
        source_subjects = [s for s in subjects if s != held_out]
        source_X = np.concatenate([per_subject[s][0] for s in source_subjects], axis=0)
        source_y = np.concatenate([per_subject[s][1] for s in source_subjects], axis=0)
        target_X, target_y = per_subject[held_out]

        p_fgmdm = fgmdm_probs(source_X, source_y, target_X, classes)
        p_rt = rt_v2_probs(source_X, source_y, target_X, n_classes, rt_encoder, device)
        weighted_sum = weights["fgmdm"] * p_fgmdm + weights["rt_v2"] * p_rt
        total_weight = weights["fgmdm"] + weights["rt_v2"]
        if use_spdnet:
            p_spd = spdnet_probs(source_X, source_y, target_X, n_classes, spdnet_encoder, device)
            weighted_sum = weighted_sum + weights["spdnet"] * p_spd
            total_weight += weights["spdnet"]

        ensemble_probs = weighted_sum / total_weight
        pred_ensemble = ensemble_probs.argmax(axis=1)
        pred_fgmdm = p_fgmdm.argmax(axis=1)
        pred_rt = p_rt.argmax(axis=1)

        for key, pred in [("fgmdm", pred_fgmdm), ("rt_v2", pred_rt), ("ensemble", pred_ensemble)]:
            m = classification_metrics(target_y, pred)
            m["held_out_subject"] = held_out
            fold_results[key].append(m)
        if use_spdnet:
            m = classification_metrics(target_y, p_spd.argmax(axis=1))
            m["held_out_subject"] = held_out
            fold_results["spdnet"].append(m)

        print(f"[ensemble] held-out subject {held_out}: "
              f"fgmdm={fold_results['fgmdm'][-1]['accuracy']:.4f} "
              f"rt_v2={fold_results['rt_v2'][-1]['accuracy']:.4f} "
              + (f"spdnet={fold_results['spdnet'][-1]['accuracy']:.4f} " if use_spdnet else "")
              + f"ENSEMBLE={fold_results['ensemble'][-1]['accuracy']:.4f}")

    summaries = {}
    for key, results in fold_results.items():
        if not results:
            continue
        accs = np.array([r["accuracy"] for r in results])
        kappas = np.array([r["kappa"] for r in results])
        summaries[key] = {"accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
                           "kappa_mean": kappas.mean(), "kappa_std": kappas.std()}
        print(f"[{key}] LOSO accuracy: {summaries[key]['accuracy_mean']:.4f} +/- "
              f"{summaries[key]['accuracy_std']:.4f}  kappa: {summaries[key]['kappa_mean']:.4f} +/- "
              f"{summaries[key]['kappa_std']:.4f}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "zero_shot_ensemble.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "accuracy", "kappa"])
        writer.writeheader()
        for r in fold_results["ensemble"]:
            writer.writerow(r)

    return summaries, fold_results


if __name__ == "__main__":
    run()
