"""Variant of the v2 zero-shot eval that weights each SOURCE subject's contribution to the MLP
head's training loss by how close its covariance geometry is to the held-out TARGET subject's
(unsupervised Riemannian distance between per-subject mean covariances -- no labels involved, so
this stays zero-shot-legal). Motivation: subject 2 has never crossed 30% accuracy under any
method tried in this project; pooling all 8 source subjects equally may be diluting whatever
source subjects are actually close to subject 2's covariance geometry with subjects that are far
from it. This is an eval-time-only change on top of the same frozen v2 encoder checkpoint --
no retraining required, so it's cheap to test (CPU, seconds) as soon as the checkpoint exists.

    python -m paper.eval_zero_shot_riemannian_transformer_v2_weighted
"""
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_riemann
from torch.utils.data import DataLoader, TensorDataset

from paper import config
from paper.data import load_bnci_all
from paper.eval_zero_shot_riemannian_transformer_v2 import embed_all_trials, load_frozen_encoder
from paper.models.heads import MLPProbe
from paper.utils import classification_metrics


def compute_subject_weights(source_subjects, source_covs_by_subject, target_covs):
    """Per-source-subject weight = softmax(-Riemannian distance to target), temperature = std of
    distances. Both means are unsupervised (no labels), so this is valid at zero-shot eval time."""
    target_mean = mean_riemann(target_covs)
    dists = np.array([
        distance_riemann(mean_riemann(source_covs_by_subject[s]), target_mean)
        for s in source_subjects
    ])
    temperature = dists.std() + 1e-8
    weights = np.exp(-dists / temperature)
    weights /= weights.sum()
    return dict(zip(source_subjects, weights)), dists


def train_mlp_head_weighted(feats, labels, sample_weights, n_classes, device):
    head = MLPProbe(config.RT_EMBED_DIM, n_classes, config.RT_HEAD_HIDDEN, config.RT_HEAD_DROPOUT).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=config.PROBE_LR)
    loader = DataLoader(TensorDataset(feats, labels, sample_weights), batch_size=64, shuffle=True)
    head.train()
    for _ in range(config.RT_HEAD_EPOCHS):
        for feat, label, w in loader:
            feat, label, w = feat.to(device), label.to(device), w.to(device)
            optimizer.zero_grad()
            per_sample_loss = F.cross_entropy(head(feat), label, reduction="none")
            loss = (per_sample_loss * w).sum() / w.sum()
            loss.backward()
            optimizer.step()
    return head


def run(subjects=None, device: str = None):
    config.set_seed(config.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    subjects = subjects or config.EVAL_SUBJECTS

    per_subject, label_encoder = load_bnci_all(subjects, align="ea")
    n_classes = len(label_encoder.classes_)
    encoder = load_frozen_encoder(device)
    n_trials = config.RTV2_N_TRIALS

    fold_results = []
    for held_out in subjects:
        source_subjects = [s for s in subjects if s != held_out]
        target_X, target_y = per_subject[held_out]
        target_X = target_X.astype(np.float64)

        cov = Covariances(estimator="lwf")
        source_covs_by_subject = {s: cov.fit_transform(per_subject[s][0].astype(np.float64)) for s in source_subjects}
        target_covs = cov.transform(target_X)

        subject_weights, dists = compute_subject_weights(source_subjects, source_covs_by_subject, target_covs)

        source_X = np.concatenate([per_subject[s][0] for s in source_subjects], axis=0).astype(np.float64)
        source_y = np.concatenate([per_subject[s][1] for s in source_subjects], axis=0)
        source_subject_ids = np.concatenate([np.full(len(per_subject[s][0]), s) for s in source_subjects])
        source_sample_weights = np.array([subject_weights[s] for s in source_subject_ids], dtype=np.float32)

        source_covs = cov.fit_transform(source_X)
        ts = TangentSpace(metric="riemann")
        source_vecs = ts.fit_transform(source_covs).astype(np.float32)
        target_vecs = ts.transform(target_covs).astype(np.float32)

        source_feats = torch.from_numpy(embed_all_trials(encoder, source_vecs, n_trials, device))
        target_feats = torch.from_numpy(embed_all_trials(encoder, target_vecs, n_trials, device))

        head = train_mlp_head_weighted(
            source_feats, torch.from_numpy(source_y).long(),
            torch.from_numpy(source_sample_weights), n_classes, device,
        )
        head.eval()
        with torch.no_grad():
            pred = head(target_feats.to(device)).argmax(-1).cpu().numpy()

        metrics = classification_metrics(target_y, pred)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        weight_str = ", ".join(f"s{s}:{subject_weights[s]:.3f}" for s in source_subjects)
        print(f"[riemannian_transformer_v2_weighted] held-out subject {held_out}: "
              f"acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}  source_weights=[{weight_str}]")

    accs = np.array([r["accuracy"] for r in fold_results])
    kappas = np.array([r["kappa"] for r in fold_results])
    summary = {"variant": "riemannian_transformer_v2_weighted", "accuracy_mean": accs.mean(),
               "accuracy_std": accs.std(), "kappa_mean": kappas.mean(), "kappa_std": kappas.std()}
    print(f"[riemannian_transformer_v2_weighted] LOSO accuracy: {summary['accuracy_mean']:.4f} +/- "
          f"{summary['accuracy_std']:.4f}  kappa: {summary['kappa_mean']:.4f} +/- {summary['kappa_std']:.4f}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "zero_shot_riemannian_transformer_v2_weighted.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "accuracy", "kappa"])
        writer.writeheader()
        for r in fold_results:
            writer.writerow(r)

    return summary, fold_results


if __name__ == "__main__":
    run()
