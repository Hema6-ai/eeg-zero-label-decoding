"""Stage 2: zero-shot LOSO eval of the frozen SPDNet encoder. EA per subject -> Covariances(lwf)
-> frozen BiMap/ReEig/LogEig encoder (no held-out-subject data or labels touch the encoder,
matching every other zero-shot eval in this project) -> MLP head trained on source, evaluated on
target.

    python -m paper.eval_zero_shot_spdnet
"""
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from pyriemann.estimation import Covariances
from torch.utils.data import DataLoader, TensorDataset

from paper import config
from paper.data import load_bnci_all
from paper.models.heads import MLPProbe
from paper.models.spdnet import SPDNetEncoder
from paper.utils import classification_metrics


def load_frozen_encoder(device):
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "pretrain_spdnet_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder = SPDNetEncoder(dims=config.SPDNET_DIMS).to(device)
    encoder_state = {k[len("encoder."):]: v for k, v in ckpt["model"].items() if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def embed_covs(encoder, covs, device, batch_size=128):
    embeds = []
    with torch.no_grad():
        for i in range(0, len(covs), batch_size):
            batch = torch.from_numpy(covs[i:i + batch_size]).to(device)
            embeds.append(encoder(batch).cpu())
    return torch.cat(embeds, dim=0)


def train_mlp_head(feats, labels, n_classes, device):
    head = MLPProbe(feats.shape[1], n_classes, config.SPDNET_HEAD_HIDDEN, config.SPDNET_HEAD_DROPOUT).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=config.PROBE_LR)
    loader = DataLoader(TensorDataset(feats, labels), batch_size=64, shuffle=True)
    head.train()
    for _ in range(config.SPDNET_HEAD_EPOCHS):
        for feat, label in loader:
            feat, label = feat.to(device), label.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(head(feat), label)
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
    cov = Covariances(estimator="lwf")

    fold_results = []
    for held_out in subjects:
        source_subjects = [s for s in subjects if s != held_out]
        source_X = np.concatenate([per_subject[s][0] for s in source_subjects], axis=0).astype(np.float64)
        source_y = np.concatenate([per_subject[s][1] for s in source_subjects], axis=0)
        target_X, target_y = per_subject[held_out]
        target_X = target_X.astype(np.float64)

        source_covs = cov.fit_transform(source_X).astype(np.float32)
        target_covs = cov.transform(target_X).astype(np.float32)

        source_feats = embed_covs(encoder, source_covs, device)
        target_feats = embed_covs(encoder, target_covs, device)

        head = train_mlp_head(source_feats, torch.from_numpy(source_y).long(), n_classes, device)
        head.eval()
        with torch.no_grad():
            pred = head(target_feats.to(device)).argmax(-1).cpu().numpy()

        metrics = classification_metrics(target_y, pred)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        print(f"[spdnet] held-out subject {held_out}: acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

    accs = np.array([r["accuracy"] for r in fold_results])
    kappas = np.array([r["kappa"] for r in fold_results])
    summary = {"variant": "spdnet", "accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
               "kappa_mean": kappas.mean(), "kappa_std": kappas.std()}
    print(f"[spdnet] LOSO accuracy: {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}  "
          f"kappa: {summary['kappa_mean']:.4f} +/- {summary['kappa_std']:.4f}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "zero_shot_spdnet.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "accuracy", "kappa"])
        writer.writeheader()
        for r in fold_results:
            writer.writerow(r)

    return summary, fold_results


if __name__ == "__main__":
    run()
