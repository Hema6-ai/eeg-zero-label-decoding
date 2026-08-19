"""Stage 2: zero-shot LOSO evaluation of the frozen Riemannian Transformer encoder on BCI-IV 2a.

Per fold: raw EEG -> EA (per-subject) -> Covariances(lwf) -> TangentSpace fit on SOURCE subjects'
covariances only (never touches the held-out subject's data) -> frozen encoder -> CLS embedding
-> fresh MLP head trained on source embeddings -> evaluate on held-out subject.

    python -m paper.eval_zero_shot_riemannian_transformer
"""
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from torch.utils.data import DataLoader, TensorDataset

from paper import config
from paper.align import align_per_subject
from paper.data import load_bnci_all
from paper.models.heads import MLPProbe
from paper.models.riemannian_transformer import TangentSpaceTransformer
from paper.utils import classification_metrics


def load_frozen_encoder(device):
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "pretrain_riemannian_transformer_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder = TangentSpaceTransformer(
        tangent_dim=config.RT_TANGENT_DIM, embed_dim=config.RT_EMBED_DIM,
        depth=config.RT_DEPTH, heads=config.RT_HEADS,
    ).to(device)
    encoder_state = {k[len("encoder."):]: v for k, v in ckpt["model"].items() if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def train_mlp_head(feats, labels, n_classes, device):
    head = MLPProbe(config.RT_EMBED_DIM, n_classes, config.RT_HEAD_HIDDEN, config.RT_HEAD_DROPOUT).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=config.PROBE_LR)
    loader = DataLoader(TensorDataset(feats, labels), batch_size=64, shuffle=True)
    head.train()
    for _ in range(config.RT_HEAD_EPOCHS):
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

    per_subject, label_encoder = load_bnci_all(subjects)
    n_classes = len(label_encoder.classes_)
    encoder = load_frozen_encoder(device)

    aligned = {s: (align_per_subject({s: per_subject[s][0]}, "ea")[s].astype(np.float64), per_subject[s][1])
               for s in subjects}

    fold_results = []
    for held_out in subjects:
        source_subjects = [s for s in subjects if s != held_out]
        source_X = np.concatenate([aligned[s][0] for s in source_subjects], axis=0)
        source_y = np.concatenate([aligned[s][1] for s in source_subjects], axis=0)
        target_X, target_y = aligned[held_out]

        cov = Covariances(estimator="lwf")
        source_covs = cov.fit_transform(source_X)
        target_covs = cov.transform(target_X)

        # TangentSpace is fit on the SOURCE pool only -- the held-out subject's covariances are
        # only ever transformed through it, never used to compute the reference point.
        ts = TangentSpace(metric="riemann")
        source_vecs = ts.fit_transform(source_covs).astype(np.float32)
        target_vecs = ts.transform(target_covs).astype(np.float32)

        with torch.no_grad():
            source_feats = encoder.cls_representation(torch.from_numpy(source_vecs).to(device)).cpu()
            target_feats = encoder.cls_representation(torch.from_numpy(target_vecs).to(device)).cpu()

        head = train_mlp_head(source_feats, torch.from_numpy(source_y).long(), n_classes, device)
        head.eval()
        with torch.no_grad():
            pred = head(target_feats.to(device)).argmax(-1).cpu().numpy()

        metrics = classification_metrics(target_y, pred)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        print(f"[riemannian_transformer] held-out subject {held_out}: "
              f"acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

    accs = np.array([r["accuracy"] for r in fold_results])
    kappas = np.array([r["kappa"] for r in fold_results])
    summary = {"variant": "riemannian_transformer", "accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
               "kappa_mean": kappas.mean(), "kappa_std": kappas.std()}
    print(f"[riemannian_transformer] LOSO accuracy: {summary['accuracy_mean']:.4f} +/- "
          f"{summary['accuracy_std']:.4f}  kappa: {summary['kappa_mean']:.4f} +/- {summary['kappa_std']:.4f}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "zero_shot_riemannian_transformer.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "accuracy", "kappa"])
        writer.writeheader()
        for r in fold_results:
            writer.writerow(r)

    return summary, fold_results


if __name__ == "__main__":
    run()
