"""Zero-shot LOSO evaluation for the SupCon-pretrained encoder on BCI-IV 2a.

    python -m paper.eval_zero_shot_supcon
"""
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from paper import config
from paper.data import load_bnci_all
from paper.models.encoder import EEGTransformerEncoder
from paper.models.heads import MLPProbe
from paper.utils import classification_metrics


def load_frozen_supcon_encoder(n_channels: int, n_times: int, device: str):
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "pretrain_supcon_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    encoder = EEGTransformerEncoder(
        n_channels=n_channels, n_times=n_times, patch_size=config.PATCH_SIZE,
        embed_dim=config.EMBED_DIM, depth=config.ENCODER_DEPTH, heads=config.ENCODER_HEADS,
    ).to(device)
    # Checkpoint is SupConModel's state dict (encoder + discarded projector); only the encoder
    # submodule is ever used downstream.
    encoder_state = {k[len("encoder."):]: v for k, v in ckpt["model"].items() if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def extract_cls_features(encoder, X, device, batch_size=256):
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).to(device)
            feats.append(encoder.cls_representation(batch).cpu())
    return torch.cat(feats, dim=0)


def train_mlp_head(feats, labels, n_classes, device):
    head = MLPProbe(config.EMBED_DIM, n_classes, config.SUPCON_MLP_HIDDEN, config.SUPCON_MLP_DROPOUT).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=config.PROBE_LR)
    loader = DataLoader(TensorDataset(feats, labels), batch_size=64, shuffle=True)
    head.train()
    for _ in range(config.SUPCON_HEAD_EPOCHS):
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
    n_channels, n_times = next(iter(per_subject.values()))[0].shape[1:]
    n_classes = len(label_encoder.classes_)
    encoder = load_frozen_supcon_encoder(n_channels, n_times, device)

    fold_results = []
    all_true, all_pred = [], []
    for held_out in subjects:
        source_subjects = [s for s in subjects if s != held_out]
        source_X = np.concatenate([per_subject[s][0] for s in source_subjects], axis=0)
        source_y = np.concatenate([per_subject[s][1] for s in source_subjects], axis=0)
        target_X, target_y = per_subject[held_out]

        source_feats = extract_cls_features(encoder, source_X, device)
        target_feats = extract_cls_features(encoder, target_X, device)

        head = train_mlp_head(source_feats, torch.from_numpy(source_y).long(), n_classes, device)
        head.eval()
        with torch.no_grad():
            pred = head(target_feats.to(device)).argmax(-1).cpu().numpy()
        metrics = classification_metrics(target_y, pred)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        all_true.append(target_y)
        all_pred.append(pred)
        print(f"[supcon] held-out subject {held_out}: acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

    accs = np.array([r["accuracy"] for r in fold_results])
    kappas = np.array([r["kappa"] for r in fold_results])
    summary = {"variant": "supcon", "accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
               "kappa_mean": kappas.mean(), "kappa_std": kappas.std()}
    print(f"[supcon] LOSO accuracy: {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}  "
          f"kappa: {summary['kappa_mean']:.4f} +/- {summary['kappa_std']:.4f}")

    # Per-class accuracy pooled across all folds (PhysioNet has no 'tongue' class, so this
    # checks whether that specific class underperforms at zero-shot eval time).
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    per_class = {}
    for i, name in enumerate(label_encoder.classes_):
        mask = all_true == i
        per_class[name] = (all_pred[mask] == all_true[mask]).mean() if mask.sum() > 0 else float("nan")
        print(f"[supcon] per-class accuracy -- {name}: {per_class[name]:.4f} (n={mask.sum()})")
    summary["per_class"] = per_class

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "zero_shot_supcon.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "accuracy", "kappa"])
        writer.writeheader()
        for r in fold_results:
            writer.writerow({k: r[k] for k in ["held_out_subject", "accuracy", "kappa"]})

    with open(os.path.join(config.RESULTS_DIR, "zero_shot_supcon_per_class.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "accuracy", "n"])
        for i, name in enumerate(label_encoder.classes_):
            mask = all_true == i
            writer.writerow([name, per_class[name], int(mask.sum())])

    return summary, fold_results


if __name__ == "__main__":
    run()
