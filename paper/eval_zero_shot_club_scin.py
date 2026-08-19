"""Zero-shot LOSO evaluation for the MAE+CLUB+SCIN checkpoint on BCI-IV 2a.

Mirrors eval_zero_shot.py but: (1) uses EEGMAEClubScin.encode_zero_shot (SCIN mean-fallback,
since BNCI subjects were never seen during PhysioNet pretraining), (2) uses a 2-layer MLP head
instead of a linear probe, trained for more epochs, and (3) ensembles several independently
initialized heads per fold and averages their softmax outputs before taking the final class.

    python -m paper.eval_zero_shot_club_scin
"""
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from paper import config
from paper.data import load_bnci_all
from paper.models.mae_club import EEGMAEClubScin
from paper.models.heads import MLPProbe
from paper.utils import classification_metrics


def load_frozen_model(n_channels, n_times, n_subjects, device):
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "pretrain_mae_club_scin_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = EEGMAEClubScin(
        n_channels=n_channels, n_times=n_times, n_subjects=n_subjects,
        patch_size=config.PATCH_SIZE, embed_dim=config.EMBED_DIM,
        encoder_depth=config.ENCODER_DEPTH, encoder_heads=config.ENCODER_HEADS,
        decoder_dim=config.DECODER_DIM, decoder_depth=config.DECODER_DEPTH,
        decoder_heads=config.ENCODER_HEADS, mask_ratio=config.MASK_RATIO,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def extract_features(model, X, device, batch_size=256):
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).to(device)
            feats.append(model.encode_zero_shot(batch).cpu())
    return torch.cat(feats, dim=0)


def train_one_head(feats, labels, n_classes, device, seed):
    torch.manual_seed(seed)
    head = MLPProbe(config.EMBED_DIM, n_classes, config.ZEROSHOT_MLP_HIDDEN,
                     config.ZEROSHOT_MLP_DROPOUT).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=config.PROBE_LR)
    loader = DataLoader(TensorDataset(feats, labels), batch_size=64, shuffle=True)
    head.train()
    for _ in range(config.ZEROSHOT_MLP_EPOCHS):
        for feat, label in loader:
            feat, label = feat.to(device), label.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(head(feat), label)
            loss.backward()
            optimizer.step()
    return head


def run(subjects=None, device: str = None, n_pretrain_subjects: int = None):
    config.set_seed(config.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    subjects = subjects or config.EVAL_SUBJECTS

    per_subject, label_encoder = load_bnci_all(subjects)
    n_channels, n_times = next(iter(per_subject.values()))[0].shape[1:]
    n_classes = len(label_encoder.classes_)

    # n_subjects here is only used to size SCIN/CLUB's embedding tables inside the checkpoint;
    # the actual value must match how many PhysioNet subjects the checkpoint was pretrained on.
    n_pretrain_subjects = n_pretrain_subjects or len(config.PRETRAIN_SUBJECTS)
    model = load_frozen_model(n_channels, n_times, n_pretrain_subjects, device)

    fold_results = []
    for held_out in subjects:
        source_subjects = [s for s in subjects if s != held_out]

        source_X = np.concatenate([per_subject[s][0] for s in source_subjects], axis=0)
        source_y = np.concatenate([per_subject[s][1] for s in source_subjects], axis=0)
        target_X, target_y = per_subject[held_out]

        source_feats = extract_features(model, source_X, device)
        target_feats = extract_features(model, target_X, device)
        source_labels = torch.from_numpy(source_y).long()

        probs_sum = None
        for i in range(config.ZEROSHOT_ENSEMBLE_SIZE):
            head = train_one_head(source_feats, source_labels, n_classes, device,
                                   seed=config.SEED + i)
            head.eval()
            with torch.no_grad():
                probs = F.softmax(head(target_feats.to(device)), dim=-1).cpu()
            probs_sum = probs if probs_sum is None else probs_sum + probs

        pred = (probs_sum / config.ZEROSHOT_ENSEMBLE_SIZE).argmax(-1).numpy()
        metrics = classification_metrics(target_y, pred)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        print(f"[mae_club_scin] held-out subject {held_out}: "
              f"acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

    accs = np.array([r["accuracy"] for r in fold_results])
    kappas = np.array([r["kappa"] for r in fold_results])
    summary = {
        "variant": "mae_club_scin",
        "accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
        "kappa_mean": kappas.mean(), "kappa_std": kappas.std(),
    }
    print(f"[mae_club_scin] LOSO accuracy: {summary['accuracy_mean']:.4f} +/- "
          f"{summary['accuracy_std']:.4f}  kappa: {summary['kappa_mean']:.4f} +/- "
          f"{summary['kappa_std']:.4f}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "zero_shot_mae_club_scin.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "accuracy", "kappa"])
        writer.writeheader()
        for r in fold_results:
            writer.writerow(r)

    return summary, fold_results


if __name__ == "__main__":
    run()
