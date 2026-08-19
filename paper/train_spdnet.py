"""Stage 1: pretrain SPDNet on PhysioNet. Unlike the Riemannian Transformer, there's no
trial-chunking here -- each trial's covariance matrix is classified independently (same per-trial
protocol as CSP/FgMDM/v1 Riemannian Transformer), since SPDNet's BiMap/ReEig layers have nothing
analogous to cross-trial attention; the novelty here is staying on the SPD manifold, not relating
trials to each other.

    python -m paper.train_spdnet
"""
import os

import numpy as np
import torch
from pyriemann.estimation import Covariances
from torch.utils.data import DataLoader, TensorDataset

from paper import config
from paper.align import align_per_subject
from paper.data import load_physionet_supcon
from paper.models.spdnet import SPDNetModel
from paper.utils import EpochLogger, maybe_save_periodic_and_best


def build_scheduler(optimizer, warmup_epochs, total_epochs):
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs))
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def prepare_covariances(subjects=None):
    """EA per subject -> Covariances(lwf). Returns (covs (M,22,22), labels (M,), subject_ids (M,))."""
    X, y, subject_ids, label_encoder, subject_encoder = load_physionet_supcon(subjects)
    per_subject_X = {s: X[subject_ids == s] for s in np.unique(subject_ids)}
    aligned = align_per_subject(per_subject_X, "ea")

    out_X, out_y, out_ids = [], [], []
    for s in aligned:
        mask = subject_ids == s
        covs = Covariances(estimator="lwf").fit_transform(aligned[s].astype(np.float64)).astype(np.float32)
        out_X.append(covs)
        out_y.append(y[mask])
        out_ids.append(subject_ids[mask])
    return np.concatenate(out_X), np.concatenate(out_y), np.concatenate(out_ids), label_encoder, subject_encoder


def run(subjects=None, epochs: int = None, device: str = None):
    config.set_seed(config.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epochs or config.SPDNET_EPOCHS

    covs, y, subject_ids, label_encoder, subject_encoder = prepare_covariances(subjects)
    print(f"SPDNet pretrain classes: {list(label_encoder.classes_)}, "
          f"n_trials={len(covs)}, n_subjects={len(np.unique(subject_ids))}")

    all_subjects = np.unique(subject_ids)
    n_val_subjects = max(1, int(0.1 * len(all_subjects)))
    rng = np.random.RandomState(config.SEED)
    val_subjects = set(rng.choice(all_subjects, n_val_subjects, replace=False))
    train_mask = ~np.isin(subject_ids, list(val_subjects))
    val_mask = ~train_mask

    n_classes = len(label_encoder.classes_)
    model = SPDNetModel(dims=config.SPDNET_DIMS, n_classes=n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.SPDNET_LR, weight_decay=config.SPDNET_WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, config.SPDNET_WARMUP_EPOCHS, epochs)
    logger = EpochLogger(os.path.join(config.RESULTS_DIR, "pretrain_spdnet.csv"))

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(covs[train_mask]), torch.from_numpy(y[train_mask]).long()),
        batch_size=config.SPDNET_BATCH_SIZE, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(covs[val_mask]), torch.from_numpy(y[val_mask]).long()),
        batch_size=config.SPDNET_BATCH_SIZE, shuffle=False,
    )

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, n_batches = 0.0, 0
        for cov, label in train_loader:
            cov, label = cov.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(cov)
            loss = torch.nn.functional.cross_entropy(logits, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for cov, label in val_loader:
                cov, label = cov.to(device), label.to(device)
                logits = model(cov)
                loss = torch.nn.functional.cross_entropy(logits, label)
                val_loss += loss.item() * len(label)
                val_correct += (logits.argmax(-1) == label).sum().item()
                val_total += len(label)

        train_loss /= max(1, n_batches)
        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        logger.log(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_acc=val_acc)

        best_val_loss = maybe_save_periodic_and_best(
            model, optimizer, epoch, val_loss, best_val_loss,
            config.CHECKPOINT_DIR, "pretrain_spdnet", config.SPDNET_CHECKPOINT_EVERY,
        )

    return model, subject_encoder


if __name__ == "__main__":
    run()
