"""Supervised Contrastive (SupCon) pretraining on PhysioNet, replacing MAE reconstruction.

No decoder, no masking -- the encoder sees the full trial. SupCon loss is computed on a
projection head's output (discarded after pretraining), not on the backbone CLS token directly --
see models/projection_head.py for why: optimizing the raw backbone representation against a
sharp-temperature contrastive loss collapsed to a degenerate constant embedding in an earlier run
(confirmed via loss frozen at ln(2B-1) and >0.998 pairwise cosine similarity across arbitrary
inputs). Zero-shot eval and the subject probe use the raw backbone CLS token, never the projector.

    python -m paper.train_supcon
"""
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from paper import config
from paper.augment_supcon import make_two_views
from paper.data import load_physionet_supcon
from paper.models.encoder import EEGTransformerEncoder
from paper.models.projection_head import ProjectionHead
from paper.models.supcon_loss import supcon_loss
from paper.utils import EpochLogger, maybe_save_periodic_and_best


class SupConModel(nn.Module):
    """Bundles encoder + projection head so a single checkpoint holds both, with encoder weights
    still cleanly extractable (state dict keys prefixed "encoder." / "projector.")."""

    def __init__(self, n_channels, n_times, embed_dim=256, proj_dim=128):
        super().__init__()
        self.encoder = EEGTransformerEncoder(
            n_channels=n_channels, n_times=n_times, patch_size=config.PATCH_SIZE,
            embed_dim=embed_dim, depth=config.ENCODER_DEPTH, heads=config.ENCODER_HEADS,
        )
        self.projector = ProjectionHead(embed_dim=embed_dim, proj_dim=proj_dim)

    def forward(self, x):
        cls = self.encoder.cls_representation(x)
        return self.projector(cls)


def build_scheduler(optimizer, warmup_epochs, total_epochs):
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs))
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def run(subjects=None, epochs: int = None, device: str = None):
    config.set_seed(config.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epochs or config.SUPCON_EPOCHS

    X, y, subject_ids, label_encoder, subject_encoder = load_physionet_supcon(subjects)
    n_channels, n_times = X.shape[1], X.shape[2]
    print(f"SupCon pretraining classes: {list(label_encoder.classes_)}, "
          f"n_trials={len(X)}, n_subjects={len(subject_encoder.classes_)}")

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
    n_val = max(1, int(0.1 * len(dataset)))
    train_set, val_set = random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(config.SEED),
    )
    train_loader = DataLoader(train_set, batch_size=config.SUPCON_BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config.SUPCON_BATCH_SIZE, shuffle=False, drop_last=True)

    model = SupConModel(n_channels, n_times, embed_dim=config.EMBED_DIM).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.SUPCON_LR,
                                   weight_decay=config.SUPCON_WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, config.SUPCON_WARMUP_EPOCHS, epochs)
    logger = EpochLogger(os.path.join(config.RESULTS_DIR, "pretrain_supcon.csv"))

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_n = 0.0, 0
        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            view1, view2 = make_two_views(x)
            batch = torch.cat([view1, view2], dim=0)
            batch_labels = torch.cat([labels, labels], dim=0)

            optimizer.zero_grad()
            projected = model(batch)
            loss = supcon_loss(projected, batch_labels, temperature=config.SUPCON_TEMPERATURE)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_n += x.size(0)
        scheduler.step()

        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for x, labels in val_loader:
                x, labels = x.to(device), labels.to(device)
                view1, view2 = make_two_views(x)
                batch = torch.cat([view1, view2], dim=0)
                batch_labels = torch.cat([labels, labels], dim=0)
                projected = model(batch)
                loss = supcon_loss(projected, batch_labels, temperature=config.SUPCON_TEMPERATURE)
                val_loss += loss.item() * x.size(0)
                val_n += x.size(0)
        val_loss /= val_n

        logger.log(
            epoch=epoch,
            supcon_loss=train_loss / train_n,
            val_supcon_loss=val_loss,
            lr=optimizer.param_groups[0]["lr"],
        )

        if epoch == 20 and (train_loss / train_n) > 4.0:
            print(f"WARNING: SupCon loss still above 4.0 after 20 epochs "
                  f"({train_loss / train_n:.4f}) -- something may be wrong, check before continuing.")

        best_val_loss = maybe_save_periodic_and_best(
            model, optimizer, epoch, val_loss, best_val_loss,
            config.CHECKPOINT_DIR, "pretrain_supcon", config.SUPCON_CHECKPOINT_EVERY,
        )

    return model, subject_encoder


if __name__ == "__main__":
    run()
