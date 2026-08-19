"""Pretrain EEG-MAE + SCIN + CLUB + VIB (the proposed method) on PhysioNet.

    python -m paper.train_club_scin

Uses two optimizers (CLUB requires its own MLE-fitting step, separate from the main
encoder/decoder/SCIN/VIB optimizer -- see models/club.py for why) and a 10-epoch linear
warmup followed by cosine decay over the remaining epochs.
"""
import os

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from paper import config
from paper.augment import augment
from paper.data import load_physionet_pretrain
from paper.models.mae_club import EEGMAEClubScin
from paper.utils import EpochLogger, maybe_save_periodic_and_best


def build_scheduler(optimizer, warmup_epochs, total_epochs):
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs - warmup_epochs),
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
    )


def run(subjects=None, epochs: int = None, device: str = None):
    config.set_seed(config.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epochs or config.CLUB_SCIN_EPOCHS

    X, subject_ids, subject_encoder = load_physionet_pretrain(subjects)
    n_subjects = len(subject_encoder.classes_)
    n_channels, n_times = X.shape[1], X.shape[2]

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(subject_ids).long())
    n_val = max(1, int(0.1 * len(dataset)))
    train_set, val_set = random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(config.SEED),
    )
    train_loader = DataLoader(train_set, batch_size=config.PRETRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config.PRETRAIN_BATCH_SIZE, shuffle=False)

    model = EEGMAEClubScin(
        n_channels=n_channels, n_times=n_times, n_subjects=n_subjects,
        patch_size=config.PATCH_SIZE, embed_dim=config.EMBED_DIM,
        encoder_depth=config.ENCODER_DEPTH, encoder_heads=config.ENCODER_HEADS,
        decoder_dim=config.DECODER_DIM, decoder_depth=config.DECODER_DEPTH,
        decoder_heads=config.ENCODER_HEADS, mask_ratio=config.MASK_RATIO,
        club_lambda=config.CLUB_LAMBDA, vib_beta=config.VIB_BETA,
    ).to(device)

    main_params = [p for n, p in model.named_parameters() if not n.startswith("club.")]
    main_optimizer = torch.optim.AdamW(main_params, lr=config.PRETRAIN_LR)
    club_optimizer = torch.optim.AdamW(model.club.parameters(), lr=config.CLUB_LR)
    scheduler = build_scheduler(main_optimizer, config.CLUB_SCIN_WARMUP_EPOCHS, epochs)

    logger = EpochLogger(os.path.join(config.RESULTS_DIR, "pretrain_mae_club_scin.csv"))

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_recon, train_club, train_vib, train_n = 0.0, 0.0, 0.0, 0
        for x, subj in train_loader:
            x, subj = x.to(device), subj.to(device)
            x = augment(
                x, time_warp_prob=config.AUGMENT_TIME_WARP_PROB,
                max_warp=config.AUGMENT_MAX_WARP,
                channel_dropout_prob=config.AUGMENT_CHANNEL_DROPOUT_PROB,
            )

            losses = model(x, subj)

            # Order matters: main_loss's graph includes club.net(z) (non-detached), while
            # club_mle_loss uses club.net(z.detach()) -- a separate graph through the same
            # weights. Stepping club_optimizer first would modify those weights in-place while
            # main_loss's backward still needed their pre-step values (autograd raises a
            # version-mismatch error). Doing main_loss first avoids the conflict entirely.
            main_optimizer.zero_grad()
            losses["main_loss"].backward()
            main_optimizer.step()

            club_optimizer.zero_grad()
            losses["club_mle_loss"].backward()
            club_optimizer.step()

            train_recon += losses["recon_loss"].item() * x.size(0)
            train_club += losses["club_bound"].item() * x.size(0)
            train_vib += losses["vib_kl"].item() * x.size(0)
            train_n += x.size(0)
        scheduler.step()

        model.eval()
        val_recon, val_n = 0.0, 0
        with torch.no_grad():
            for x, subj in val_loader:
                x, subj = x.to(device), subj.to(device)
                losses = model(x, subj)
                val_recon += losses["recon_loss"].item() * x.size(0)
                val_n += x.size(0)
        val_recon /= val_n

        logger.log(
            epoch=epoch,
            recon_loss=train_recon / train_n,
            club_bound=train_club / train_n,
            vib_kl=train_vib / train_n,
            val_recon_loss=val_recon,
            lr=main_optimizer.param_groups[0]["lr"],
        )

        best_val_loss = maybe_save_periodic_and_best(
            model, main_optimizer, epoch, val_recon, best_val_loss,
            config.CHECKPOINT_DIR, "pretrain_mae_club_scin", config.CLUB_SCIN_CHECKPOINT_EVERY,
        )

    return model, subject_encoder


if __name__ == "__main__":
    run()
