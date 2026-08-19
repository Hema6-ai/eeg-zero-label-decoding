"""MAE(+GRL) pretraining on PhysioNet. Run twice with --variant mae_only / mae_grl for the ablation.

    python -m paper.train_pretrain --variant mae_grl
    python -m paper.train_pretrain --variant mae_only
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from paper import config
from paper.data import load_physionet_pretrain
from paper.models.mae import EEGMAE
from paper.utils import EpochLogger, maybe_save_periodic_and_best


def grl_lambda_schedule(epoch: int, warmup_epochs: int, lambda_max: float) -> float:
    return lambda_max * min(1.0, epoch / max(1, warmup_epochs))


def run(variant: str, subjects=None, epochs: int = None, device: str = None, align: str = "none"):
    assert variant in ("mae_only", "mae_grl")
    use_grl = variant == "mae_grl"
    config.set_seed(config.SEED)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epochs or config.PRETRAIN_EPOCHS
    run_name = f"pretrain_{variant}" + (f"_{align}" if align != "none" else "")

    X, subject_ids, subject_encoder = load_physionet_pretrain(subjects, align=align)
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

    model = EEGMAE(
        n_channels=n_channels, n_times=n_times, n_subjects=n_subjects,
        patch_size=config.PATCH_SIZE, embed_dim=config.EMBED_DIM,
        encoder_depth=config.ENCODER_DEPTH, encoder_heads=config.ENCODER_HEADS,
        decoder_dim=config.DECODER_DIM, decoder_depth=config.DECODER_DEPTH, decoder_heads=config.ENCODER_HEADS,
        mask_ratio=config.MASK_RATIO, use_grl=use_grl, grl_lambda=0.0,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.PRETRAIN_LR)
    logger = EpochLogger(os.path.join(config.RESULTS_DIR, f"{run_name}.csv"))

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        if use_grl:
            model.set_grl_lambda(grl_lambda_schedule(epoch, config.GRL_WARMUP_EPOCHS, config.GRL_LAMBDA_MAX))

        model.train()
        train_recon, train_adv, train_n = 0.0, 0.0, 0
        for x, subj in train_loader:
            x, subj = x.to(device), subj.to(device)
            optimizer.zero_grad()
            losses = model(x, subj if use_grl else None)
            losses["loss"].backward()
            optimizer.step()

            train_recon += losses["recon_loss"].item() * x.size(0)
            if use_grl:
                train_adv += losses["adv_loss"].item() * x.size(0)
            train_n += x.size(0)

        model.eval()
        val_recon, val_adv, val_adv_correct, val_n = 0.0, 0.0, 0, 0
        with torch.no_grad():
            for x, subj in val_loader:
                x, subj = x.to(device), subj.to(device)
                losses = model(x, subj if use_grl else None)
                val_recon += losses["recon_loss"].item() * x.size(0)
                if use_grl:
                    val_adv += losses["adv_loss"].item() * x.size(0)
                    val_adv_correct += (losses["subject_logits"].argmax(-1) == subj).sum().item()
                val_n += x.size(0)

        val_recon /= val_n
        val_adv /= val_n if use_grl else 1
        subject_probe_acc = (val_adv_correct / val_n) if use_grl else float("nan")

        logger.log(
            epoch=epoch,
            recon_loss=train_recon / train_n,
            adv_loss=(train_adv / train_n) if use_grl else 0.0,
            val_recon_loss=val_recon,
            subject_probe_acc=subject_probe_acc,
        )

        best_val_loss = maybe_save_periodic_and_best(
            model, optimizer, epoch, val_recon, best_val_loss,
            config.CHECKPOINT_DIR, run_name, config.CHECKPOINT_EVERY,
        )

    return model, subject_encoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["mae_only", "mae_grl"], required=True)
    parser.add_argument("--subjects", type=int, nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--align", choices=["none", "ea", "ra"], default="none")
    args = parser.parse_args()
    run(args.variant, subjects=args.subjects, epochs=args.epochs, align=args.align)
