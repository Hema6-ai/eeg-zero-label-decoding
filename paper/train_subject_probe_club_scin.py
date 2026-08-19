"""Post-hoc subject-ID probe for the MAE+CLUB+SCIN checkpoint (mirrors train_subject_probe.py,
but uses EEGMAEClubScin.encode_known_subject for feature extraction since SCIN needs the
subject's own embedding at this stage -- these are PhysioNet subjects seen during pretraining,
unlike the zero-shot BNCI evaluation).

    python -m paper.train_subject_probe_club_scin
"""
import os

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from paper import config
from paper.data import load_physionet_pretrain
from paper.models.mae_club import EEGMAEClubScin
from paper.models.heads import LinearProbe
from paper.utils import EpochLogger


def extract_features(model, X, subject_ids, device, batch_size=256):
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x = X[i:i + batch_size].to(device)
            s = subject_ids[i:i + batch_size].to(device)
            feats.append(model.encode_known_subject(x, s).cpu())
    return torch.cat(feats, dim=0)


def run(subjects=None, epochs: int = None, device: str = None):
    config.set_seed(config.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epochs or config.PROBE_EPOCHS

    X, subject_ids, subject_encoder = load_physionet_pretrain(subjects)
    n_subjects = len(subject_encoder.classes_)
    n_channels, n_times = X.shape[1], X.shape[2]

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
    for p in model.parameters():
        p.requires_grad_(False)

    subj_t = torch.from_numpy(subject_ids).long()
    feats = extract_features(model, torch.from_numpy(X), subj_t, device)
    labels = subj_t

    dataset = TensorDataset(feats, labels)
    n_val = max(1, int(0.2 * len(dataset)))
    train_set, val_set = random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(config.SEED),
    )
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)

    probe = LinearProbe(config.EMBED_DIM, n_subjects).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=config.PROBE_LR)
    logger = EpochLogger(os.path.join(config.RESULTS_DIR, "subject_probe_mae_club_scin.csv"))

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        probe.train()
        for feat, label in train_loader:
            feat, label = feat.to(device), label.to(device)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(probe(feat), label)
            loss.backward()
            optimizer.step()

        probe.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for feat, label in val_loader:
                feat, label = feat.to(device), label.to(device)
                pred = probe(feat).argmax(-1)
                correct += (pred == label).sum().item()
                total += label.size(0)
        val_acc = correct / total
        best_val_acc = max(best_val_acc, val_acc)
        logger.log(epoch=epoch, subject_probe_val_acc=val_acc)

    chance = 1.0 / n_subjects
    print(f"[mae_club_scin] best subject-probe accuracy: {best_val_acc:.4f} (chance = {chance:.4f})")
    return best_val_acc


if __name__ == "__main__":
    run()
