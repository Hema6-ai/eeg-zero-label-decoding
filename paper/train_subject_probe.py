"""Post-hoc subject-ID probe: freeze the pretrained encoder, train a fresh linear probe
to predict subject identity from the pooled CLS representation.

Near-chance accuracy (1/109 = 0.9%) indicates the encoder's global representation has been
successfully disentangled from subject identity. Run once per ablation variant:

    python -m paper.train_subject_probe --variant mae_grl
    python -m paper.train_subject_probe --variant mae_only
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from paper import config
from paper.data import load_physionet_pretrain
from paper.models.encoder import EEGTransformerEncoder
from paper.models.heads import LinearProbe
from paper.utils import EpochLogger


def extract_cls_features(encoder, X, device, batch_size=256):
    encoder.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size].to(device)
            feats.append(encoder.cls_representation(batch).cpu())
    return torch.cat(feats, dim=0)


def run(variant: str, subjects=None, epochs: int = None, device: str = None):
    config.set_seed(config.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epochs or config.PROBE_EPOCHS

    X, subject_ids, subject_encoder = load_physionet_pretrain(subjects)
    n_subjects = len(subject_encoder.classes_)
    n_channels, n_times = X.shape[1], X.shape[2]

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"pretrain_{variant}_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    encoder = EEGTransformerEncoder(
        n_channels=n_channels, n_times=n_times, patch_size=config.PATCH_SIZE,
        embed_dim=config.EMBED_DIM, depth=config.ENCODER_DEPTH, heads=config.ENCODER_HEADS,
    ).to(device)
    encoder_state = {k[len("encoder."):]: v for k, v in ckpt["model"].items() if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_state)
    for p in encoder.parameters():
        p.requires_grad_(False)

    X_t = torch.from_numpy(X)
    feats = extract_cls_features(encoder, X_t, device)
    labels = torch.from_numpy(subject_ids).long()

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
    logger = EpochLogger(os.path.join(config.RESULTS_DIR, f"subject_probe_{variant}.csv"))

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
    print(f"[{variant}] best subject-probe accuracy: {best_val_acc:.4f} (chance = {chance:.4f})")
    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["mae_only", "mae_grl"], required=True)
    parser.add_argument("--subjects", type=int, nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    run(args.variant, subjects=args.subjects, epochs=args.epochs)
