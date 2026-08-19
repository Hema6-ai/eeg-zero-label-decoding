"""Post-hoc subject-ID probe for the SupCon-pretrained encoder (mirrors train_subject_probe.py).

    python -m paper.train_subject_probe_supcon
"""
import os

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from paper import config
from paper.data import load_physionet_supcon
from paper.eval_zero_shot_supcon import load_frozen_supcon_encoder
from paper.models.heads import LinearProbe
from paper.utils import EpochLogger


def extract_cls_features(encoder, X, device, batch_size=256):
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size].to(device)
            feats.append(encoder.cls_representation(batch).cpu())
    return torch.cat(feats, dim=0)


def run(subjects=None, epochs: int = None, device: str = None):
    config.set_seed(config.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epochs or config.PROBE_EPOCHS

    X, y, subject_ids, label_encoder, subject_encoder = load_physionet_supcon(subjects)
    n_subjects = len(subject_encoder.classes_)
    n_channels, n_times = X.shape[1], X.shape[2]

    encoder = load_frozen_supcon_encoder(n_channels, n_times, device)

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
    logger = EpochLogger(os.path.join(config.RESULTS_DIR, "subject_probe_supcon.csv"))

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
    print(f"[supcon] best subject-probe accuracy: {best_val_acc:.4f} (chance = {chance:.4f})")
    return best_val_acc


if __name__ == "__main__":
    run()
