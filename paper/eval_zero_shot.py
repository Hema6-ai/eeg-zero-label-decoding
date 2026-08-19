"""Zero-shot leave-one-subject-out (LOSO) evaluation on BNCI2014_001 (BCI-IV 2a).

For each held-out subject: freeze the PhysioNet-pretrained encoder, train a linear MI
classifier on the CLS representations of the *other* 8 subjects only, then evaluate on the
held-out subject. No calibration data from the held-out subject is used anywhere.

    python -m paper.eval_zero_shot --variant mae_grl
    python -m paper.eval_zero_shot --variant mae_only
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from paper import config
from paper.data import load_bnci_all
from paper.models.encoder import EEGTransformerEncoder
from paper.models.heads import LinearProbe
from paper.utils import classification_metrics


def load_frozen_encoder(variant: str, n_channels: int, n_times: int, device: str, align: str = "none"):
    run_name = f"pretrain_{variant}" + (f"_{align}" if align != "none" else "")
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{run_name}_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    encoder = EEGTransformerEncoder(
        n_channels=n_channels, n_times=n_times, patch_size=config.PATCH_SIZE,
        embed_dim=config.EMBED_DIM, depth=config.ENCODER_DEPTH, heads=config.ENCODER_HEADS,
    ).to(device)
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


def train_linear_head(feats, labels, n_classes, device, epochs=None, lr=None):
    epochs = epochs or config.PROBE_EPOCHS
    lr = lr or config.PROBE_LR
    head = LinearProbe(config.EMBED_DIM, n_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    loader = DataLoader(TensorDataset(feats, labels), batch_size=64, shuffle=True)
    head.train()
    for _ in range(epochs):
        for feat, label in loader:
            feat, label = feat.to(device), label.to(device)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(head(feat), label)
            loss.backward()
            optimizer.step()
    return head


def run(variant: str, subjects=None, device: str = None, align: str = "none"):
    config.set_seed(config.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    subjects = subjects or config.EVAL_SUBJECTS

    per_subject, label_encoder = load_bnci_all(subjects, align=align)
    n_channels, n_times = next(iter(per_subject.values()))[0].shape[1:]
    n_classes = len(label_encoder.classes_)

    encoder = load_frozen_encoder(variant, n_channels, n_times, device, align=align)

    fold_results = []
    for held_out in subjects:
        source_subjects = [s for s in subjects if s != held_out]

        source_X = np.concatenate([per_subject[s][0] for s in source_subjects], axis=0)
        source_y = np.concatenate([per_subject[s][1] for s in source_subjects], axis=0)
        target_X, target_y = per_subject[held_out]

        source_feats = extract_cls_features(encoder, source_X, device)
        target_feats = extract_cls_features(encoder, target_X, device)

        head = train_linear_head(source_feats, torch.from_numpy(source_y).long(), n_classes, device)

        head.eval()
        with torch.no_grad():
            pred = head(target_feats.to(device)).argmax(-1).cpu().numpy()
        metrics = classification_metrics(target_y, pred)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        print(f"[{variant}] held-out subject {held_out}: acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

    run_name = f"{variant}" + (f"_{align}" if align != "none" else "")
    accs = np.array([r["accuracy"] for r in fold_results])
    kappas = np.array([r["kappa"] for r in fold_results])
    summary = {
        "variant": run_name,
        "accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
        "kappa_mean": kappas.mean(), "kappa_std": kappas.std(),
    }
    print(f"[{run_name}] LOSO accuracy: {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}  "
          f"kappa: {summary['kappa_mean']:.4f} +/- {summary['kappa_std']:.4f}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    import csv
    with open(os.path.join(config.RESULTS_DIR, f"zero_shot_{run_name}.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "accuracy", "kappa"])
        writer.writeheader()
        for r in fold_results:
            writer.writerow(r)

    return summary, fold_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["mae_only", "mae_grl"], required=True)
    parser.add_argument("--subjects", type=int, nargs="*", default=None)
    parser.add_argument("--align", choices=["none", "ea", "ra"], default="none")
    args = parser.parse_args()
    run(args.variant, subjects=args.subjects, align=args.align)
