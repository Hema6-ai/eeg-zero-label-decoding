"""TMLR revision round 2, task E2(c): CBraMod zero-shot comparison, mirroring
paper/eval_foundation_model_labram.py's exact protocol (frozen backbone, linear probe per LOSO
fold, no fine-tuning, strict zero-shot).

Why CBraMod but not BIOT (verified this session, not guessed from a title/README):
  - CBraMod (braindecode `CBraMod` class, GitHub main branch, not yet on PyPI) takes channel
    COUNT only (n_chans) via an "Adaptive Channel/Position Encoding" design explicitly built to
    generalize across different montages without retraining -- no fixed channel-name vocabulary,
    so this project's 22-channel BCI-IV-2a montage is directly usable.
  - BIOT's released checkpoints (`braindecode/biot-pretrained-*`) require a FIXED 18-entry
    bipolar (TCP-style) channel list (e.g. "FP1-F7", "F7-T7", ..., "C3-A2", "C4-A1") baked into a
    position-indexed channel-token embedding. This project's 22-channel motor-cortex montage
    (Fz/FC.../C.../CP.../P...) is monopolar and does not even include the source electrodes
    (F7, F8, T7, T8, P7, P8, O1, O2) needed to form BIOT's required bipolar derivations -- not a
    channel-SUBSET problem (which could be worked around like LaBraM's 20-of-22 case), a
    channel-SCHEME mismatch. BIOT is therefore not run; documented here rather than attempted
    with silently-misaligned channel tokens.

Requires braindecode's `from_pretrained`/Hub support, which is on the GitHub main branch only as
of this writing (not yet released to PyPI; installed via `pip install braindecode[hug]` from git
in the Kaggle kernel entry point, not locally -- see kaggle_kernel/e2_cbramod_push/run_kernel.py).

Checkpoint: `braindecode/cbramod-pretrained` (reported config: n_chans=22, n_times=1000,
sfreq=250). This project's standard window is [0,3]s at 128Hz; for this comparison only (same
disclosed-deviation pattern as LaBraM's 200Hz requirement), BCI-IV 2a is resampled to 250Hz and
windowed to [0,4]s (1000 samples, exactly 5 patches of CBraMod's 200-sample patch size).

    python -m paper.eval_foundation_model_cbramod
"""
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from paper import config
from paper.models.heads import LinearProbe
from paper.utils import classification_metrics

CBRAMOD_SFREQ = 250
CBRAMOD_TMIN, CBRAMOD_TMAX = 0.0, 4.0  # -> 1000 samples at 250Hz = 5 patches of 200


def load_bnci_250hz(subjects=None):
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery
    from sklearn.preprocessing import LabelEncoder

    subjects = subjects or config.EVAL_SUBJECTS
    paradigm = MotorImagery(channels=config.COMMON_CHANNELS, resample=CBRAMOD_SFREQ,
                             tmin=CBRAMOD_TMIN, tmax=CBRAMOD_TMAX)
    dataset = BNCI2014_001()
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
    X = X[..., :1000]  # guard the same +/-1-sample MNE resample edge case seen elsewhere

    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    X = ((X - mean) / (std + 1e-6)).astype(np.float32)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    per_subject = {}
    for subject in subjects:
        idx = metadata["subject"].values == subject
        per_subject[subject] = (X[idx], y[idx])
    return per_subject, label_encoder


def load_frozen_cbramod(device):
    from braindecode.models import CBraMod
    # n_outputs is required to construct the checkpoint's classification head (a LazyLinear we
    # never actually use -- only return_features=True / the "features" dict entry is read, see
    # extract_features() below), so its value here is arbitrary and does not affect the frozen
    # backbone features at all.
    model = CBraMod.from_pretrained("braindecode/cbramod-pretrained", n_outputs=2)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def extract_features(model, X, device, batch_size=32):
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).to(device)
            out = model(batch, return_features=True)
            f = out["features"] if isinstance(out, dict) else out
            f = f.reshape(f.shape[0], -1)  # flatten per-channel/per-patch grid for a linear probe
            feats.append(f.cpu())
    return torch.cat(feats, dim=0)


def train_linear_head(feats, labels, n_classes, device, epochs=None, lr=None):
    epochs = epochs or config.PROBE_EPOCHS
    lr = lr or config.PROBE_LR
    head = LinearProbe(feats.shape[1], n_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(feats, labels), batch_size=64, shuffle=True)
    head.train()
    for _ in range(epochs):
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

    per_subject, label_encoder = load_bnci_250hz(subjects)
    n_classes = len(label_encoder.classes_)
    n_chans, n_times = next(iter(per_subject.values()))[0].shape[1:]
    print(f"CBraMod comparison: n_chans={n_chans}, n_times={n_times} @ {CBRAMOD_SFREQ}Hz, "
          f"n_classes={n_classes}", flush=True)

    encoder = load_frozen_cbramod(device)

    fold_results = []
    for held_out in subjects:
        source_subjects = [s for s in subjects if s != held_out]
        source_X = np.concatenate([per_subject[s][0] for s in source_subjects], axis=0)
        source_y = np.concatenate([per_subject[s][1] for s in source_subjects], axis=0)
        target_X, target_y = per_subject[held_out]

        source_feats = extract_features(encoder, source_X, device)
        target_feats = extract_features(encoder, target_X, device)

        head = train_linear_head(source_feats, torch.from_numpy(source_y).long(), n_classes, device)
        head.eval()
        with torch.no_grad():
            pred = head(target_feats.to(device)).argmax(-1).cpu().numpy()

        metrics = classification_metrics(target_y, pred)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        print(f"[cbramod] held-out subject {held_out}: acc={metrics['accuracy']:.4f} "
              f"kappa={metrics['kappa']:.4f}", flush=True)

    accs = np.array([r["accuracy"] for r in fold_results])
    kappas = np.array([r["kappa"] for r in fold_results])
    summary = {"variant": "cbramod_zeroshot_linear_probe", "n_subjects": len(fold_results),
               "n_chans_used": n_chans, "accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
               "kappa_mean": kappas.mean(), "kappa_std": kappas.std()}
    print(f"\n=== CBraMod zero-shot LOSO: {accs.mean()*100:.2f}% +/- {accs.std()*100:.2f}% "
          f"kappa={kappas.mean():.3f} ===", flush=True)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "zero_shot_cbramod.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "accuracy", "kappa"])
        writer.writeheader()
        for r in fold_results:
            writer.writerow(r)
    with open(os.path.join(config.RESULTS_DIR, "cbramod_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    return summary, fold_results


if __name__ == "__main__":
    run()
