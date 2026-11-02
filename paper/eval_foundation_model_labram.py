"""Lightweight, no-fine-tuning zero-shot comparison against LaBraM (Jiang et al., 2024), a
pretrained EEG foundation model, addressing TMLR review Critical Issue D (absence of comparisons
with recent EEG foundation models).

Model and checkpoint
---------------------
- Architecture: `braindecode.models.Labram` (braindecode >= 1.2.0), the maintained, MIT-licensed
  integration of LaBraM (https://arxiv.org/abs/2405.18765, ICLR 2024 spotlight) used here instead
  of the original research-code repo for reproducibility (pip-installable, documented API).
- Checkpoint: `braindecode/Labram-Braindecode` on HuggingFace (MIT-licensed, a declared "shallow
  conversion from the original weight for braindecode"), file `braindecode_labram_base.pt` --
  downloaded via `huggingface_hub.hf_hub_download`, not re-hosted or modified here.
- This is the "base" checkpoint: position_embedding has shape (1, 65, 200) -- 1 CLS token + 64
  channel slots -- i.e. this specific pretrained checkpoint has a FIXED, SMALLER 64-channel
  position-embedding vocabulary (not a dynamic/universal one), unlike what the model architecture
  in principle supports.

Channel mapping (why 20 channels, not this project's usual 22)
----------------------------------------------------------------
LaBraM's original repository (https://github.com/935963004/LaBraM, `utils.py`) defines a channel
name/order vocabulary `standard_1020` and a `get_input_chans(ch_names)` helper that maps channel
names to table row indices (offset by 1 for the CLS token). That exact list and function are
reproduced below (verified against the live source file this session, not paraphrased or
guessed) for the channel-index lookup. Checking this project's usual 22-channel montage
(`config.COMMON_CHANNELS`) against it: 20 of 22 channels fall within this checkpoint's 64-slot
table; **P2** (row index 64, i.e. slot 65 -- one past the last valid slot) and **POz** (row index
74) do not. Both are dropped for this experiment only; every other script in this project
continues to use the full 22-channel montage. This is a real, disclosed scope reduction specific
to this one pretrained checkpoint's vocabulary size, not a general property of LaBraM or of this
project's pipeline.

Sampling rate and window
--------------------------
LaBraM's temporal patching is a fixed `patch_size=200` samples; the project's default 128 Hz would
make each patch 1.5625s, an unusual choice. This experiment resamples BCI-IV 2a to 200 Hz
(independent of the rest of the project's 128 Hz convention) and keeps tmin=0/tmax=3s (standard
project window), giving 600 samples -> 3 patches/channel, a conventional setup for this model.

Protocol: strict zero-shot LOSO (Protocol A)
-----------------------------------------------
The LaBraM backbone is used purely as a FROZEN feature extractor -- no fine-tuning of any LaBraM
parameter, ever (matching the review's definition: a pretrained EEG model that requires
target-subject supervised fine-tuning would NOT be a valid strict zero-shot baseline, so this
explicitly avoids that). A linear probe is fit on the frozen features of the source (8) subjects
and evaluated on the held-out subject, for every fold -- identical protocol to every other Family
1/2 method in this project (`paper/eval_zero_shot.py` etc).

    python -m paper.eval_foundation_model_labram
"""
import csv
import os

import moabb.datasets as _md
if not hasattr(_md, "BNCI2014001"):
    # braindecode 1.2.0 imports moabb.datasets.BNCI2014001 (pre-rename); this project's moabb
    # version uses BNCI2014_001. Compatibility alias only, no behavior change to either library.
    # MUST be at module level (not just in __main__, where it lived before and silently failed
    # to apply whenever this module was imported as a library rather than run directly -- the
    # actual cause of both LaBraM-fine-tune Kaggle kernels' first-attempt failures) so it applies
    # regardless of how this module is imported.
    _md.BNCI2014001 = _md.BNCI2014_001

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from mne.filter import resample
from torch.utils.data import DataLoader, TensorDataset

from paper import config
from paper.models.heads import LinearProbe
from paper.utils import classification_metrics

# Reproduced verbatim from https://github.com/935963004/LaBraM/blob/main/utils.py (verified live
# this session), truncated after the first 64 entries since this checkpoint's position_embedding
# table only has 64 channel slots (see module docstring).
STANDARD_1020_FIRST_64 = [
    "FP1", "FPZ", "FP2", "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8", "AF10",
    "F9", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "F10",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T9", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "T10",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "PZ",
]

# This project's standard 22-channel montage (paper/config.py::COMMON_CHANNELS), minus the 2
# channels not covered by this checkpoint's 64-slot vocabulary (P2, POz).
LABRAM_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz",
]
DROPPED_CHANNELS = ["P2", "POz"]

LABRAM_SFREQ = 200
LABRAM_PATCH_SIZE = 200
LABRAM_TMIN, LABRAM_TMAX = config.TMIN, config.TMAX  # 0.0, 3.0s -- same window as the rest of the project


def get_input_chans(ch_names):
    """Verbatim port of LaBraM's own `get_input_chans` (see module docstring for source)."""
    input_chans = [0]  # CLS token
    for ch_name in ch_names:
        input_chans.append(STANDARD_1020_FIRST_64.index(ch_name.upper()) + 1)
    return input_chans


def load_bnci_200hz(subjects=None):
    """BCI-IV 2a at 200 Hz, restricted to LABRAM_CHANNELS, for this experiment only -- mirrors
    paper/data.py::load_bnci_all but with this experiment's own sampling rate/channel subset
    rather than touching the shared config (this deviation is specific to this one comparison)."""
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery
    from sklearn.preprocessing import LabelEncoder

    subjects = subjects or config.EVAL_SUBJECTS
    paradigm = MotorImagery(channels=LABRAM_CHANNELS, resample=LABRAM_SFREQ,
                             tmin=LABRAM_TMIN, tmax=LABRAM_TMAX)
    dataset = BNCI2014_001()
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
    # MNE's resample(tmin=0, tmax=3) can yield 601 samples (inclusive endpoint) instead of the
    # exact 600 = 3 patches x 200 samples LaBraM's fixed patch_size expects; truncate to exactly
    # n_patches*patch_size, dropping at most one trailing sample (3.3ms at 200Hz, negligible).
    n_times_exact = (X.shape[-1] // LABRAM_PATCH_SIZE) * LABRAM_PATCH_SIZE
    X = X[..., :n_times_exact]

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


def load_frozen_labram(n_chans, n_times, device):
    """Downloads the braindecode-converted LaBraM-base checkpoint (MIT-licensed, see module
    docstring) and returns a frozen encoder whose position_embedding has been sliced to exactly
    this experiment's channel subset (in LABRAM_CHANNELS order) and whose temporal_embedding has
    been sliced to this experiment's patch count."""
    from braindecode.models import Labram

    ckpt_path = hf_hub_download("braindecode/Labram-Braindecode", "braindecode_labram_base.pt")
    state_dict = dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    orig_pos_embed = state_dict.pop("position_embedding")
    orig_temp_embed = state_dict.pop("temporal_embedding")
    # The classification head is task-specific (this checkpoint happens to still carry one from
    # its own conversion/testing); we never use it -- features come from forward_features(), which
    # stops before final_layer. Drop it so load_state_dict doesn't choke on the shape mismatch.
    for key in [k for k in state_dict if k.startswith("final_layer")]:
        state_dict.pop(key)

    model = Labram(n_chans=n_chans, n_times=n_times, n_outputs=2, patch_size=LABRAM_PATCH_SIZE)
    incompatible = model.load_state_dict(state_dict, strict=False)
    expected_missing = {"position_embedding", "temporal_embedding", "final_layer.weight", "final_layer.bias"}
    assert set(incompatible.missing_keys) == expected_missing, (
        f"Unexpected missing keys when loading the LaBraM backbone: {incompatible.missing_keys}")

    input_chans = get_input_chans(LABRAM_CHANNELS)
    with torch.no_grad():
        model.position_embedding.copy_(orig_pos_embed[:, input_chans])
        n_patch = model.temporal_embedding.shape[1]
        model.temporal_embedding.copy_(orig_temp_embed[:, :n_patch])

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def extract_features(model, X, device, batch_size=64):
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).to(device)
            out = model.forward_features(batch, return_all_tokens=False, return_patch_tokens=False)
            feats.append(out.cpu())
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

    per_subject, label_encoder = load_bnci_200hz(subjects)
    n_classes = len(label_encoder.classes_)
    n_chans, n_times = next(iter(per_subject.values()))[0].shape[1:]
    print(f"LaBraM comparison: n_chans={n_chans} (dropped {DROPPED_CHANNELS}), "
          f"n_times={n_times} @ {LABRAM_SFREQ}Hz, n_classes={n_classes}")

    encoder = load_frozen_labram(n_chans, n_times, device)

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
        print(f"[labram] held-out subject {held_out}: acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

    accs = np.array([r["accuracy"] for r in fold_results])
    kappas = np.array([r["kappa"] for r in fold_results])
    summary = {"variant": "labram_zeroshot_linear_probe", "n_subjects": len(fold_results),
               "n_chans_used": n_chans, "dropped_channels": ",".join(DROPPED_CHANNELS),
               "accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
               "kappa_mean": kappas.mean(), "kappa_std": kappas.std()}
    print(f"[labram] LOSO accuracy: {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}  "
          f"kappa: {summary['kappa_mean']:.4f} +/- {summary['kappa_std']:.4f}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "zero_shot_labram.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "accuracy", "kappa"])
        writer.writeheader()
        for r in fold_results:
            writer.writerow({"held_out_subject": r["held_out_subject"],
                              "accuracy": r["accuracy"], "kappa": r["kappa"]})

    return summary, fold_results


if __name__ == "__main__":
    run()
