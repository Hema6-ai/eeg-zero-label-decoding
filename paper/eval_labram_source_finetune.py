"""TMLR revision round 2, task A: LaBraM SOURCE-ONLY fine-tuning. This is a zero-label-legal
regime distinct from the strict frozen-backbone zero-shot protocol in
paper/eval_foundation_model_labram.py: the LaBraM backbone itself is fine-tuned (not just a
linear probe on frozen features), but ONLY on the 8 source subjects' data per LOSO fold --
the held-out target subject's data is never touched during training or model selection. Early
stopping uses one source subject held out as a validation split (never the target), so there is
no test-subject tuning anywhere. Reported in a separate, clearly-labeled row from the strict
zero-shot result, per the task's own instruction.

Two fine-tuning regimes:
  - "full": every LaBraM backbone parameter is trainable.
  - "last2": only the last 2 of LaBraM-base's 12 transformer blocks (+ the final norm) are
    trainable; everything earlier (patch embedding, positional/temporal embeddings, blocks 0-9)
    stays frozen. A common, cheaper fine-tuning regime that still adapts high-level features.

A fresh linear classification head (never the pretrained final_layer, which was for a different
task) is trained jointly with whichever backbone parameters are unfrozen.

    python -m paper.eval_labram_source_finetune --mode full
    python -m paper.eval_labram_source_finetune --mode last2
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from paper import config
from paper.eval_foundation_model_labram import load_bnci_200hz, load_frozen_labram, DROPPED_CHANNELS
from paper.utils import classification_metrics

MAX_EPOCHS = 40
PATIENCE = 6
BATCH_SIZE = 32
LR_BACKBONE = 1e-5
LR_HEAD = 1e-3
N_LAST_BLOCKS = 2


def set_trainable(model, mode):
    for p in model.parameters():
        p.requires_grad_(mode == "full")
    if mode == "last2":
        for blk in model.blocks[-N_LAST_BLOCKS:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        for p in model.norm.parameters():
            p.requires_grad_(True)
        if model.fc_norm is not None:
            for p in model.fc_norm.parameters():
                p.requires_grad_(True)


def forward_cls(model, x):
    """Full-gradient forward to the pooled (mean-over-patch-tokens, or CLS if no fc_norm)
    representation -- mirrors eval_foundation_model_labram.py::extract_features's
    forward_features call, but WITHOUT torch.no_grad(), for fine-tuning."""
    return model.forward_features(x, return_all_tokens=False, return_patch_tokens=False)


def train_one_fold(mode, Xtr, ytr, Xval, yval, Xtest, ytest, n_chans, n_times, n_classes, device, seed=42):
    torch.manual_seed(seed)
    model = load_frozen_labram(n_chans, n_times, device)  # starts fully frozen
    set_trainable(model, mode)

    with torch.no_grad():
        sample_feat = forward_cls(model, torch.from_numpy(Xtr[:1]).to(device))
    head = nn.Linear(sample_feat.shape[1], n_classes).to(device)

    backbone_params = [p for p in model.parameters() if p.requires_grad]
    param_groups = [{"params": head.parameters(), "lr": LR_HEAD}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": LR_BACKBONE})
    optimizer = torch.optim.Adam(param_groups)

    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr).long()
    Xval_t = torch.from_numpy(Xval).float().to(device)
    Xtest_t = torch.from_numpy(Xtest).float().to(device)
    n = len(Xtr_t)

    best_val_acc, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        head.train()
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb = Xtr_t[idx].to(device)
            yb = ytr_t[idx].to(device)
            optimizer.zero_grad()
            feat = forward_cls(model, xb)
            loss = F.cross_entropy(head(feat), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        head.eval()
        with torch.no_grad():
            val_feat = forward_cls(model, Xval_t)
            val_pred = head(val_feat).argmax(dim=1).cpu().numpy()
        val_acc = (val_pred == yval).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = ({k: v.detach().clone() for k, v in model.state_dict().items()},
                           {k: v.detach().clone() for k, v in head.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    model.load_state_dict(best_state[0])
    head.load_state_dict(best_state[1])
    model.eval()
    head.eval()
    with torch.no_grad():
        test_feat = forward_cls(model, Xtest_t)
        test_pred = head(test_feat).argmax(dim=1).cpu().numpy()
    metrics = classification_metrics(ytest, test_pred)
    return metrics, epoch + 1, best_val_acc


def run(mode, subjects=None, device=None):
    config.set_seed(config.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    subjects = subjects or config.EVAL_SUBJECTS

    per_subject, label_encoder = load_bnci_200hz(subjects)
    n_classes = len(label_encoder.classes_)
    n_chans, n_times = next(iter(per_subject.values()))[0].shape[1:]
    print(f"LaBraM source-only fine-tune (mode={mode}): n_chans={n_chans} "
          f"(dropped {DROPPED_CHANNELS}), n_times={n_times}, n_classes={n_classes}", flush=True)

    out_path = os.path.join(config.RESULTS_DIR, f"labram_source_finetune_{mode}.csv")
    done = {}
    if os.path.exists(out_path):
        with open(out_path, newline="") as f:
            for row in csv.DictReader(f):
                done[int(row["held_out_subject"])] = row

    fold_results = list(done.values())
    for held_out in subjects:
        if held_out in done:
            continue
        source_subjects = [s for s in subjects if s != held_out]
        val_subject = max(source_subjects)
        train_subjects = [s for s in source_subjects if s != val_subject]

        Xtr = np.concatenate([per_subject[s][0] for s in train_subjects], axis=0)
        ytr = np.concatenate([per_subject[s][1] for s in train_subjects], axis=0)
        Xval, yval = per_subject[val_subject]
        Xtest, ytest = per_subject[held_out]

        metrics, n_epochs, best_val_acc = train_one_fold(
            mode, Xtr, ytr, Xval, yval, Xtest, ytest, n_chans, n_times, n_classes, device)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        print(f"[labram-finetune-{mode}] held_out={held_out} acc={metrics['accuracy']:.4f} "
              f"kappa={metrics['kappa']:.4f} epochs={n_epochs} val_acc={best_val_acc:.4f}", flush=True)

        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        write_header = not os.path.exists(out_path)
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["held_out_subject", "accuracy", "kappa"])
            if write_header:
                writer.writeheader()
            writer.writerow({"held_out_subject": held_out, "accuracy": metrics["accuracy"],
                              "kappa": metrics["kappa"]})

    accs = np.array([float(r["accuracy"]) for r in fold_results])
    kappas = np.array([float(r["kappa"]) for r in fold_results])
    print(f"\n=== LaBraM source-only fine-tune ({mode}): {accs.mean()*100:.2f}% +/- "
          f"{accs.std()*100:.2f}% kappa={kappas.mean():.3f} (n={len(accs)}) ===", flush=True)
    return accs.mean(), accs.std()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "last2"], default="full")
    args = parser.parse_args()
    run(args.mode)
