"""TMLR final-pass Phase 1 robustness check: a small hyperparameter search for matched-protocol
SPDNet, selected on source-validation accuracy only (never target/held-out-subject accuracy).

Grid: BiMap final output dimension in {8, 12, 16, 22}, depth (number of BiMap/ReEig stages) in
{1, 2, 3}, learning rate in {1e-3, 3e-4, 1e-4} -- 36 configurations. Intermediate BiMap dims for
depth>1 are evenly spaced between 22 (input channels) and the final output dimension (see
dims_for below); this is the only reasonable choice the grid leaves unspecified, since the
directive gives final dimension and depth, not every intermediate width.

Run on PhysioNet30 first (ea="ea", window="default", seed=42 -- the primary single-seed
comparison used throughout this paper), on a fixed, deterministic subset of folds (documented
below) to keep the search small, as requested. Selection is by mean validation accuracy
(best_val_acc from early stopping) across that subset, never by target/test accuracy.

Resumable: per-config results are appended to a CSV; already-completed configs are skipped.

    python -m paper.tune_spdnet_matched_protocol
"""
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from pyriemann.estimation import Covariances

import moabb.datasets as _moabb_datasets
if not hasattr(_moabb_datasets, "BNCI2014001"):
    _moabb_datasets.BNCI2014001 = _moabb_datasets.BNCI2014_001

from paper import config
from paper.models.spdnet import SPDNetModel
from paper.run_dl_loso_sweep import load_window_subjects, set_all_seeds, physionet_subset, WINDOWS

RESULTS_DIR = config.RESULTS_DIR
COV_ESTIMATOR = "lwf"
PATIENCE = 20
SEARCH_EPOCHS = config.SPDNET_EPOCHS

OUTPUT_DIMS = [8, 12, 16, 22]
DEPTHS = [1, 2, 3]
LRS = [1e-3, 3e-4, 1e-4]

# A fixed, deterministic subset of PhysioNet30's folds for the search (every 3rd subject in
# sorted order of the 30-subject list) -- 10 folds, to keep a 36-config grid tractable. The
# selected configuration is then validated on the FULL fold set (all three datasets) in the
# follow-up full re-run script, not just this subset.
SEARCH_FOLD_FRACTION = 3  # take every Nth subject


def dims_for(output_dim, depth, input_dim=22):
    if depth == 1:
        pts = [input_dim, output_dim]
    else:
        pts = list(np.round(np.linspace(input_dim, output_dim, depth + 1)).astype(int))
        pts[0] = input_dim
        pts[-1] = output_dim
        for i in range(1, len(pts)):
            if pts[i] > pts[i - 1]:
                pts[i] = pts[i - 1]
    return tuple(int(x) for x in pts)


def covariances_for(per_subject):
    out = {}
    for s, (X, y) in per_subject.items():
        covs = Covariances(estimator=COV_ESTIMATOR).fit_transform(X.astype(np.float64)).astype(np.float32)
        out[s] = (covs, y)
    return out


def train_fold_for_validation(Xtr, ytr, Xval, yval, seed, n_classes, dims, lr, device):
    """Same training loop as eval_spdnet_matched_protocol.train_fold_spdnet, but returns only the
    best validation accuracy (never touches test/target data -- there is no test split here)."""
    set_all_seeds(seed)
    model = SPDNetModel(dims=dims, n_classes=n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.SPDNET_WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr).long()
    Xval_t = torch.from_numpy(Xval).float().to(device)

    n = Xtr_t.shape[0]
    batch_size = config.SPDNET_BATCH_SIZE
    best_val_acc, bad_epochs = -1.0, 0
    for epoch in range(SEARCH_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = Xtr_t[idx].to(device)
            yb = ytr_t[idx].to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xval_t).argmax(dim=1).cpu().numpy()
        val_acc = (val_pred == yval).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break
    return float(best_val_acc)


def _results_path():
    return os.path.join(RESULTS_DIR, "spdnet_hparam_search_physionet30.csv")


def _load_done():
    path = _results_path()
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {(int(r["output_dim"]), int(r["depth"]), float(r["lr"])): r for r in csv.DictReader(f)}


def _append(row):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = _results_path()
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    subjects = physionet_subset()
    search_subjects = sorted(subjects)[::SEARCH_FOLD_FRACTION]
    print(f"PhysioNet30 full subject list ({len(subjects)}): {sorted(subjects)}", flush=True)
    print(f"Search fold subset ({len(search_subjects)} of {len(subjects)}, every "
          f"{SEARCH_FOLD_FRACTION}th subject): {search_subjects}", flush=True)

    tmin, tmax = WINDOWS["default"]
    per_subject, label_encoder = load_window_subjects("physionet30", tmin, tmax, "ea", subjects)
    per_subject = covariances_for(per_subject)
    n_classes = len(label_encoder.classes_)
    available = [s for s in subjects if s in per_subject]
    search_subjects = [s for s in search_subjects if s in available]

    done = _load_done()
    total = len(OUTPUT_DIMS) * len(DEPTHS) * len(LRS)
    i = 0
    for depth in DEPTHS:
        for output_dim in OUTPUT_DIMS:
            dims = dims_for(output_dim, depth)
            for lr in LRS:
                i += 1
                key = (output_dim, depth, lr)
                if key in done:
                    print(f"[{i}/{total}] SKIP (done) dims={dims} lr={lr}", flush=True)
                    continue
                val_accs = []
                t0 = time.time()
                for held_out in search_subjects:
                    source_subjects = [s for s in available if s != held_out]
                    val_subject = max(source_subjects)
                    train_subjects = [s for s in source_subjects if s != val_subject]
                    Xtr = np.concatenate([per_subject[s][0] for s in train_subjects], axis=0)
                    ytr = np.concatenate([per_subject[s][1] for s in train_subjects], axis=0)
                    Xval, yval = per_subject[val_subject]
                    val_acc = train_fold_for_validation(Xtr, ytr, Xval, yval, 42, n_classes, dims, lr, device)
                    val_accs.append(val_acc)
                mean_val_acc = float(np.mean(val_accs))
                elapsed = time.time() - t0
                row = {
                    "output_dim": output_dim, "depth": depth, "dims": str(dims), "lr": lr,
                    "mean_val_acc": mean_val_acc, "n_folds": len(val_accs), "elapsed_s": round(elapsed, 1),
                }
                _append(row)
                print(f"[{i}/{total}] dims={dims} lr={lr} mean_val_acc={mean_val_acc*100:.2f}% "
                      f"({len(val_accs)} folds, {elapsed:.1f}s)", flush=True)

    done = _load_done()
    best_key = max(done, key=lambda k: float(done[k]["mean_val_acc"]))
    print(f"\nBest config (by source-validation accuracy only): output_dim={best_key[0]}, "
          f"depth={best_key[1]}, lr={best_key[2]}, dims={done[best_key]['dims']}, "
          f"mean_val_acc={float(done[best_key]['mean_val_acc'])*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
