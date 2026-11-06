"""TMLR revision round 3, Reviewer c1um's requested fairness check: SPDNet trained end-to-end,
supervised, from scratch, on source subjects only, under the EXACT same LOSO fold structure,
subject lists, seeds, EA on/off, and window sweep as paper/run_dl_loso_sweep.py's EEGNetv4 runs
(and paper/eval_e1_eegnet_physionet_full.py for the full-PhysioNet cell) -- not the population-
swapped self-supervised pretraining paper/train_spdnet.py uses elsewhere in this paper.

Input representation differs from EEGNetv4 (raw EEG time series) because SPDNet's BiMap/ReEig/
LogEig layers require SPD covariance matrices, not raw time series -- this is an architectural
property of SPDNet, not a protocol choice, so it is kept and documented rather than worked around.
Per-subject covariances are computed with the paper's canonical Ledoit-Wolf estimator after EA
(when ea="ea"), matching paper/train_spdnet.py's own convention exactly.

Hyperparameters (dims=(22,16,10), lr, weight decay, batch size, max epochs, patience) are the
project's existing, previously-chosen SPDNet defaults (paper/config.py), applied identically to
every fold with no per-fold search -- the same zero-per-fold-search budget EEGNetv4/ShallowFBCSPNet/
ATCNet receive in run_dl_loso_sweep.py (fixed lr=1e-3, weight_decay=1e-4, batch_size=64 there too).
Both architectures' hyperparameters were chosen once, in advance, not tuned per fold or per target
subject; this is disclosed explicitly so the comparison's search-budget symmetry is stated, not
assumed.

Resumable: per-fold resumable CSVs, mirroring run_dl_loso_sweep.py's pattern exactly.

    python -m paper.eval_spdnet_matched_protocol                      # bci2a + physionet30, full grid
    python -m paper.eval_spdnet_matched_protocol --physionet-full      # physionet_full, matching
                                                                        # eval_e1_eegnet_physionet_full.py's
                                                                        # reduced scope (seeds 42/0/123,
                                                                        # default window, EA on only)
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from pyriemann.estimation import Covariances
from sklearn.metrics import cohen_kappa_score

import moabb.datasets as _moabb_datasets
if not hasattr(_moabb_datasets, "BNCI2014001"):
    _moabb_datasets.BNCI2014001 = _moabb_datasets.BNCI2014_001

from paper import config
from paper.models.spdnet import SPDNetModel
from paper.run_dl_loso_sweep import (
    load_window_subjects, set_all_seeds, physionet_subset, WINDOWS, EA_OPTIONS, SEEDS,
    load_baseline_dict,
)
from paper.stats_analysis import paired_comparison_from_dicts, holm_bonferroni, save_results

RESULTS_DIR = config.RESULTS_DIR
COV_ESTIMATOR = "lwf"
PATIENCE = 20


def covariances_for(per_subject):
    """per_subject: {subject: (X (n,ch,t), y (n,))} of raw (already EA-aligned-or-not) trials.
    Returns the same dict with X replaced by (n, ch, ch) SPD covariance matrices."""
    out = {}
    for s, (X, y) in per_subject.items():
        covs = Covariances(estimator=COV_ESTIMATOR).fit_transform(X.astype(np.float64)).astype(np.float32)
        out[s] = (covs, y)
    return out


def train_fold_spdnet(Xtr, ytr, Xval, yval, Xtest, ytest, seed, n_classes, device):
    set_all_seeds(seed)
    model = SPDNetModel(dims=config.SPDNET_DIMS, n_classes=n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.SPDNET_LR, weight_decay=config.SPDNET_WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr).long()
    Xval_t = torch.from_numpy(Xval).float().to(device)
    Xtest_t = torch.from_numpy(Xtest).float().to(device)

    n = Xtr_t.shape[0]
    batch_size = config.SPDNET_BATCH_SIZE
    best_val_acc, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(config.SPDNET_EPOCHS):
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
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(Xtest_t).argmax(dim=1).cpu().numpy()
    acc = (test_pred == ytest).mean()
    kappa = cohen_kappa_score(ytest, test_pred)
    return float(acc), float(kappa), epoch + 1


def _cell_key(ea, window_name, seed, dataset):
    return f"spdnet_matched_{ea}_{window_name}_seed{seed}_{dataset}"


def _fold_csv_path(cell_key):
    return os.path.join(RESULTS_DIR, f"e1_dlloso_{cell_key}.csv")


def _load_done_folds(cell_key):
    path = _fold_csv_path(cell_key)
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {int(r["held_out_subject"]): {"accuracy": float(r["accuracy"]), "kappa": float(r["kappa"])}
                for r in csv.DictReader(f)}


def _append_fold_csv(cell_key, held_out, accuracy, kappa):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = _fold_csv_path(cell_key)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "accuracy", "kappa"])
        if write_header:
            writer.writeheader()
        writer.writerow({"held_out_subject": held_out, "accuracy": accuracy, "kappa": kappa})


def run_cell(ea, window_name, seed, dataset, subjects, device, tmin_tmax=None):
    cell_key = _cell_key(ea, window_name, seed, dataset)
    done = _load_done_folds(cell_key)
    if len(done) >= len(subjects):
        return done, cell_key

    tmin, tmax = tmin_tmax if tmin_tmax is not None else WINDOWS[window_name]
    per_subject, label_encoder = load_window_subjects(dataset if dataset != "physionet_full" else "physionet",
                                                        tmin, tmax, ea, subjects)
    per_subject = covariances_for(per_subject)
    n_classes = len(label_encoder.classes_)
    available = [s for s in subjects if s in per_subject]

    for held_out in available:
        if held_out in done:
            continue
        source_subjects = [s for s in available if s != held_out]
        val_subject = max(source_subjects)
        train_subjects = [s for s in source_subjects if s != val_subject]

        Xtr = np.concatenate([per_subject[s][0] for s in train_subjects], axis=0)
        ytr = np.concatenate([per_subject[s][1] for s in train_subjects], axis=0)
        Xval, yval = per_subject[val_subject]
        Xtest, ytest = per_subject[held_out]

        t0 = time.time()
        acc, kappa, n_epochs = train_fold_spdnet(Xtr, ytr, Xval, yval, Xtest, ytest, seed, n_classes, device)
        elapsed = time.time() - t0
        print(f"[{cell_key}] held_out={held_out} acc={acc*100:.2f}% kappa={kappa:.3f} "
              f"epochs={n_epochs} ({elapsed:.1f}s)", flush=True)
        _append_fold_csv(cell_key, held_out, acc, kappa)
        done[held_out] = {"accuracy": acc, "kappa": kappa}
    return done, cell_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--physionet-full", action="store_true",
                         help="Run only the full-PhysioNet cell, matching "
                              "eval_e1_eegnet_physionet_full.py's reduced scope.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary_rows = []
    original_tmin, original_tmax = config.TMIN, config.TMAX
    try:
        if args.physionet_full:
            seeds = [42, 0, 123]
            subjects = sorted(config.PRETRAIN_SUBJECTS)
            seed_accs, seed42_per_subject = [], None
            for seed in seeds:
                done, cell_key = run_cell("ea", "default", seed, "physionet_full", subjects, device,
                                           tmin_tmax=(0.0, 3.0))
                accs = np.array([v["accuracy"] for v in done.values()])
                seed_accs.append(accs.mean())
                if seed == 42:
                    seed42_per_subject = {s: v["accuracy"] for s, v in done.items()}
            mean_across_seeds = float(np.mean(seed_accs))
            std_across_seeds = float(np.std(seed_accs))
            summary_rows.append({
                "dataset": "physionet_full", "arch": "spdnet_matched", "ea": "ea", "window": "default",
                "seed42_accuracy": seed_accs[0], "mean_accuracy_across_seeds": mean_across_seeds,
                "std_accuracy_across_seeds": std_across_seeds, "n_seeds": len(seeds),
                "n_subjects": len(subjects),
            })
            print(f"=== [physionet_full/spdnet_matched/ea=ea/default] seed42={seed_accs[0]*100:.2f}% "
                  f"mean({len(seeds)} seeds)={mean_across_seeds*100:.2f}% +/- {std_across_seeds*100:.2f}% ===",
                  flush=True)
        else:
            bci2a_subjects = config.EVAL_SUBJECTS
            pn_subjects = physionet_subset()
            dataset_subjects = {"bci2a": bci2a_subjects, "physionet30": pn_subjects}
            for dataset in ["bci2a", "physionet30"]:
                subjects = dataset_subjects[dataset]
                for ea in EA_OPTIONS:
                    for window_name in WINDOWS:
                        seed_accs, seed42_per_subject = [], None
                        for seed in SEEDS:
                            done, cell_key = run_cell(ea, window_name, seed, dataset, subjects, device)
                            accs = np.array([v["accuracy"] for v in done.values()])
                            seed_accs.append(accs.mean())
                            if seed == 42:
                                seed42_per_subject = {s: v["accuracy"] for s, v in done.items()}
                        mean_across_seeds = float(np.mean(seed_accs))
                        std_across_seeds = float(np.std(seed_accs))
                        summary_rows.append({
                            "dataset": dataset, "arch": "spdnet_matched", "ea": ea, "window": window_name,
                            "seed42_accuracy": seed_accs[0],
                            "mean_accuracy_across_seeds": mean_across_seeds,
                            "std_accuracy_across_seeds": std_across_seeds, "n_seeds": len(SEEDS),
                            "n_subjects": len(subjects),
                        })
                        print(f"=== [{dataset}/spdnet_matched/ea={ea}/{window_name}] "
                              f"seed42={seed_accs[0]*100:.2f}% mean(5 seeds)={mean_across_seeds*100:.2f}% "
                              f"+/- {std_across_seeds*100:.2f}% ===", flush=True)
    finally:
        config.TMIN, config.TMAX = original_tmin, original_tmax

    suffix = "_physionet_full" if args.physionet_full else ""
    summary_path = os.path.join(RESULTS_DIR, f"e1_spdnet_matched_summary{suffix}.csv")
    with open(summary_path, "w", newline="") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"\nWrote {len(summary_rows)} summary rows to {summary_path}")


if __name__ == "__main__":
    main()
