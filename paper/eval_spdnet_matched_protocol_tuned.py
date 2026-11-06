"""TMLR final-pass Phase 1: re-run the matched-protocol SPDNet fairness check (Section 13.3) with
the hyperparameter configuration selected by paper/tune_spdnet_matched_protocol.py (selected on
PhysioNet30 source-validation accuracy only, never target accuracy). Mirrors
paper/eval_spdnet_matched_protocol.py exactly except for (a) the dims/lr override and (b) a
"_tuned" cell-key suffix, so these results never overwrite the original, untuned-config files --
both are kept, since the manuscript reports whichever one is now correct per the robustness check.

    python -m paper.eval_spdnet_matched_protocol_tuned --dims 22,22 --depth-label 1 --lr 0.001
    python -m paper.eval_spdnet_matched_protocol_tuned --dims 22,22 --depth-label 1 --lr 0.001 --physionet-full
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
)

RESULTS_DIR = config.RESULTS_DIR
COV_ESTIMATOR = "lwf"
PATIENCE = 20


def covariances_for(per_subject):
    out = {}
    for s, (X, y) in per_subject.items():
        covs = Covariances(estimator=COV_ESTIMATOR).fit_transform(X.astype(np.float64)).astype(np.float32)
        out[s] = (covs, y)
    return out


def train_fold_spdnet(Xtr, ytr, Xval, yval, Xtest, ytest, seed, n_classes, device, dims, lr):
    set_all_seeds(seed)
    model = SPDNetModel(dims=dims, n_classes=n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.SPDNET_WEIGHT_DECAY)
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
    return f"spdnet_tuned_{ea}_{window_name}_seed{seed}_{dataset}"


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


def run_cell(ea, window_name, seed, dataset, subjects, device, dims, lr, tmin_tmax=None):
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
        acc, kappa, n_epochs = train_fold_spdnet(Xtr, ytr, Xval, yval, Xtest, ytest, seed, n_classes,
                                                   device, dims, lr)
        elapsed = time.time() - t0
        print(f"[{cell_key}] held_out={held_out} acc={acc*100:.2f}% kappa={kappa:.3f} "
              f"epochs={n_epochs} ({elapsed:.1f}s)", flush=True)
        _append_fold_csv(cell_key, held_out, acc, kappa)
        done[held_out] = {"accuracy": acc, "kappa": kappa}
    return done, cell_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=str, required=True, help="comma-separated, e.g. 22,19,16")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--physionet-full", action="store_true")
    args = parser.parse_args()
    dims = tuple(int(x) for x in args.dims.split(","))
    lr = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, dims={dims}, lr={lr}", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary_rows = []
    original_tmin, original_tmax = config.TMIN, config.TMAX
    try:
        if args.physionet_full:
            seeds = [42, 0, 123]
            subjects = sorted(config.PRETRAIN_SUBJECTS)
            seed_accs = []
            for seed in seeds:
                done, cell_key = run_cell("ea", "default", seed, "physionet_full", subjects, device,
                                           dims, lr, tmin_tmax=(0.0, 3.0))
                accs = np.array([v["accuracy"] for v in done.values()])
                seed_accs.append(accs.mean())
            mean_across_seeds = float(np.mean(seed_accs))
            std_across_seeds = float(np.std(seed_accs))
            summary_rows.append({
                "dataset": "physionet_full", "arch": "spdnet_tuned", "ea": "ea", "window": "default",
                "dims": str(dims), "lr": lr,
                "seed42_accuracy": seed_accs[0], "mean_accuracy_across_seeds": mean_across_seeds,
                "std_accuracy_across_seeds": std_across_seeds, "n_seeds": len(seeds),
                "n_subjects": len(subjects),
            })
            print(f"=== [physionet_full/spdnet_tuned/ea=ea/default] seed42={seed_accs[0]*100:.2f}% "
                  f"mean({len(seeds)} seeds)={mean_across_seeds*100:.2f}% +/- {std_across_seeds*100:.2f}% ===",
                  flush=True)
        else:
            # Only the primary comparison cell per dataset (seed 42, EA, default window) --
            # the one Table 8 actually reports against EEGNetv4/EA+FgMDM/MEKT/ensemble. Re-running
            # the full 5-seed x 2EA x 2window grid with the tuned config would cost as much compute
            # as the original search itself, which is out of scope for a "small" robustness check.
            bci2a_subjects = config.EVAL_SUBJECTS
            pn_subjects = physionet_subset()
            dataset_subjects = {"bci2a": bci2a_subjects, "physionet30": pn_subjects}
            for dataset in ["bci2a", "physionet30"]:
                subjects = dataset_subjects[dataset]
                ea, window_name, seed = "ea", "default", 42
                done, cell_key = run_cell(ea, window_name, seed, dataset, subjects, device, dims, lr)
                accs = np.array([v["accuracy"] for v in done.values()])
                acc_mean = float(accs.mean())
                summary_rows.append({
                    "dataset": dataset, "arch": "spdnet_tuned", "ea": ea, "window": window_name,
                    "dims": str(dims), "lr": lr,
                    "seed42_accuracy": acc_mean,
                    "mean_accuracy_across_seeds": acc_mean,
                    "std_accuracy_across_seeds": 0.0, "n_seeds": 1,
                    "n_subjects": len(subjects),
                })
                print(f"=== [{dataset}/spdnet_tuned/ea={ea}/{window_name}] "
                      f"seed42={acc_mean*100:.2f}% (primary comparison cell only) ===", flush=True)
    finally:
        config.TMIN, config.TMAX = original_tmin, original_tmax

    suffix = "_physionet_full" if args.physionet_full else ""
    summary_path = os.path.join(RESULTS_DIR, f"e1_spdnet_tuned_summary{suffix}.csv")
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(summary_rows)
    print(f"\nWrote {len(summary_rows)} summary rows to {summary_path}")


if __name__ == "__main__":
    main()
