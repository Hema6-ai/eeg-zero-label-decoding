"""TMLR revision round 2, task E5 (final piece): "the best E1 CNN" on Cho2017. E1's full sweep
(paper/run_dl_loso_sweep.py) ran EEGNetv4, ShallowFBCSPNet, and ATCNet on both BCI-IV 2a and
PhysioNet; only EEGNetv4 held up on BOTH datasets (ShallowFBCSPNet and especially ATCNet lost
significantly to every classical/DA baseline on PhysioNet -- see reports/revision_r2/
PHASE_2_PROGRESS.md for the full comparison). EEGNetv4 is therefore "the best E1 CNN" by this
paper's own evidence, not an arbitrary choice.

Protocol: identical to run_dl_loso_sweep.py's EA-aligned, default-window cell (the canonical cell
used throughout this round's comparisons) -- source-subjects-only training, LOSO, one source
subject held out per fold for early-stopping validation, 5 seeds (42, 0, 123, 21, 7). Only one
cell is run here (not the full ea/window sweep) since the task only asks for a single "best CNN"
row, matching how FgMDM/MEKT-default are also reported as single-setting rows for this dataset
(paper/eval_e5_third_dataset.py).

    python -m paper.eval_e5_best_cnn
"""
import csv
import os

import numpy as np
import torch

from paper import config
from paper.eval_e5_third_dataset import load_cho2017
from paper.run_dl_loso_sweep import SEEDS, MAX_EPOCHS, PATIENCE, train_fold

RESULTS_DIR = config.RESULTS_DIR
ARCH = "eegnetv4"


def _ea_align_per_subject(per_subject):
    from paper.align import align_per_subject
    X_only = {s: X for s, (X, y) in per_subject.items()}
    aligned = align_per_subject(X_only, "ea")
    return {s: (aligned[s].astype(np.float32), y) for s, (_, y) in per_subject.items()}


def _fold_csv_path(seed):
    return os.path.join(RESULTS_DIR, f"e5_best_cnn_{ARCH}_ea_seed{seed}.csv")


def _load_done_folds(seed):
    path = _fold_csv_path(seed)
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {int(r["held_out_subject"]): {"accuracy": float(r["accuracy"]), "kappa": float(r["kappa"])}
                for r in csv.DictReader(f)}


def _append_fold_csv(seed, held_out, accuracy, kappa):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = _fold_csv_path(seed)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "accuracy", "kappa"])
        if write_header:
            writer.writeheader()
        writer.writerow({"held_out_subject": held_out, "accuracy": accuracy, "kappa": kappa})


def main(subjects=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    subjects = subjects or list(range(1, 53))

    per_subject, label_encoder, good_subjects = load_cho2017(subjects)
    per_subject = _ea_align_per_subject(per_subject)
    n_classes = len(label_encoder.classes_)
    print(f"E5 best-CNN ({ARCH}): {len(good_subjects)}/{len(subjects)} subjects loaded, "
          f"chance=50% (2-class)", flush=True)

    seed_accs = []
    for seed in SEEDS:
        done = _load_done_folds(seed)
        for held_out in good_subjects:
            if held_out in done:
                continue
            source_subjects = [s for s in good_subjects if s != held_out]
            val_subject = max(source_subjects)
            train_subjects = [s for s in source_subjects if s != val_subject]

            Xtr = np.concatenate([per_subject[s][0] for s in train_subjects], axis=0)
            ytr = np.concatenate([per_subject[s][1] for s in train_subjects], axis=0)
            Xval, yval = per_subject[val_subject]
            Xtest, ytest = per_subject[held_out]

            acc, kappa, n_epochs = train_fold(ARCH, Xtr, ytr, Xval, yval, Xtest, ytest, seed,
                                               n_classes, device)
            print(f"[seed{seed}] held_out={held_out} acc={acc*100:.2f}% kappa={kappa:.3f} "
                  f"epochs={n_epochs}", flush=True)
            _append_fold_csv(seed, held_out, acc, kappa)
            done[held_out] = {"accuracy": acc, "kappa": kappa}

        accs = np.array([v["accuracy"] for v in done.values()])
        seed_accs.append(accs.mean())
        print(f"=== seed{seed}: {accs.mean()*100:.2f}% +/- {accs.std()*100:.2f}% "
              f"(n={len(accs)}) ===", flush=True)

    mean_across_seeds = float(np.mean(seed_accs))
    std_across_seeds = float(np.std(seed_accs))
    print(f"\n=== E5 best-CNN ({ARCH}) on Cho2017: mean(5 seeds)={mean_across_seeds*100:.2f}% "
          f"+/- {std_across_seeds*100:.2f}% (chance=50%) ===", flush=True)

    summary_path = os.path.join(RESULTS_DIR, "e5_best_cnn_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["arch", "seed42_accuracy", "mean_accuracy_across_5_seeds",
                                                "std_accuracy_across_5_seeds", "n_subjects"])
        writer.writeheader()
        writer.writerow({"arch": ARCH, "seed42_accuracy": seed_accs[0],
                          "mean_accuracy_across_5_seeds": mean_across_seeds,
                          "std_accuracy_across_5_seeds": std_across_seeds,
                          "n_subjects": len(good_subjects)})
    print(f"Wrote summary to {summary_path}")
    return mean_across_seeds, std_across_seeds


if __name__ == "__main__":
    main()
