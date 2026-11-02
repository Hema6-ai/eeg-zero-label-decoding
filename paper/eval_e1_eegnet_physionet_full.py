"""TMLR revision round 2, item 1 of the Phase-5 follow-up directive: EEGNetv4+EA, trained
source-only LOSO, on ALL AVAILABLE PhysioNet subjects (not the fixed 30-subject subset
`paper/run_dl_loso_sweep.py` used for the rest of E1) -- the strongest CNN this project has
tested, at the full scale the rest of the PhysioNet comprehensive comparison uses, directly
testing the "architectures are mostly tied" claim against the dataset with the most statistical
power.

Protocol: default window (0.0, 3.0), EA on, seeds {42, 0, 123} (a subset of E1's full 5-seed
list -- 3 seeds chosen here specifically per this item's own request, not a shortcut: the other
E1 cells already used 5; this one is new and large enough on its own that 3 well-separated seeds
is a reasonable, disclosed scope decision). Reuses paper/run_dl_loso_sweep.py's train_fold /
load_window_subjects / per-fold resumable-CSV machinery directly rather than reimplementing it.

    python -m paper.eval_e1_eegnet_physionet_full
"""
import csv
import os

import numpy as np
import torch

import moabb.datasets as _moabb_datasets
if not hasattr(_moabb_datasets, "BNCI2014001"):
    _moabb_datasets.BNCI2014001 = _moabb_datasets.BNCI2014_001

from paper import config
from paper.run_dl_loso_sweep import load_window_subjects, train_fold, set_all_seeds
from paper.stats_analysis import paired_comparison_from_dicts, holm_bonferroni, save_results

RESULTS_DIR = config.RESULTS_DIR
SEEDS = [42, 0, 123]
ARCH = "eegnetv4"
WINDOW = (0.0, 3.0)
DATASET = "physionet_full"


def _cell_key(seed):
    return f"e1_eegnetv4_ea_default_seed{seed}_physionet_full"


def _fold_csv_path(cell_key):
    return os.path.join(RESULTS_DIR, f"{cell_key}.csv")


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


def load_baseline_dict(path, restrict_to=None):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        subj_col = "held_out_subject" if "held_out_subject" in reader.fieldnames else "held_out"
        d = {int(r[subj_col]): float(r["accuracy"]) for r in reader}
    if restrict_to is not None:
        d = {s: v for s, v in d.items() if s in restrict_to}
    return d


def main(subjects=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    subjects = subjects or config.PRETRAIN_SUBJECTS
    print(f"Target: ALL {len(subjects)} PhysioNet subjects (not the n=30 subset)", flush=True)

    tmin, tmax = WINDOW
    # load_window_subjects dispatches on "bci2a" vs anything else (-> PhysioNet path).
    per_subject, label_encoder = load_window_subjects("physionet", tmin, tmax, "ea", subjects)
    n_classes = len(label_encoder.classes_)
    available = sorted(per_subject.keys())
    missing = sorted(set(subjects) - set(available))
    print(f"Loaded {len(available)}/{len(subjects)} subjects at window {WINDOW}, EA on. "
          f"Missing: {missing}", flush=True)

    seed_accs_per_subject = {seed: {} for seed in SEEDS}
    for seed in SEEDS:
        cell_key = _cell_key(seed)
        done = _load_done_folds(cell_key)
        for held_out in available:
            if held_out in done:
                seed_accs_per_subject[seed][held_out] = done[held_out]["accuracy"]
                continue
            source_subjects = [s for s in available if s != held_out]
            val_subject = max(source_subjects)
            train_subjects = [s for s in source_subjects if s != val_subject]

            Xtr = np.concatenate([per_subject[s][0] for s in train_subjects], axis=0)
            ytr = np.concatenate([per_subject[s][1] for s in train_subjects], axis=0)
            Xval, yval = per_subject[val_subject]
            Xtest, ytest = per_subject[held_out]

            acc, kappa, n_epochs = train_fold(ARCH, Xtr, ytr, Xval, yval, Xtest, ytest, seed,
                                               n_classes, device)
            print(f"[seed={seed}] held_out={held_out} acc={acc*100:.2f}% kappa={kappa:.3f} "
                  f"epochs={n_epochs}", flush=True)
            _append_fold_csv(cell_key, held_out, acc, kappa)
            seed_accs_per_subject[seed][held_out] = acc

    # Per-subject, seed-averaged accuracy (only over subjects with all SEEDS present).
    common_subjects = sorted(set.intersection(*[set(d) for d in seed_accs_per_subject.values()]))
    seed_avg_per_subject = {
        s: float(np.mean([seed_accs_per_subject[seed][s] for seed in SEEDS]))
        for s in common_subjects
    }

    summary_path = os.path.join(RESULTS_DIR, "e1_eegnetv4_physionet_full_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "seed_avg_accuracy"] +
                                 [f"seed{s}_accuracy" for s in SEEDS])
        writer.writeheader()
        for s in common_subjects:
            row = {"held_out_subject": s, "seed_avg_accuracy": seed_avg_per_subject[s]}
            row.update({f"seed{seed}_accuracy": seed_accs_per_subject[seed][s] for seed in SEEDS})
            writer.writerow(row)

    accs = np.array(list(seed_avg_per_subject.values()))
    print(f"\n=== EEGNetv4+EA, PhysioNet FULL (n={len(common_subjects)}): "
          f"{accs.mean()*100:.2f}% +/- {accs.std()*100:.2f}% ===", flush=True)
    print(f"Missing from all 3 seeds (not in common_subjects): "
          f"{sorted(set(available) - set(common_subjects))}", flush=True)

    # Paired comparisons vs MEKT default, FgMDM (both at full PhysioNet scale, same subjects where
    # available), and the E4 nested ensemble (only available for the n=30 subset -- compared on
    # that subset's intersection with this run, disclosed explicitly in the printed output).
    baseline_paths = {
        "EA+FgMDM (full PhysioNet)": os.path.join(RESULTS_DIR, "physionet_comprehensive_ea_fgmdm.csv"),
        "MEKT default (full PhysioNet)": os.path.join(RESULTS_DIR, "physionet_comprehensive_mekt_default.csv"),
    }
    comparison_results = []
    for name, path in baseline_paths.items():
        baseline_dict = load_baseline_dict(path, restrict_to=set(common_subjects))
        if baseline_dict is None:
            print(f"WARNING: baseline file not found: {path}", flush=True)
            continue
        result = paired_comparison_from_dicts(f"EEGNetv4+EA (full PN) vs {name}",
                                               baseline_dict, seed_avg_per_subject)
        if result is not None:
            comparison_results.append(result)

    # Nested ensemble: only computed for the PhysioNet30 subset (paper/results/
    # e4_nested_selection_physionet30_reducedgrid.csv). Compare on the intersection.
    nested_path = os.path.join(RESULTS_DIR, "e4_nested_selection_physionet30_reducedgrid.csv")
    if os.path.exists(nested_path):
        with open(nested_path, newline="") as f:
            nested_dict = {int(r["held_out_subject"]): float(r["ensemble_nested_accuracy"])
                           for r in csv.DictReader(f)}
        overlap = sorted(set(nested_dict) & set(seed_avg_per_subject))
        print(f"Nested-ensemble comparison restricted to the physionet30-subset/full-run overlap: "
              f"n={len(overlap)} subjects ({overlap})", flush=True)
        if overlap:
            result = paired_comparison_from_dicts(
                "EEGNetv4+EA (full PN, n=30-overlap) vs E4 nested ensemble (physionet30)",
                {s: nested_dict[s] for s in overlap}, {s: seed_avg_per_subject[s] for s in overlap})
            if result is not None:
                comparison_results.append(result)
    else:
        print(f"WARNING: nested-ensemble file not found: {nested_path}", flush=True)

    if comparison_results:
        comparison_results = holm_bonferroni(comparison_results)
        save_results(comparison_results, "e1_eegnetv4_physionet_full_comparisons.csv")
        for r in comparison_results:
            print(f"  {r['contrast']}: {r['mean_diff_pp']:+.2f}pp, p={r['wilcoxon_p']:.4f}, "
                  f"Holm-sig={r['holm_significant']}")


if __name__ == "__main__":
    main()
