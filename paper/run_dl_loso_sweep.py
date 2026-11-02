"""TMLR revision round 2, task E1: supervised cross-subject deep-learning baselines, filling the
"limited DL architecture selection" hole the reviewer flagged. Trains EEGNetv4, ShallowFBCSPNet,
and ATCNet (braindecode) from scratch on SOURCE SUBJECTS ONLY, LOSO, zero target data -- a
genuinely different regime from every other Family 1/2 method in this paper, which pretrains
self-supervised/contrastive on an upstream corpus and only fits a linear/MLP head zero-shot. These
are supervised end-to-end baselines, trained directly for 3/4-class motor-imagery classification,
to test whether a purpose-built supervised CNN/attention architecture -- not a self-supervised one
-- can beat MEKT/the 8-member ensemble when given the same zero-target-label constraint.

Protocol (exactly as task E1 specifies):
  - LOSO: for each held-out TARGET subject, train only on the other SOURCE subjects.
  - Early stopping uses one SOURCE subject held out as a validation subject (never the target,
    never any subject outside the current fold's source set) -- so no test-subject tuning anywhere.
  - Each (architecture, EA on/off, window) cell is trained under 5 seeds: 42, 0, 123, 21, 7.
  - Same 22-channel/128 Hz canonical input as every other method in this paper
    (paper/config.py::COMMON_CHANNELS, RESAMPLE_HZ).
  - Two datasets: BCI-IV 2a (9 subjects, full LOSO) and PhysioNet. Full 109-subject PhysioNet LOSO
    would require 3 archs x 2 EA x 2 windows x 5 seeds x 109 folds = 6,540 individual trainings --
    not feasible in a single Kaggle GPU session (9-12h limit, 30h/week quota). Per the task's own
    documented fallback ("otherwise a fixed random 30-subject subset (seed 42), and say so"), this
    script uses a fixed, seed-42-sampled 30-subject PhysioNet subset instead, and this choice is
    disclosed here and in the resulting report -- it is NOT the full 109-subject population used
    elsewhere in this paper's PhysioNet comprehensive comparison.

Resumable: every individual LOSO fold is appended to its own per-cell CSV the moment it finishes
(mirrors paper/eval_factorial.py's `_append_fold_csv`/`_load_done_folds` pattern exactly), so an
interrupted Kaggle session can be restarted and will skip already-completed folds.

Designed to run standalone on Kaggle (GPU, internet enabled for the MOABB/PhysioNet download):
    python -m paper.run_dl_loso_sweep
Pass a subset of args via environment variables to split the sweep across multiple Kaggle kernel
pushes (each kernel session is time-limited): E1_ARCHS, E1_DATASETS (comma-separated).
"""
import csv
import importlib
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

import moabb.datasets as _moabb_datasets
if not hasattr(_moabb_datasets, "BNCI2014001"):
    # braindecode 1.2.0's top-level import chain still names the old (pre-1.0) moabb dataset
    # class; moabb >=1.x dropped the alias. This is a pure import-compatibility shim -- it does
    # not change what dataset class this script itself uses (paper/data.py uses BNCI2014_001
    # directly throughout, as everywhere else in this repo).
    _moabb_datasets.BNCI2014001 = _moabb_datasets.BNCI2014_001

from paper import config
from paper.stats_analysis import paired_comparison_from_dicts, holm_bonferroni, save_results

RESULTS_DIR = config.RESULTS_DIR
SEEDS = [42, 0, 123, 21, 7]
WINDOWS = {"default": (0.0, 3.0), "alt": (0.3, 2.5)}
EA_OPTIONS = ["none", "ea"]
ARCHS = list(filter(None, os.environ.get("E1_ARCHS", "eegnetv4,shallowfbcspnet,atcnet").split(",")))
DATASETS = list(filter(None, os.environ.get("E1_DATASETS", "bci2a,physionet30").split(",")))
PHYSIONET_SUBSET_SEED = 42
PHYSIONET_SUBSET_SIZE = 30

MAX_EPOCHS = 150
PATIENCE = 20
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(arch, n_chans, n_times, n_classes):
    from braindecode.models import EEGNetv4, ShallowFBCSPNet, ATCNet
    if arch == "eegnetv4":
        return EEGNetv4(n_chans=n_chans, n_outputs=n_classes, n_times=n_times)
    if arch == "shallowfbcspnet":
        return ShallowFBCSPNet(n_chans=n_chans, n_outputs=n_classes, n_times=n_times,
                                final_conv_length="auto")
    if arch == "atcnet":
        return ATCNet(n_chans=n_chans, n_outputs=n_classes, n_times=n_times)
    raise ValueError(arch)


def load_window_subjects(dataset, tmin, tmax, ea, subjects):
    """Reloads data at a given window + alignment choice. Mirrors paper/eval_factorial.py's
    load_window_data pattern: config.TMIN/TMAX are read at MotorImagery-paradigm-construction
    time, so paper.data must be reloaded after changing them."""
    config.TMIN, config.TMAX = tmin, tmax
    import paper.data as data_mod
    importlib.reload(data_mod)
    if dataset == "bci2a":
        per_subject, label_encoder = data_mod.load_bnci_all(subjects, align=ea)
        return per_subject, label_encoder
    per_subject, label_encoder, good_subjects = data_mod.load_physionet_zeroshot(subjects, align=ea)
    # PhysioNet subjects' raw recordings resample to very slightly different trial lengths at some
    # (tmin,tmax) windows (off-by-one-sample edge case, found while building task E3) -- crop every
    # subject to the shortest common n_times at this window so concatenation doesn't fail.
    n_times_per_subject = {s: X.shape[-1] for s, (X, y) in per_subject.items()}
    min_n_times, max_n_times = min(n_times_per_subject.values()), max(n_times_per_subject.values())
    if max_n_times - min_n_times > 1:
        raise RuntimeError(f"Unexpected >1-sample trial-length mismatch across PhysioNet subjects: "
                            f"{n_times_per_subject}")
    if min_n_times != max_n_times:
        per_subject = {s: (X[..., :min_n_times], y) for s, (X, y) in per_subject.items()}
    return per_subject, label_encoder


def physionet_subset(all_subjects=None):
    all_subjects = all_subjects or config.PRETRAIN_SUBJECTS
    rng = random.Random(PHYSIONET_SUBSET_SEED)
    return sorted(rng.sample(list(all_subjects), PHYSIONET_SUBSET_SIZE))


def train_fold(arch, Xtr, ytr, Xval, yval, Xtest, ytest, seed, n_classes, device):
    set_all_seeds(seed)
    n_chans, n_times = Xtr.shape[1], Xtr.shape[2]
    model = build_model(arch, n_chans, n_times, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr).long()
    Xval_t = torch.from_numpy(Xval).float().to(device)
    yval_t = torch.from_numpy(yval).long()
    Xtest_t = torch.from_numpy(Xtest).float().to(device)

    n = Xtr_t.shape[0]
    best_val_acc, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
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
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(ytest, test_pred)
    return float(acc), float(kappa), epoch + 1


def _cell_key(arch, ea, window_name, seed, dataset):
    return f"{arch}_{ea}_{window_name}_seed{seed}_{dataset}"


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


def run_cell(arch, ea, window_name, seed, dataset, subjects, device):
    cell_key = _cell_key(arch, ea, window_name, seed, dataset)
    done = _load_done_folds(cell_key)
    if len(done) >= len(subjects):
        return done

    tmin, tmax = WINDOWS[window_name]
    per_subject, label_encoder = load_window_subjects(dataset, tmin, tmax, ea, subjects)
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
        acc, kappa, n_epochs = train_fold(arch, Xtr, ytr, Xval, yval, Xtest, ytest, seed, n_classes, device)
        elapsed = time.time() - t0
        print(f"[{cell_key}] held_out={held_out} acc={acc*100:.2f}% kappa={kappa:.3f} "
              f"epochs={n_epochs} ({elapsed:.1f}s)", flush=True)
        _append_fold_csv(cell_key, held_out, acc, kappa)
        done[held_out] = {"accuracy": acc, "kappa": kappa}
    return done


def load_baseline_dict(path, restrict_to=None):
    """Baseline per-subject CSVs in this repo use either 'held_out_subject' or (older files)
    'held_out' as the subject-id column name -- handle both."""
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        subj_col = "held_out_subject" if "held_out_subject" in reader.fieldnames else "held_out"
        d = {int(r[subj_col]): float(r["accuracy"]) for r in reader}
    if restrict_to is not None:
        d = {s: v for s, v in d.items() if s in restrict_to}
    return d


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    bci2a_subjects = config.EVAL_SUBJECTS
    pn_subjects = physionet_subset()
    print(f"PhysioNet fixed 30-subject subset (seed={PHYSIONET_SUBSET_SEED}): {pn_subjects}", flush=True)

    dataset_subjects = {"bci2a": bci2a_subjects, "physionet30": pn_subjects}
    baseline_paths = {
        "bci2a": {
            "EA+FgMDM": os.path.join(RESULTS_DIR, "factorial_fgmdm_t0.0_3.0_lwf.csv"),
            "MEKT": os.path.join(RESULTS_DIR, "zero_shot_mekt.csv"),
            "8-member ensemble": os.path.join(RESULTS_DIR, "zero_shot_mekt_ensemble.csv"),
        },
        "physionet30": {
            "EA+FgMDM": os.path.join(RESULTS_DIR, "physionet_comprehensive_ea_fgmdm.csv"),
            "MEKT": os.path.join(RESULTS_DIR, "physionet_comprehensive_mekt_default.csv"),
            "8-member ensemble": os.path.join(RESULTS_DIR, "physionet_full_pipeline_109.csv"),
        },
    }

    summary_rows = []
    comparison_results = []
    original_tmin, original_tmax = config.TMIN, config.TMAX
    try:
        for dataset in DATASETS:
            subjects = dataset_subjects[dataset]
            for arch in ARCHS:
                for ea in EA_OPTIONS:
                    for window_name in WINDOWS:
                        seed_accs = []
                        seed42_per_subject = None
                        for seed in SEEDS:
                            folds = run_cell(arch, ea, window_name, seed, dataset, subjects, device)
                            accs = np.array([v["accuracy"] for v in folds.values()])
                            seed_accs.append(accs.mean())
                            if seed == 42:
                                seed42_per_subject = {s: v["accuracy"] for s, v in folds.items()}

                        mean_across_seeds = float(np.mean(seed_accs))
                        std_across_seeds = float(np.std(seed_accs))
                        summary_rows.append({
                            "dataset": dataset, "arch": arch, "ea": ea, "window": window_name,
                            "seed42_accuracy": seed_accs[0],
                            "mean_accuracy_across_5_seeds": mean_across_seeds,
                            "std_accuracy_across_5_seeds": std_across_seeds,
                            "n_subjects": len(subjects),
                        })
                        print(f"=== [{dataset}/{arch}/ea={ea}/{window_name}] "
                              f"seed42={seed_accs[0]*100:.2f}% mean(5 seeds)={mean_across_seeds*100:.2f}% "
                              f"+/- {std_across_seeds*100:.2f}% ===", flush=True)

                        for baseline_name, path in baseline_paths[dataset].items():
                            baseline_dict = load_baseline_dict(path, restrict_to=set(subjects))
                            if baseline_dict is None or seed42_per_subject is None:
                                continue
                            result = paired_comparison_from_dicts(
                                f"{dataset}/{arch}/ea={ea}/{window_name} vs {baseline_name}",
                                baseline_dict, seed42_per_subject)
                            if result is not None:
                                result.update({"dataset": dataset, "arch": arch, "ea": ea,
                                                "window": window_name, "baseline": baseline_name})
                                comparison_results.append(result)
    finally:
        config.TMIN, config.TMAX = original_tmin, original_tmax

    summary_path = os.path.join(RESULTS_DIR, "e1_dlloso_summary.csv")
    with open(summary_path, "w", newline="") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"\nWrote {len(summary_rows)} summary rows to {summary_path}")

    if comparison_results:
        comparison_results = holm_bonferroni(comparison_results)
        save_results(comparison_results, "e1_dlloso_comparisons.csv")
        n_sig = sum(1 for r in comparison_results if r.get("holm_significant"))
        print(f"Wrote {len(comparison_results)} comparisons "
              f"({n_sig} Holm-significant at alpha=0.05) to "
              f"{os.path.join(RESULTS_DIR, 'e1_dlloso_comparisons.csv')}")


if __name__ == "__main__":
    main()
