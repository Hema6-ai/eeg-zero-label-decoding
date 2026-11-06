"""TMLR final-pass Phase 1: same comparisons as paper/stats_spdnet_matched_protocol.py (paired
Wilcoxon, Holm family, bootstrap CI, rank-biserial effect size, vs. EEGNetv4/EA+FgMDM/MEKT/
ensemble, plus the protocol-effect isolation vs. original-protocol SPDNet), but for the
hyperparameter-tuned matched-protocol SPDNet configuration selected by
paper/tune_spdnet_matched_protocol.py and run by paper/eval_spdnet_matched_protocol_tuned.py.
Also adds one further comparison: tuned vs. untuned matched-protocol SPDNet, to check whether
tuning itself changed anything.

Primary comparison cells only (seed 42, EA, default window for BCI-IV 2a/PhysioNet30; the
seeds {42,0,123} average for PhysioNet-full) -- the tuned run does not repeat the full
5-seed x 2EA x 2window grid (see eval_spdnet_matched_protocol_tuned.py's docstring).

    python -m paper.stats_spdnet_tuned_protocol
"""
import csv
import os

from paper import config
from paper.run_dl_loso_sweep import load_baseline_dict
from paper.stats_analysis import paired_comparison_from_dicts, holm_bonferroni, save_results

RESULTS_DIR = config.RESULTS_DIR
EXPECTED_N = {"bci2a": 9, "physionet30": 30, "physionet_full": 109}


def load_fold_csv(cell_key):
    path = os.path.join(RESULTS_DIR, f"e1_dlloso_{cell_key}.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return {int(r["held_out_subject"]): float(r["accuracy"]) for r in csv.DictReader(f)}


def load_nested_ensemble_dict(path, restrict_to=None):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        d = {int(r["held_out_subject"]): float(r["ensemble_nested_accuracy"]) for r in csv.DictReader(f)}
    if restrict_to is not None:
        d = {s: v for s, v in d.items() if s in restrict_to}
    return d


def load_seed_averaged_dict(cell_key_prefix, seeds):
    per_seed = []
    for seed in seeds:
        d = load_fold_csv(f"{cell_key_prefix}_seed{seed}_physionet_full")
        if d is None:
            return None
        per_seed.append(d)
    subjects = set(per_seed[0].keys())
    for d in per_seed[1:]:
        subjects &= set(d.keys())
    return {s: sum(d[s] for d in per_seed) / len(per_seed) for s in subjects}


def load_eegnetv4_seed_avg_dict(path, restrict_to=None):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        d = {int(r["held_out_subject"]): float(r["seed_avg_accuracy"]) for r in csv.DictReader(f)}
    if restrict_to is not None:
        d = {s: v for s, v in d.items() if s in restrict_to}
    return d


def main():
    all_results = []

    datasets_cells = {
        "bci2a": {
            "spdnet_tuned": "spdnet_tuned_ea_default_seed42_bci2a",
            "spdnet_matched": "spdnet_matched_ea_default_seed42_bci2a",
            "eegnetv4": "eegnetv4_ea_default_seed42_bci2a",
            "original_protocol_file": "zero_shot_spdnet",
            "baselines": {
                "EA+FgMDM": os.path.join(RESULTS_DIR, "factorial_fgmdm_t0.0_3.0_lwf.csv"),
                "MEKT": os.path.join(RESULTS_DIR, "zero_shot_mekt.csv"),
                "nested ensemble": os.path.join(RESULTS_DIR, "e4_nested_selection_bci2a_reducedgrid.csv"),
            },
        },
        "physionet30": {
            "spdnet_tuned": "spdnet_tuned_ea_default_seed42_physionet30",
            "spdnet_matched": "spdnet_matched_ea_default_seed42_physionet30",
            "eegnetv4": "eegnetv4_ea_default_seed42_physionet30",
            "original_protocol_file": "physionet_loso_spdnet",
            "baselines": {
                "EA+FgMDM": os.path.join(RESULTS_DIR, "physionet_comprehensive_ea_fgmdm.csv"),
                "MEKT": os.path.join(RESULTS_DIR, "physionet_comprehensive_mekt_default.csv"),
                "full ensemble": os.path.join(RESULTS_DIR, "physionet_full_pipeline_109.csv"),
            },
        },
        "physionet_full": {
            "spdnet_tuned": None,  # seed-averaged below
            "spdnet_matched": None,
            "eegnetv4": None,
            "original_protocol_file": "physionet_loso_spdnet",
            "baselines": {
                "EA+FgMDM": os.path.join(RESULTS_DIR, "physionet_comprehensive_ea_fgmdm.csv"),
                "MEKT": os.path.join(RESULTS_DIR, "physionet_comprehensive_mekt_default.csv"),
                "full ensemble": os.path.join(RESULTS_DIR, "physionet_full_pipeline_109.csv"),
            },
        },
    }

    for dataset, info in datasets_cells.items():
        if dataset == "physionet_full":
            spdnet_tuned = load_seed_averaged_dict("spdnet_tuned_ea_default", [42, 0, 123])
            spdnet_matched = load_seed_averaged_dict("spdnet_matched_ea_default", [42, 0, 123])
        else:
            spdnet_tuned = load_fold_csv(info["spdnet_tuned"])
            spdnet_matched = load_fold_csv(info["spdnet_matched"])
        if spdnet_tuned is None:
            print(f"[{dataset}] tuned matched-protocol SPDNet results not found yet, skipping")
            continue
        subjects = set(spdnet_tuned.keys())
        print(f"[{dataset}] tuned matched-protocol SPDNet: n={len(spdnet_tuned)} subjects")
        if len(spdnet_tuned) < EXPECTED_N[dataset]:
            print(f"[{dataset}] INCOMPLETE: expected {EXPECTED_N[dataset]}, have {len(spdnet_tuned)} "
                  f"-- skipping comparison for now")
            continue

        if dataset == "physionet_full":
            eegnetv4 = load_eegnetv4_seed_avg_dict(
                os.path.join(RESULTS_DIR, "e1_eegnetv4_physionet_full_summary.csv"), restrict_to=subjects)
        else:
            eegnetv4 = load_fold_csv(info["eegnetv4"])
        family = []
        if eegnetv4:
            common = set(eegnetv4.keys()) & subjects
            r = paired_comparison_from_dicts(
                f"[{dataset}] tuned SPDNet vs EEGNetv4",
                {s: eegnetv4[s] for s in common}, {s: spdnet_tuned[s] for s in common})
            if r:
                family.append(r)

        for name, path in info["baselines"].items():
            if name == "nested ensemble":
                baseline = load_nested_ensemble_dict(path, restrict_to=subjects)
            else:
                baseline = load_baseline_dict(path, restrict_to=subjects)
            if baseline is None:
                print(f"[{dataset}] baseline {name} not found at {path}")
                continue
            common = set(baseline.keys()) & subjects
            r = paired_comparison_from_dicts(
                f"[{dataset}] tuned SPDNet vs {name}",
                baseline, {s: spdnet_tuned[s] for s in common})
            if r:
                family.append(r)

        family = holm_bonferroni(family)
        all_results.extend(family)
        for r in family:
            print(f"  {r['comparison']}: diff={r['mean_diff_pp']:+.2f}pp p={r['wilcoxon_p']:.4f} "
                  f"holm_sig={r['holm_significant']}")

        # Tuned vs untuned (does the search itself change anything?) -- single comparison, not
        # part of the 4-comparison Holm family above.
        if spdnet_matched is not None:
            common = set(spdnet_matched.keys()) & subjects
            r = paired_comparison_from_dicts(
                f"[{dataset}] tuned SPDNet vs untuned matched-protocol SPDNet",
                {s: spdnet_matched[s] for s in common}, {s: spdnet_tuned[s] for s in common})
            if r:
                r["holm_significant"] = None
                all_results.append(r)
                print(f"  {r['comparison']}: diff={r['mean_diff_pp']:+.2f}pp p={r['wilcoxon_p']:.4f}")

        # Protocol-effect isolation: tuned matched-protocol vs original population-swap SPDNet.
        orig_path = os.path.join(RESULTS_DIR, f"{info['original_protocol_file']}.csv")
        original = load_baseline_dict(orig_path, restrict_to=subjects) if os.path.exists(orig_path) else None
        if original is not None:
            common = set(original.keys()) & subjects
            r = paired_comparison_from_dicts(
                f"[{dataset}] tuned SPDNet vs original-protocol SPDNet",
                original, {s: spdnet_tuned[s] for s in common})
            if r:
                r["holm_significant"] = None
                all_results.append(r)
                print(f"  {r['comparison']}: diff={r['mean_diff_pp']:+.2f}pp p={r['wilcoxon_p']:.4f}")

    if all_results:
        save_results(all_results, "stats_spdnet_tuned_protocol.csv")
        print(f"\nWrote {len(all_results)} rows to "
              f"{os.path.join(RESULTS_DIR, 'stats_spdnet_tuned_protocol.csv')}")


if __name__ == "__main__":
    main()
