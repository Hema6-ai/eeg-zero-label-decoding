"""Reviewer c1um's requested matched-protocol SPDNet comparisons: paired Wilcoxon, Holm family,
bootstrap CI, and rank-biserial effect size for matched-protocol SPDNet vs. EEGNetv4, MEKT
(default), FgMDM, and the nested/full ensemble, on each dataset where the matched-protocol run
completed; plus matched-protocol SPDNet vs. the original, population-swap-protocol SPDNet, to
isolate the protocol effect itself.

    python -m paper.stats_spdnet_matched_protocol
"""
import csv
import os

from paper import config
from paper.run_dl_loso_sweep import load_baseline_dict
from paper.stats_analysis import paired_comparison_from_dicts, holm_bonferroni, save_results

RESULTS_DIR = config.RESULTS_DIR


def load_fold_csv(cell_key):
    path = os.path.join(RESULTS_DIR, f"e1_dlloso_{cell_key}.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return {int(r["held_out_subject"]): float(r["accuracy"]) for r in csv.DictReader(f)}


def load_nested_ensemble_dict(path, restrict_to=None):
    """paper/results/e4_nested_selection_*.csv uses 'ensemble_nested_accuracy', not 'accuracy'."""
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        d = {int(r["held_out_subject"]): float(r["ensemble_nested_accuracy"]) for r in csv.DictReader(f)}
    if restrict_to is not None:
        d = {s: v for s, v in d.items() if s in restrict_to}
    return d


def load_seed_averaged_dict(cell_key_prefix, seeds):
    """Average per-subject accuracy across seeds for the matched-protocol SPDNet physionet_full
    cells, matching how e1_eegnetv4_physionet_full_summary.csv's seed_avg_accuracy column and the
    manuscript text report EEGNetv4's full-PhysioNet number."""
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


EXPECTED_N = {"bci2a": 9, "physionet30": 30, "physionet_full": 109}


def main():
    all_results = []

    datasets_cells = {
        "bci2a": {
            "spdnet_matched": "spdnet_matched_ea_default_seed42_bci2a",
            "eegnetv4": "eegnetv4_ea_default_seed42_bci2a",
            "spdnet_original": None,  # handled separately below (different CSV format)
            "baselines": {
                "EA+FgMDM": os.path.join(RESULTS_DIR, "factorial_fgmdm_t0.0_3.0_lwf.csv"),
                "MEKT": os.path.join(RESULTS_DIR, "zero_shot_mekt.csv"),
                "nested ensemble": os.path.join(RESULTS_DIR, "e4_nested_selection_bci2a_reducedgrid.csv"),
            },
        },
        "physionet30": {
            "spdnet_matched": "spdnet_matched_ea_default_seed42_physionet30",
            "eegnetv4": "eegnetv4_ea_default_seed42_physionet30",
            "spdnet_original": None,
            "baselines": {
                "EA+FgMDM": os.path.join(RESULTS_DIR, "physionet_comprehensive_ea_fgmdm.csv"),
                "MEKT": os.path.join(RESULTS_DIR, "physionet_comprehensive_mekt_default.csv"),
                "full ensemble": os.path.join(RESULTS_DIR, "physionet_full_pipeline_109.csv"),
            },
        },
    }

    datasets_cells["physionet_full"] = {
        "spdnet_matched": None,  # seed-averaged below, not a single cell-key file
        "eegnetv4": None,  # seed-averaged below, from the full-PhysioNet summary file
        "spdnet_original": None,
        "baselines": {
            "EA+FgMDM": os.path.join(RESULTS_DIR, "physionet_comprehensive_ea_fgmdm.csv"),
            "MEKT": os.path.join(RESULTS_DIR, "physionet_comprehensive_mekt_default.csv"),
            "full ensemble": os.path.join(RESULTS_DIR, "physionet_full_pipeline_109.csv"),
        },
    }

    for dataset, info in datasets_cells.items():
        if dataset == "physionet_full":
            spdnet_matched = load_seed_averaged_dict("spdnet_matched_ea_default", [42, 0, 123])
        else:
            spdnet_matched = load_fold_csv(info["spdnet_matched"])
        if spdnet_matched is None:
            print(f"[{dataset}] matched-protocol SPDNet results not found yet, skipping")
            continue
        subjects = set(spdnet_matched.keys())
        print(f"[{dataset}] matched-protocol SPDNet: n={len(spdnet_matched)} subjects")
        if len(spdnet_matched) < EXPECTED_N[dataset]:
            print(f"[{dataset}] INCOMPLETE: expected {EXPECTED_N[dataset]} subjects, have "
                  f"{len(spdnet_matched)} -- training still in progress, skipping comparison for now")
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
                f"[{dataset}] matched SPDNet vs EEGNetv4",
                {s: eegnetv4[s] for s in common}, {s: spdnet_matched[s] for s in common})
            if r:
                family.append(r)
        else:
            print(f"[{dataset}] EEGNetv4 per-subject results not found at expected path")

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
                f"[{dataset}] matched SPDNet vs {name}",
                baseline, {s: spdnet_matched[s] for s in common})
            if r:
                family.append(r)

        family = holm_bonferroni(family)
        all_results.extend(family)
        for r in family:
            print(f"  {r['comparison']}: diff={r['mean_diff_pp']:+.2f}pp p={r['wilcoxon_p']:.4f} "
                  f"holm_sig={r['holm_significant']}")

    # Protocol-effect isolation: matched vs original SPDNet (population-swap), same dataset.
    # Original SPDNet per-subject BCI-2a results live in the main-results pipeline; original
    # PhysioNet (population-swap) SPDNet results live in the comprehensive comparison.
    protocol_effect_sources = {
        "bci2a": {
            "matched": "spdnet_matched_ea_default_seed42_bci2a",
            "original": "zero_shot_spdnet",  # BCI-2a main-results SPDNet (lwf, population-swap via PhysioNet pretrain)
        },
        "physionet30": {
            "matched": "spdnet_matched_ea_default_seed42_physionet30",
            "original": "physionet_loso_spdnet",  # full-109 population-swap SPDNet, restricted below
        },
        "physionet_full": {
            "matched": None,  # seed-averaged below
            "original": "physionet_loso_spdnet",  # same file, all 109 subjects, no restriction needed
        },
    }
    for dataset, src in protocol_effect_sources.items():
        if dataset == "physionet_full":
            matched = load_seed_averaged_dict("spdnet_matched_ea_default", [42, 0, 123])
        else:
            matched = load_fold_csv(src["matched"])
        if matched is None or src["original"] is None:
            continue
        if len(matched) < EXPECTED_N[dataset]:
            print(f"[{dataset}] protocol-effect isolation: matched-protocol SPDNet incomplete "
                  f"({len(matched)}/{EXPECTED_N[dataset]}), skipping for now")
            continue
        orig_path = os.path.join(RESULTS_DIR, f"{src['original']}.csv")
        original = load_baseline_dict(orig_path, restrict_to=set(matched.keys())) if os.path.exists(orig_path) else None
        if original is None:
            print(f"[{dataset}] original-protocol SPDNet not found at {orig_path}, skipping protocol-effect isolation")
            continue
        common = set(original.keys()) & set(matched.keys())
        r = paired_comparison_from_dicts(
            f"[{dataset}] matched SPDNet vs original-protocol SPDNet",
            original, {s: matched[s] for s in common})
        if r:
            r["holm_significant"] = None  # single comparison, not part of a correction family
            all_results.append(r)
            print(f"  {r['comparison']}: diff={r['mean_diff_pp']:+.2f}pp p={r['wilcoxon_p']:.4f}")

    if all_results:
        save_results(all_results, "stats_spdnet_matched_protocol.csv")
        print(f"\nWrote {len(all_results)} rows to "
              f"{os.path.join(RESULTS_DIR, 'stats_spdnet_matched_protocol.csv')}")


if __name__ == "__main__":
    main()
