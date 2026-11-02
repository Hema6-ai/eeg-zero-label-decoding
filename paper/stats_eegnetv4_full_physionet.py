"""TMLR revision round 2, Phase-5 follow-up item 1: EEGNetv4+EA on the FULL PhysioNet population
(paper/eval_e1_eegnet_physionet_full.py, n=109, 3 seeds) vs. the full 8-member ensemble
(physionet_full_pipeline_109.csv) -- the comparison paper/eval_e1_eegnet_physionet_full.py itself
did not run (it only compared against FgMDM/MEKT individually and the physionet30-subset nested
ensemble). Completes the full-scale EEGNetv4-vs-ensemble picture.

    python -m paper.stats_eegnetv4_full_physionet
"""
import csv
import os

from paper import config
from paper.stats_analysis import paired_comparison_from_dicts, save_results

RESULTS_DIR = config.RESULTS_DIR


def main():
    with open(os.path.join(RESULTS_DIR, "e1_eegnetv4_physionet_full_summary.csv"), newline="") as f:
        eegnet = {int(r["held_out_subject"]): float(r["seed_avg_accuracy"]) for r in csv.DictReader(f)}
    with open(os.path.join(RESULTS_DIR, "physionet_full_pipeline_109.csv"), newline="") as f:
        ensemble = {int(r["held_out"]): float(r["accuracy"]) for r in csv.DictReader(f)}

    result = paired_comparison_from_dicts("8-member ensemble (full PN, in-sample) -> EEGNetv4+EA (full PN)",
                                           ensemble, eegnet)
    save_results([result], "stats_eegnetv4_vs_ensemble_full_physionet.csv")
    print(f"\n{result['comparison']}: {result['mean_diff_pp']:+.2f}pp, "
          f"95% CI [{result['ci_lo_pp']:+.2f},{result['ci_hi_pp']:+.2f}]pp, "
          f"p={result['wilcoxon_p']:.4f}, n={result['n_subjects']}, "
          f"{result['n_improved']}/{result['n_worse']} improved/worse")
    return result


if __name__ == "__main__":
    main()
