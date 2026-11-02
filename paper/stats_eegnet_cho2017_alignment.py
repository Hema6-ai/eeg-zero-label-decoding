"""TMLR revision round 2, Phase-5 follow-up item 1(b)/(c): EEGNetv4 EA on/off on Cho2017,
per-subject seed-averaged accuracy, paired Wilcoxon -- completes the alignment-effect picture for
EEGNetv4 across all three datasets (BCI-2a, PhysioNet30, and now Cho2017).

    python -m paper.stats_eegnet_cho2017_alignment
"""
import csv
import os

import numpy as np

from paper import config
from paper.stats_analysis import paired_comparison_from_dicts, save_results

RESULTS_DIR = config.RESULTS_DIR
SEEDS = [42, 0, 123, 21, 7]


def load_seed_avg(suffix):
    seed_dicts = []
    for seed in SEEDS:
        path = os.path.join(RESULTS_DIR, f"e5_best_cnn_eegnetv4_{suffix}_seed{seed}.csv")
        with open(path, newline="") as f:
            seed_dicts.append({int(r["held_out_subject"]): float(r["accuracy"]) for r in csv.DictReader(f)})
    common = sorted(set.intersection(*[set(d) for d in seed_dicts]))
    return {s: float(np.mean([d[s] for d in seed_dicts])) for s in common}


def main():
    ea = load_seed_avg("ea")
    noea = load_seed_avg("noea")
    result = paired_comparison_from_dicts("EEGNetv4 (Cho2017): no-EA -> EA", noea, ea)
    save_results([result], "stats_eegnet_cho2017_alignment.csv")
    print(f"\n{result['comparison']}: {result['mean_diff_pp']:+.2f}pp, "
          f"95% CI [{result['ci_lo_pp']:+.2f},{result['ci_hi_pp']:+.2f}]pp, "
          f"p={result['wilcoxon_p']:.4f}, n={result['n_subjects']}, "
          f"{result['n_improved']}/{result['n_worse']} improved/worse")
    return result


if __name__ == "__main__":
    main()
