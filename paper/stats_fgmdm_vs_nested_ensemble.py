"""TMLR revision round 2, Phase-5 follow-up item 3/task F(ii): the manuscript's
`tab:pairedstats` reports FgMDM -> ensemble (+4.94pp, p=0.0039) using the non-nested,
in-sample-selected 52.39% ensemble. This recomputes that exact comparison using the honestly
nested ensemble instead (paper/eval_e4_nested_selection.py's `ensemble_nested_accuracy`, which
re-runs the full window/estimator/method search per outer fold on source subjects only). This
number was previously derived ad hoc in a revision-tracking report with no saved script+CSV
backing it -- a violation of this project's own rule; this script fixes that.

    python -m paper.stats_fgmdm_vs_nested_ensemble
"""
import csv
import os

from paper import config
from paper.stats_analysis import paired_comparison_from_dicts, save_results

RESULTS_DIR = config.RESULTS_DIR


def main():
    fgmdm_path = os.path.join(RESULTS_DIR, "factorial_fgmdm_t0.0_3.0_lwf.csv")
    nested_path = os.path.join(RESULTS_DIR, "e4_nested_selection_bci2a_reducedgrid.csv")

    with open(fgmdm_path, newline="") as f:
        fgmdm = {int(r["held_out_subject"]): float(r["accuracy"]) for r in csv.DictReader(f)}
    with open(nested_path, newline="") as f:
        nested_ensemble = {int(r["held_out_subject"]): float(r["ensemble_nested_accuracy"])
                            for r in csv.DictReader(f)}

    result = paired_comparison_from_dicts("FgMDM -> E4 nested-selected ensemble (bci2a)",
                                           fgmdm, nested_ensemble)
    save_results([result], "stats_fgmdm_vs_nested_ensemble.csv")
    print(f"\nFgMDM -> nested ensemble: {result['mean_diff_pp']:+.2f}pp, "
          f"95% CI [{result['ci_lo_pp']:+.2f},{result['ci_hi_pp']:+.2f}]pp, "
          f"p={result['wilcoxon_p']:.4f}, n={result['n_subjects']}, "
          f"{result['n_improved']}/{result['n_worse']} improved/worse")
    print("Compare to the manuscript's existing non-nested claim: FgMDM -> ensemble (in-sample, "
          "52.39%): +4.94pp, 95% CI [+3.38,+6.58]pp, p=0.0039 (Table tab:pairedstats).")
    return result


if __name__ == "__main__":
    main()
