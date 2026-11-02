"""Statistical analysis for task E3 (paper/eval_e3_physionet_factorial.py): main effects of
covariance estimator, method, and time window at PhysioNet scale (up to 109 subjects), using the
exact same subject-paired Wilcoxon + bootstrap-CI + rank-biserial methodology as
paper/stats_factorial.py (reused directly, not reimplemented), Holm-corrected across every
contrast in this file (not per-lever) since this is the headline comparison for reviewer Critical
Issue 2.

    python -m paper.stats_e3_physionet_factorial
"""
import os

from paper.eval_e3_physionet_factorial import _cell_key, _load_done_folds, OUT_PREFIX
from paper.eval_factorial import WINDOWS, ESTIMATORS
from paper.stats_analysis import holm_bonferroni, save_results
from paper.stats_factorial import average_over, paired_test

METHODS = ["fgmdm", "mekt_default"]


def load_cell(method, tmin, tmax, estimator):
    key = _cell_key(method, tmin, tmax, estimator)
    done = _load_done_folds(key)
    return {s: r["accuracy"] for s, r in done.items()}


def main():
    cells = {}
    for method in METHODS:
        for tmin, tmax in WINDOWS:
            for estimator in ESTIMATORS:
                cells[(method, tmin, tmax, estimator)] = load_cell(method, tmin, tmax, estimator)
    subjects = sorted(set.intersection(*[set(c) for c in cells.values()]))
    print(f"=== E3 (PhysioNet factorial) analysis: {len(cells)} cells, {len(subjects)} common subjects ===\n")

    results = []

    print("--- Main effect: covariance estimator (lwf -> corr), averaged over window x method ---")
    lwf_cells = [v for (m, tmin, tmax, e), v in cells.items() if e == "lwf"]
    corr_cells = [v for (m, tmin, tmax, e), v in cells.items() if e == "corr"]
    lwf_avg = average_over(lwf_cells, subjects)
    corr_avg = average_over(corr_cells, subjects)
    paired_test("E3 estimator main effect: lwf -> corr", lwf_avg, corr_avg, results)

    print("\n--- Main effect: method (FgMDM -> MEKT default), averaged over window x estimator ---")
    fgmdm_cells = [v for (m, tmin, tmax, e), v in cells.items() if m == "fgmdm"]
    mekt_cells = [v for (m, tmin, tmax, e), v in cells.items() if m == "mekt_default"]
    fgmdm_avg = average_over(fgmdm_cells, subjects)
    mekt_avg = average_over(mekt_cells, subjects)
    paired_test("E3 method main effect: FgMDM -> MEKT default", fgmdm_avg, mekt_avg, results)

    print("\n--- Main effect: time window (vs. default (0,3)), averaged over estimator x method ---")
    default_cells = [v for (m, tmin, tmax, e), v in cells.items() if (tmin, tmax) == (0.0, 3.0)]
    default_avg = average_over(default_cells, subjects)
    for tmin, tmax in WINDOWS:
        if (tmin, tmax) == (0.0, 3.0):
            continue
        window_cells = [v for (m, t0, t1, e), v in cells.items() if (t0, t1) == (tmin, tmax)]
        window_avg = average_over(window_cells, subjects)
        paired_test(f"E3 window main effect: default -> ({tmin},{tmax})", default_avg, window_avg, results)

    results = holm_bonferroni(results)
    save_results(results, f"{OUT_PREFIX}_statistical_analysis.csv")

    print("\n=== Summary: does this replicate Section 10.1's factorial finding at n="
          f"{len(subjects)}? ===")
    for r in results:
        print(f"  {r['contrast']}: {r['mean_diff_pp']:+.2f}pp, p={r['wilcoxon_p']:.4f}, "
              f"Holm-sig={r['holm_significant']}")
    return results


if __name__ == "__main__":
    main()
