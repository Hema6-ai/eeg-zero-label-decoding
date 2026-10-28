"""Subject-level robustness check (TMLR revision, Phase VIII.5): for each key paired comparison,
leave-one-subject-out jackknife -- recompute the paired Wilcoxon test with each single subject
removed in turn, to check whether any one subject's removal flips the significance conclusion.
Distinct from Holm-Bonferroni correction (which controls the multiple-comparison family-wise
error rate) and from the bootstrap CI (which resamples WITH replacement); this instead asks "is
the conclusion fragile to any single subject," a different robustness question.

    python -m paper.robustness_subject_influence
"""
import csv
import os

import numpy as np
from scipy.stats import wilcoxon

from paper import config
from paper.stats_analysis import _load_per_subject

RESULTS_DIR = config.RESULTS_DIR

# (name, csv_a, csv_b): same direction convention as stats_analysis.py (b - a)
KEY_COMPARISONS = [
    ("PhysioNet: SPDNet -> MEKT (default)", "physionet_loso_spdnet.csv", "physionet_comprehensive_mekt_default.csv"),
    ("PhysioNet: MDM -> FgMDM", "physionet_comprehensive_ea_mdm.csv", "physionet_comprehensive_ea_fgmdm.csv"),
    ("PhysioNet: FgMDM -> MEKT (default)", "physionet_comprehensive_ea_fgmdm.csv", "physionet_comprehensive_mekt_default.csv"),
    ("PhysioNet: MAE+GRL -> SPDNet", "physionet_loso_mae_grl.csv", "physionet_loso_spdnet.csv"),
]


def jackknife_comparison(name, path_a, path_b, alpha=0.05):
    a = _load_per_subject(path_a)
    b = _load_per_subject(path_b)
    common = sorted(set(a) & set(b))
    diffs_by_subject = {s: b[s] - a[s] for s in common}
    all_diffs = np.array([diffs_by_subject[s] for s in common])

    _, p_full = wilcoxon(all_diffs, alternative="two-sided", zero_method="wilcox")
    full_sig = p_full < alpha
    print(f"\n[{name}] full sample: n={len(common)}, mean_diff={all_diffs.mean()*100:+.2f}pp, "
          f"p={p_full:.4g}, significant={full_sig}")

    flips = []
    p_values_loo = []
    for leave_out in common:
        remaining = [s for s in common if s != leave_out]
        diffs = np.array([diffs_by_subject[s] for s in remaining])
        if np.all(diffs == 0):
            p = 1.0
        else:
            try:
                _, p = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
            except ValueError:
                p = 1.0
        p_values_loo.append(p)
        sig = p < alpha
        if sig != full_sig:
            flips.append((leave_out, p))

    p_values_loo = np.array(p_values_loo)
    print(f"  leave-one-out p-value range: [{p_values_loo.min():.4g}, {p_values_loo.max():.4g}]")
    if flips:
        print(f"  FRAGILE: removing subject(s) {[s for s, _ in flips]} flips significance "
              f"(full p={p_full:.4g} -> {[(s, f'{p:.4g}') for s, p in flips]})")
    else:
        print(f"  ROBUST: significance conclusion ({'significant' if full_sig else 'not significant'}) "
              f"unchanged under removal of any single subject")

    return {
        "comparison": name, "n_subjects": len(common), "full_p": p_full, "full_significant": full_sig,
        "loo_p_min": p_values_loo.min(), "loo_p_max": p_values_loo.max(),
        "n_flips": len(flips), "flip_subjects": ";".join(str(s) for s, _ in flips),
    }


def main():
    rows = [jackknife_comparison(name, a, b) for name, a, b in KEY_COMPARISONS]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "robustness_subject_influence.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} jackknife robustness checks to {out_path}")


if __name__ == "__main__":
    main()
