"""Subject-level paired statistical analysis, computed end-to-end from real per-subject result
CSVs (not hardcoded numbers), addressing TMLR review Critical Issue C: distinguish statistically
supported findings from empirical trends, and correct for multiple comparisons.

For each named comparison (A -> B), both sides must be per-subject CSVs with a `held_out_subject`
column aligned on the same subject IDs (paired by construction -- LOSO folds are 1:1 per subject).
For each comparison, reports:
  - mean paired difference (percentage points)
  - subject-level bootstrap 95% CI (resampling WHICH SUBJECTS are included, i.e. resampling the
    paired differences themselves -- the correct unit of resampling for a 9- or 109-subject LOSO
    design; never resamples trials, which would treat correlated same-subject trials as
    independent and understate the true uncertainty)
  - Wilcoxon signed-rank test p-value (paired, two-sided)
  - rank-biserial effect size (matched-pairs)
  - win/loss count (subjects improving vs. not)
Then applies Holm-Bonferroni correction across ALL comparisons tested in a given family (not
per-comparison in isolation), and reports which survive correction.

    python -m paper.stats_analysis
"""
import csv
import os

import numpy as np
from scipy.stats import wilcoxon

from paper import config

RESULTS_DIR = config.RESULTS_DIR
RNG_SEED = config.SEED
N_BOOTSTRAP = 100_000


def _load_per_subject(csv_name, value_col="accuracy"):
    path = os.path.join(RESULTS_DIR, csv_name)
    if not os.path.exists(path):
        return None
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["held_out_subject"])] = float(row[value_col])
    return out


def paired_bootstrap_ci(diffs, n_boot=N_BOOTSTRAP, seed=RNG_SEED, alpha=0.05):
    rng = np.random.RandomState(seed)
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi, boot_means.std()


def rank_biserial_matched_pairs(diffs):
    """Matched-pairs rank-biserial effect size: (n_pos - n_neg) / n, using nonzero differences
    only for sign counts but reporting over all n folds (standard Wilcoxon-adjacent effect size,
    matches the manuscript's existing definition -- not weighted by rank magnitude)."""
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0, 0, 0
    n_pos = int((nonzero > 0).sum())
    n_neg = int((nonzero < 0).sum())
    return (n_pos - n_neg) / len(nonzero), n_pos, n_neg


def paired_comparison_from_dicts(name, a, b):
    """a -> b, both {subject_id: accuracy}: positive diff means b improves on a. Core comparison
    logic shared by every family below, regardless of how each side's dict was loaded."""
    common = sorted(set(a) & set(b))
    missing_a = sorted(set(b) - set(a))
    missing_b = sorted(set(a) - set(b))
    if missing_a or missing_b:
        print(f"[{name}] WARNING: non-overlapping subjects dropped from the paired test -- "
              f"only in b: {missing_a}, only in a: {missing_b}. Paired test uses the "
              f"{len(common)} common subjects only.")
    if len(common) < 2:
        print(f"[{name}] SKIPPED -- fewer than 2 common subjects ({len(common)})")
        return None

    diffs = np.array([b[s] - a[s] for s in common])
    mean_diff = diffs.mean()
    lo, hi, se = paired_bootstrap_ci(diffs)
    effect, n_pos, n_neg = rank_biserial_matched_pairs(diffs)

    if np.all(diffs == 0):
        p_value = 1.0
    else:
        try:
            _, p_value = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
        except ValueError:
            # all-identical or too few nonzero differences for scipy's exact/approx methods
            p_value = 1.0

    result = {
        "comparison": name, "n_subjects": len(common),
        "mean_diff_pp": mean_diff * 100, "ci_lo_pp": lo * 100, "ci_hi_pp": hi * 100,
        "bootstrap_se_pp": se * 100, "wilcoxon_p": p_value,
        "rank_biserial": effect, "n_improved": n_pos, "n_worse": n_neg,
    }
    print(f"[{name}] n={len(common)} mean_diff={mean_diff*100:+.2f}pp "
          f"95%CI=[{lo*100:+.2f},{hi*100:+.2f}]pp p={p_value:.4f} "
          f"rank_biserial={effect:.3f} ({n_pos}/{n_neg} improved/worse)")
    return result


def paired_comparison(name, csv_a, csv_b, value_col="accuracy"):
    """csv_a -> csv_b: thin wrapper around paired_comparison_from_dicts for the common case where
    both sides are plain per-subject CSVs (one `held_out_subject` row per subject, no preamble)."""
    a = _load_per_subject(csv_a, value_col)
    b = _load_per_subject(csv_b, value_col)
    if a is None or b is None:
        print(f"[{name}] SKIPPED -- missing file(s): "
              f"{csv_a if a is None else ''} {csv_b if b is None else ''}")
        return None
    return paired_comparison_from_dicts(name, a, b)


def holm_bonferroni(results, alpha=0.05):
    """Applies Holm-Bonferroni step-down correction across a family of comparisons (each already
    computed independently above) and annotates each with its corrected significance decision."""
    valid = [r for r in results if r is not None]
    order = sorted(range(len(valid)), key=lambda i: valid[i]["wilcoxon_p"])
    m = len(valid)
    reject_from = m  # index (in sorted order) from which we stop rejecting
    for rank, i in enumerate(order):
        threshold = alpha / (m - rank)
        if valid[i]["wilcoxon_p"] > threshold:
            reject_from = rank
            break
    for rank, i in enumerate(order):
        valid[i]["holm_significant"] = rank < reject_from
    return valid


def save_results(results, out_name):
    valid = [r for r in results if r is not None]
    if not valid:
        print(f"No valid comparisons to save for {out_name}")
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, out_name)
    fieldnames = list(valid[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(valid)
    print(f"Saved {len(valid)} comparisons to {path}")


def _load_fgmdm_per_fold():
    """riemannian_advanced_experiment.csv mixes a 4-method summary table and a per-fold table
    (for whichever method happened to be best, here FgMDM) in one file; load the per-fold section
    directly by seeking to its header row rather than extending the generic CSV loader for a
    one-off file layout."""
    path = os.path.join(RESULTS_DIR, "riemannian_advanced_experiment.csv")
    out = {}
    with open(path, newline="") as f:
        rows = f.readlines()
    start = next(i for i, line in enumerate(rows) if line.startswith("held_out_subject"))
    import io
    for row in csv.DictReader(io.StringIO("".join(rows[start:]))):
        out[int(row["held_out_subject"])] = float(row["accuracy"])
    return out


def main():
    # --- Family 1: BCI-IV 2a progression (manuscript Section "statanalysis") ---
    fgmdm = _load_fgmdm_per_fold()
    mekt_default = _load_per_subject("zero_shot_mekt.csv")
    mekt_window_lwf = _load_per_subject("zero_shot_mekt_window_lwf.csv")
    mekt_window_corr = _load_per_subject("zero_shot_mekt_window_corr.csv")
    ensemble = _load_per_subject("zero_shot_mekt_ensemble.csv")

    results = []
    if fgmdm and mekt_default:
        results.append(paired_comparison_from_dicts("FgMDM -> MEKT (default)", fgmdm, mekt_default))
    if mekt_default and mekt_window_lwf:
        results.append(paired_comparison_from_dicts(
            "MEKT (default) -> +window", mekt_default, mekt_window_lwf))
    if mekt_window_lwf and mekt_window_corr:
        results.append(paired_comparison_from_dicts(
            "+window -> +corr estimator", mekt_window_lwf, mekt_window_corr))
    if mekt_window_corr and ensemble:
        results.append(paired_comparison_from_dicts(
            "+corr -> 8-member ensemble", mekt_window_corr, ensemble))
    if fgmdm and ensemble:
        results.append(paired_comparison_from_dicts(
            "FgMDM (baseline) -> ensemble (final)", fgmdm, ensemble))

    results = holm_bonferroni(results)
    save_results(results, "statistical_tests_bci2a.csv")

    # --- Family 2: PhysioNet-comprehensive classical methods (this revision's new experiment) ---
    print("\n--- PhysioNet comprehensive classical-method comparisons ---")
    pn_csp_lda = _load_per_subject("physionet_comprehensive_ea_csp8_lda.csv")
    pn_csp_svm = _load_per_subject("physionet_comprehensive_ea_csp8_svm.csv")
    pn_fbcsp = _load_per_subject("physionet_comprehensive_ea_fbcsp_svm.csv")
    pn_mdm = _load_per_subject("physionet_comprehensive_ea_mdm.csv")
    pn_fgmdm = _load_per_subject("physionet_comprehensive_ea_fgmdm.csv")
    pn_rpa = _load_per_subject("physionet_comprehensive_rpa.csv")
    pn_mekt_default = _load_per_subject("physionet_comprehensive_mekt_default.csv")
    pn_mekt_tuned = _load_per_subject("physionet_comprehensive_mekt_tuned.csv")
    # Deep-learning PhysioNet results (reversed-direction pretraining, see
    # reports/physionet_deep_learning_comparison.md); SPDNet is the standout performer there.
    pn_spdnet = _load_per_subject("physionet_loso_spdnet.csv")
    pn_mae_grl = _load_per_subject("physionet_loso_mae_grl.csv")

    pn_results = []
    for name, a, b in [
        ("PhysioNet: CSP+LDA -> FgMDM", pn_csp_lda, pn_fgmdm),
        ("PhysioNet: CSP+SVM -> FBCSP+SVM", pn_csp_svm, pn_fbcsp),
        ("PhysioNet: MDM -> FgMDM", pn_mdm, pn_fgmdm),
        ("PhysioNet: FgMDM -> MEKT (default)", pn_fgmdm, pn_mekt_default),
        ("PhysioNet: RPA -> MEKT (default)", pn_rpa, pn_mekt_default),
        # Reviewer-relevant: on BCI-IV 2a, tuning (k=50/lam=1/20iter vs k=100/lam=1/5iter) IMPROVED
        # accuracy (46.26% -> 48.48%); check whether that tuning (done by looking at BCI-IV 2a's
        # own 9-subject LOSO accuracy) also helps on PhysioNet's independent 109-subject population,
        # or whether it was specific to the 9 subjects it was tuned against.
        ("PhysioNet: MEKT (default) -> MEKT (BCI-IV-2a-tuned)", pn_mekt_default, pn_mekt_tuned),
        # Best classical (MEKT default, 51.54%) vs. the standout deep-learning result (SPDNet,
        # 49.40%) -- directly tests this revision's central open question on the larger dataset.
        ("PhysioNet: SPDNet (deep learning) -> MEKT default (classical, best)", pn_spdnet, pn_mekt_default),
        ("PhysioNet: MAE+GRL (deep learning) -> SPDNet (deep learning)", pn_mae_grl, pn_spdnet),
    ]:
        if a and b:
            pn_results.append(paired_comparison_from_dicts(name, a, b))
        else:
            print(f"[{name}] SKIPPED -- one or both result files not yet available")

    pn_results = holm_bonferroni(pn_results)
    save_results(pn_results, "statistical_tests_physionet.csv")


if __name__ == "__main__":
    main()
