"""Statistical analysis of the factorial preprocessing-vs-architecture experiment
(paper/eval_factorial.py). Estimates main effects (time window, covariance estimator, method
family) and interactions, using subject-level paired contrasts throughout (never treating the 9
BCI-IV 2a subjects' repeated measurements as independent observations) -- the same nonparametric,
subject-paired methodology as paper/stats_analysis.py, applied to a full factorial design instead
of a sequential progression.

Main-effect estimation: for each level of a factor, average each subject's accuracy over the
OTHER factors (e.g. the window main effect averages over both estimators and both methods for
each window, per subject), then paired-compare levels via Wilcoxon + bootstrap CI + rank-biserial,
exactly as elsewhere in this project. This is a legitimate, standard way to estimate a main effect
in a full factorial design without requiring a parametric ANOVA assumption inappropriate for n=9.

Interaction estimation: for the window x method interaction (the question the directive's Phase IV
is centrally about -- "whether preprocessing effects depend on architecture"), compute each
subject's (best window - worst-plausible-default window) simple effect SEPARATELY within each
method, then paired-compare those per-subject simple-effect contrasts across methods. A
significant difference means the window effect's SIZE genuinely depends on which method it's
applied to.

    python -m paper.stats_factorial
"""
import csv
import os

import numpy as np
from scipy.stats import wilcoxon

from paper.eval_factorial import WINDOWS, ESTIMATORS, _cell_key, _load_done_folds

RESULTS_DIR = "paper/results"
METHODS = ["fgmdm", "mekt_default"]
N_BOOTSTRAP = 100_000
SEED = 42


def load_cell(method, tmin, tmax, estimator):
    key = _cell_key(method, tmin, tmax, estimator)
    done = _load_done_folds(key)
    return {s: r["accuracy"] for s, r in done.items()}


def bootstrap_ci(diffs, n_boot=N_BOOTSTRAP, seed=SEED, alpha=0.05):
    rng = np.random.RandomState(seed)
    n = len(diffs)
    boots = np.array([diffs[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def paired_test(name, vals_a, vals_b, results):
    """vals_a, vals_b: {subject: accuracy} dicts, already averaged over nuisance factors."""
    common = sorted(set(vals_a) & set(vals_b))
    diffs = np.array([vals_b[s] - vals_a[s] for s in common])
    mean_diff = diffs.mean()
    lo, hi = bootstrap_ci(diffs)
    if np.all(diffs == 0):
        p = 1.0
    else:
        try:
            _, p = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
        except ValueError:
            p = 1.0
    nonzero = diffs[diffs != 0]
    n_pos, n_neg = int((nonzero > 0).sum()), int((nonzero < 0).sum())
    effect = (n_pos - n_neg) / len(nonzero) if len(nonzero) else 0.0
    print(f"[{name}] n={len(common)} mean_diff={mean_diff*100:+.2f}pp 95%CI=[{lo*100:+.2f},{hi*100:+.2f}]pp "
          f"p={p:.4f} rank_biserial={effect:.3f} ({n_pos}/{n_neg})")
    results.append({"contrast": name, "n_subjects": len(common), "mean_diff_pp": mean_diff * 100,
                     "ci_lo_pp": lo * 100, "ci_hi_pp": hi * 100, "wilcoxon_p": p, "rank_biserial": effect,
                     "n_improved": n_pos, "n_worse": n_neg})
    return common, diffs


def average_over(cells, subjects):
    """cells: list of {subject: acc} dicts to average pointwise per subject."""
    out = {}
    for s in subjects:
        vals = [c[s] for c in cells if s in c]
        if len(vals) == len(cells):
            out[s] = float(np.mean(vals))
    return out


def main():
    # Load every cell.
    cells = {}
    for method in METHODS:
        for tmin, tmax in WINDOWS:
            for estimator in ESTIMATORS:
                cells[(method, tmin, tmax, estimator)] = load_cell(method, tmin, tmax, estimator)
    subjects = sorted(set.intersection(*[set(c) for c in cells.values()]))
    print(f"=== Factorial analysis: {len(cells)} cells, {len(subjects)} common subjects ===\n")

    results = []

    # --- Main effect: covariance estimator (average over window and method) ---
    print("--- Main effect: covariance estimator (lwf -> corr), averaged over window x method ---")
    lwf_cells = [v for (m, tmin, tmax, e), v in cells.items() if e == "lwf"]
    corr_cells = [v for (m, tmin, tmax, e), v in cells.items() if e == "corr"]
    lwf_avg = average_over(lwf_cells, subjects)
    corr_avg = average_over(corr_cells, subjects)
    paired_test("Estimator main effect: lwf -> corr", lwf_avg, corr_avg, results)

    # --- Main effect: method family (FgMDM -> MEKT, average over window x estimator) ---
    print("\n--- Main effect: method (FgMDM -> MEKT), averaged over window x estimator ---")
    fgmdm_cells = [v for (m, tmin, tmax, e), v in cells.items() if m == "fgmdm"]
    mekt_cells = [v for (m, tmin, tmax, e), v in cells.items() if m == "mekt_default"]
    fgmdm_avg = average_over(fgmdm_cells, subjects)
    mekt_avg = average_over(mekt_cells, subjects)
    paired_test("Method main effect: FgMDM -> MEKT", fgmdm_avg, mekt_avg, results)

    # --- Main effect: window (default (0,3) -> each other window, averaged over estimator x method) ---
    print("\n--- Main effect: time window (vs. default (0,3)), averaged over estimator x method ---")
    default_cells = [v for (m, tmin, tmax, e), v in cells.items() if (tmin, tmax) == (0.0, 3.0)]
    default_avg = average_over(default_cells, subjects)
    for tmin, tmax in WINDOWS:
        if (tmin, tmax) == (0.0, 3.0):
            continue
        window_cells = [v for (m, t0, t1, e), v in cells.items() if (t0, t1) == (tmin, tmax)]
        window_avg = average_over(window_cells, subjects)
        paired_test(f"Window main effect: default -> ({tmin},{tmax})", default_avg, window_avg, results)

    # --- Interaction: does the (default -> best non-default window) simple effect size differ
    # between FgMDM and MEKT? Best non-default window chosen by POOLED (both methods) mean, fixed
    # before looking at per-method breakdown, to avoid cherry-picking the method that benefits most. ---
    print("\n--- Interaction: window effect x method ---")
    non_default_means = {}
    for tmin, tmax in WINDOWS:
        if (tmin, tmax) == (0.0, 3.0):
            continue
        window_cells = [v for (m, t0, t1, e), v in cells.items() if (t0, t1) == (tmin, tmax)]
        window_avg = average_over(window_cells, subjects)
        non_default_means[(tmin, tmax)] = np.mean(list(window_avg.values()))
    best_window = max(non_default_means, key=non_default_means.get)
    print(f"  (best non-default window by pooled mean, prespecified selection rule: {best_window})")

    for method in METHODS:
        method_default = cells[(method, 0.0, 3.0, "lwf")]  # lwf only, to isolate window from estimator
        method_best = cells[(method, best_window[0], best_window[1], "lwf")]
        common = sorted(set(method_default) & set(method_best))
        diffs = np.array([method_best[s] - method_default[s] for s in common])
        print(f"  [{method}] default->best-window simple effect: {diffs.mean()*100:+.2f}pp (n={len(common)})")
        results.append({"contrast": f"Simple effect (window, lwf only): {method} default->{best_window}",
                         "n_subjects": len(common), "mean_diff_pp": diffs.mean() * 100,
                         "ci_lo_pp": "", "ci_hi_pp": "", "wilcoxon_p": "", "rank_biserial": "",
                         "n_improved": int((diffs > 0).sum()), "n_worse": int((diffs < 0).sum())})

    # Interaction contrast: (simple effect in MEKT) - (simple effect in FgMDM), subject-paired.
    fgmdm_default, fgmdm_best = cells[("fgmdm", 0.0, 3.0, "lwf")], cells[("fgmdm", best_window[0], best_window[1], "lwf")]
    mekt_default, mekt_best = cells[("mekt_default", 0.0, 3.0, "lwf")], cells[("mekt_default", best_window[0], best_window[1], "lwf")]
    common = sorted(set(fgmdm_default) & set(fgmdm_best) & set(mekt_default) & set(mekt_best))
    fgmdm_simple = {s: fgmdm_best[s] - fgmdm_default[s] for s in common}
    mekt_simple = {s: mekt_best[s] - mekt_default[s] for s in common}
    paired_test(f"Interaction (window x method): MEKT's simple effect -> FgMDM's simple effect",
                mekt_simple, fgmdm_simple, results)

    # --- Estimator main effect, deep-learning arm: SPDNet lwf -> corr, default window only.
    # SPDNet is retrained per estimator (unlike FgMDM/MEKT, which only need a different covariance
    # matrix at eval time), so it is not part of the 4-window grid above -- it is a single extra
    # cell pair at the default (0,3) window, checking whether the estimator main effect found (or
    # not found) in the classical/domain-adaptation arm generalizes to the deep-learning method
    # family. Reads the two already-existing zero-shot result files directly (not the resumable
    # eval_factorial.py cell cache, since SPDNet retraining lives in paper/train_spdnet.py). ---
    print("\n--- Estimator main effect, deep-learning arm: SPDNet lwf -> corr (default window) ---")

    def _load_spdnet(path):
        out = {}
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                out[int(row["held_out_subject"])] = float(row["accuracy"])
        return out

    spdnet_lwf = _load_spdnet(os.path.join(RESULTS_DIR, "zero_shot_spdnet.csv"))
    spdnet_corr = _load_spdnet(os.path.join(RESULTS_DIR, "zero_shot_spdnet_corrcov.csv"))
    paired_test("Estimator main effect (SPDNet only): lwf -> corr", spdnet_lwf, spdnet_corr, results)

    # Holm-Bonferroni across the whole family of contrasts computed above.
    valid = [r for r in results if r["wilcoxon_p"] != ""]
    order = sorted(range(len(valid)), key=lambda i: valid[i]["wilcoxon_p"])
    m = len(valid)
    reject_from = m
    for rank, i in enumerate(order):
        if valid[i]["wilcoxon_p"] > 0.05 / (m - rank):
            reject_from = rank
            break
    for rank, i in enumerate(order):
        valid[i]["holm_significant"] = rank < reject_from
    for r in results:
        if "holm_significant" not in r:
            r["holm_significant"] = ""

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "factorial_statistical_analysis.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["contrast", "n_subjects", "mean_diff_pp", "ci_lo_pp",
                                                 "ci_hi_pp", "wilcoxon_p", "rank_biserial", "n_improved",
                                                 "n_worse", "holm_significant"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} contrasts to {out_path}")


if __name__ == "__main__":
    main()
