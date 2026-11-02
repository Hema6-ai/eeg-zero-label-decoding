"""TMLR revision round 2, task E8: effect-size comparison table. Turns this paper's central claim
("preprocessing mattered more than architecture") into a measured one -- for each candidate lever,
tabulate the paired per-subject mean difference, bootstrap 95% CI, Wilcoxon p, and rank-biserial
effect size, Holm-corrected across every comparison in this table (not per-lever), on both BCI-IV
2a and PhysioNet wherever a matching per-subject CSV exists on that dataset.

This script only tabulates EXISTING per-subject result files (paper/results/*.csv) plus this
round's new E6 output -- it runs no new model. Rows that depend on E1 (supervised CNN baselines),
E3 (PhysioNet-scale factorial), or E7 (confidence-weighted fusion reproduction) are left as
explicit "pending" placeholders until those tasks complete, rather than silently omitted; rerun
this script once each dependency lands to fill them in (resumable/idempotent -- it is read-only
with respect to the CSVs it loads).

    python -m paper.eval_e8_effect_sizes
"""
import csv
import os

from paper import config
from paper.run_dl_loso_sweep import load_baseline_dict
from paper.stats_analysis import holm_bonferroni, paired_comparison_from_dicts

RESULTS_DIR = config.RESULTS_DIR


def load_e6_condition(method, condition, dataset="bci2a"):
    """method__condition rows from this round's E6 output (paper/eval_e6_transductive_inductive.py)."""
    path = os.path.join(RESULTS_DIR, "e6_transductive_inductive_folds.csv")
    if not os.path.exists(path):
        return None
    key = f"{method}__{condition}"
    with open(path, newline="") as f:
        return {int(r["held_out_subject"]): float(r["accuracy"])
                for r in csv.DictReader(f) if r["method_condition"] == key}


def main():
    comparisons = []
    pending = []

    def add(lever, dataset, name_a, dict_a, name_b, dict_b):
        if dict_a is None or dict_b is None:
            pending.append(f"{lever} ({dataset}): {name_a} -> {name_b} -- missing source file(s)")
            return
        result = paired_comparison_from_dicts(f"{lever} [{dataset}] {name_a}->{name_b}", dict_a, dict_b)
        if result is None:
            pending.append(f"{lever} ({dataset}): {name_a} -> {name_b} -- paired_comparison returned None "
                            f"(e.g. <2 common subjects)")
            return
        result.update({"lever": lever, "dataset": dataset, "a": name_a, "b": name_b})
        comparisons.append(result)

    R = lambda name: os.path.join(RESULTS_DIR, name)

    # --- Alignment on/off (FgMDM) ---
    add("alignment_on_off", "bci2a", "FgMDM (no EA)", load_e6_condition("fgmdm", "none"),
        "FgMDM (batch EA)", load_e6_condition("fgmdm", "batch_ea"))
    pending.append("alignment_on_off (physionet): no per-subject FgMDM-without-EA file exists in "
                    "this repo for PhysioNet -- would require a new (cheap, classical, CPU-tractable) "
                    "run, not currently part of any listed E-task; flagged for the author to decide "
                    "whether it is worth adding.")

    # --- Covariance features vs. raw-EEG encoders (FgMDM vs MAE-only) ---
    # NOTE: paper/results/zero_shot_mae_only.csv is a STALE, incomplete 3-subject partial run
    # (subjects 1-3 only) -- it does NOT match the manuscript's reported BCI-IV 2a MAE-only number
    # (27.84%, from the seed-42 column of Table tab:dlseedstability); that run's per-subject CSV
    # was apparently never synced back from the Kaggle session that produced it. Using the stale
    # file would silently misrepresent this comparison (n=3, inflated CI) -- so it is correctly
    # left pending below rather than computed from the wrong data.
    pending.append("covariance_vs_raw_eeg (bci2a): paper/results/zero_shot_mae_only.csv is a "
                    "stale 3-subject partial run that does not match the manuscript's reported "
                    "27.84% (Table tab:dlseedstability, seed 42); the correct per-subject CSV for "
                    "that run was not found in this repo. Needs regenerating "
                    "(paper/train_pretrain.py + paper/eval_zero_shot.py at seed 42) before this "
                    "row can be added honestly.")
    add("covariance_vs_raw_eeg", "physionet", "MAE-only (raw EEG)",
        load_baseline_dict(R("physionet_loso_mae_only.csv")),
        "EA+FgMDM (covariance)", load_baseline_dict(R("physionet_comprehensive_ea_fgmdm.csv")))

    # --- Window effect: see paper/stats_factorial.py for the full factorial main-effect estimate
    # (paired, averaged over estimator+method per subject) -- not duplicated here; E3 will extend
    # this lever to PhysioNet.
    pending.append("window_effect (bci2a): already reported with its own dedicated, paired, "
                    "averaged-over-nuisance-factors main-effect analysis in Section 10.1 "
                    "(sec:factorial, Table 4) -- not recomputed here to avoid a second, "
                    "differently-pooled p-value for the same question; see that table directly.")
    pending.append("window_effect (physionet): task E3 is now COMPLETE -- not duplicated here to "
                    "avoid a second, differently-pooled p-value for the same question; see "
                    "paper/results/e3_physionet_factorial_statistical_analysis.csv directly "
                    "(headline: default -> (0.3,2.5), +1.73pp, p=0.0798, NOT Holm-significant -- "
                    "does not replicate BCI-IV 2a's significant result at this larger n=30).")
    pending.append("estimator_effect (bci2a): same as window_effect -- see Section 10.1/Table 4.")
    pending.append("estimator_effect (physionet): task E3 is now COMPLETE -- see "
                    "paper/results/e3_physionet_factorial_statistical_analysis.csv directly "
                    "(lwf -> corr, -0.08pp, p=0.9274, not significant -- consistent null on both "
                    "datasets).")

    # --- Domain adaptation: MEKT vs FgMDM ---
    add("domain_adaptation_mekt_vs_fgmdm", "bci2a", "EA+FgMDM",
        load_baseline_dict(R("factorial_fgmdm_t0.0_3.0_lwf.csv")),
        "MEKT-lite (default)", load_baseline_dict(R("zero_shot_mekt.csv")))
    pending.append("domain_adaptation_mekt_vs_fgmdm (physionet): already reported with its own "
                    "Holm-corrected paired test in Table tab:physionet-stats "
                    "(FgMDM -> MEKT default: +2.10pp, p=0.0001, Holm-significant) -- not "
                    "recomputed here; see that table directly.")

    # --- Architecture-family gaps vs. FgMDM ---
    arch_files = {
        "bci2a": {
            # SPDNet and LaBraM per-subject CSVs are confirmed exact matches (n=9) to the
            # manuscript's reported means. Riemannian Transformer v2, MAE-only, and MAE+GRL are
            # deliberately NOT listed here -- their only local per-subject CSVs
            # (zero_shot_riemannian_transformer_v2.csv, zero_shot_mae_only.csv,
            # zero_shot_mae_grl.csv) are stale 3-subject partial runs that do not match the
            # manuscript's reported seed-42 numbers (42.38%/27.84%/27.47%, Table
            # tab:dlseedstability); see the "pending" note below instead of silently using them.
            "SPDNet": "zero_shot_spdnet.csv",
            "LaBraM": "zero_shot_labram.csv",
            # E1 result (seed 42, EA-aligned, default [0,3]s window -- the pre-specified
            # canonical cell, not cherry-picked as the best-performing one among the 4
            # ea/window combos run for each architecture).
            "EEGNetv4 (E1, supervised)": "e1_dlloso_eegnetv4_ea_default_seed42_bci2a.csv",
            "ShallowFBCSPNet (E1, supervised)": "e1_dlloso_shallowfbcspnet_ea_default_seed42_bci2a.csv",
            "ATCNet (E1, supervised)": "e1_dlloso_atcnet_ea_default_seed42_bci2a.csv",
        },
        "physionet": {
            "SPDNet": "physionet_loso_spdnet.csv",
            "Riemannian Transformer v2": "physionet_loso_riemannian_transformer_v2.csv",
            "MAE-only": "physionet_loso_mae_only.csv",
            "MAE+GRL": "physionet_loso_mae_grl.csv",
            # E1, seed 42, EA-aligned default window -- PhysioNet30 subset (30 subjects); the
            # paired_comparison_from_dicts call below automatically restricts the 109-subject
            # FgMDM baseline down to these same 30 subjects (warns, does not silently mismatch).
            "EEGNetv4 (E1, supervised)": "e1_dlloso_eegnetv4_ea_default_seed42_physionet30.csv",
            "ShallowFBCSPNet (E1, supervised)": "e1_dlloso_shallowfbcspnet_ea_default_seed42_physionet30.csv",
            "ATCNet (E1, supervised)": "e1_dlloso_atcnet_ea_default_seed42_physionet30.csv",
        },
    }
    fgmdm_files = {
        "bci2a": "factorial_fgmdm_t0.0_3.0_lwf.csv",
        "physionet": "physionet_comprehensive_ea_fgmdm.csv",
    }
    for dataset, archs in arch_files.items():
        fgmdm_dict = load_baseline_dict(R(fgmdm_files[dataset]))
        for arch_name, fname in archs.items():
            add(f"architecture_gap_vs_fgmdm", dataset, arch_name, load_baseline_dict(R(fname)),
                "EA+FgMDM", fgmdm_dict)
    # Task E1 is now fully complete (all 3 architectures, both datasets) -- no longer pending.
    pending.append("architecture_gap_vs_fgmdm (CBraMod/BIOT, both datasets): pending task E2.")
    pending.append("architecture_gap_vs_fgmdm (Riemannian Transformer v2, MAE-only, MAE+GRL on "
                    "bci2a): the only local per-subject CSVs for these methods on BCI-IV 2a are "
                    "stale 3-subject partial runs (see code comment above) that do not match the "
                    "manuscript's reported seed-42 numbers -- needs regenerating before this row "
                    "can be added honestly. The PhysioNet versions of these same three "
                    "comparisons ARE included above (confirmed exact matches, n=109).")

    if comparisons:
        comparisons = holm_bonferroni(comparisons)

    out_path = os.path.join(RESULTS_DIR, "e8_effect_sizes.csv")
    if comparisons:
        fieldnames = list(comparisons[0].keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparisons)
    n_sig = sum(1 for r in comparisons if r.get("holm_significant"))
    print(f"\nWrote {len(comparisons)} effect-size rows ({n_sig} Holm-significant at alpha=0.05) "
          f"to {out_path}")
    print(f"\n{len(pending)} rows still PENDING (dependencies not yet complete):")
    for p in pending:
        print(f"  - {p}")

    pending_path = os.path.join(RESULTS_DIR, "e8_effect_sizes_PENDING.txt")
    with open(pending_path, "w", encoding="utf-8") as f:
        f.write("\n".join(pending) + "\n")
    return comparisons, pending


if __name__ == "__main__":
    main()
