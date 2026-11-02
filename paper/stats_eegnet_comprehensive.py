"""TMLR revision round 2, task D (+ F.ii): comprehensive, Holm-corrected statistical comparison
of EEGNetv4 against every relevant baseline, on both datasets, with the EXACT variant identified
(no ambiguity about which of the 4 ea/window cells or how many seeds/subjects).

**Which EEGNetv4 variant is "the tie" reported so far**: seed 42 only (the primary/headline seed
convention used throughout this paper for statistical testing, consistent with how every other
Family-1/2 method's primary comparison is done at seed 42, with the other 4 seeds reserved for
the separate stability check), EA-aligned (align="ea"), default window ([0,3]s, matching every
classical baseline's default-window convention). n=9 subjects on BCI-IV 2a (full LOSO); n=30
subjects on the fixed, seed-42-sampled PhysioNet subset (paper/run_dl_loso_sweep.py's disclosed
fallback, not the full 109).

Runs 3 families of comparison, each independently Holm-corrected within its own family (not
pooled across families, since they answer different questions):

  1. EEGNetv4+EA vs. {EA+FgMDM, MEKT-default, SPDNet}, on each dataset separately (family size 3
     per dataset).
  2. EEGNetv4+EA vs. the TASK-E4 NESTED-SELECTED ensemble (not the in-sample candidate-set
     ensemble, which the task's own critique correctly notes carries selection bias) -- BCI-IV 2a
     only, since E4's PhysioNet repeat has not landed yet as of this script's writing. Uses
     `ensemble_nested_accuracy` from paper/results/e4_nested_selection_bci2a_reducedgrid.csv, one
     row per outer-fold held-out subject -- this IS a genuine, leakage-free per-subject estimate
     (the nested procedure never saw that subject during selection), unlike the original
     candidate-set ensemble.
  3. EEGNetv4+EA vs. EEGNetv4 WITHOUT EA (same seed, same window), on each dataset -- isolates the
     alignment effect specifically on this deep-learning architecture, directly comparable to the
     classical-method alignment-on/off result in E8's `alignment_on_off` row.

    python -m paper.stats_eegnet_comprehensive
"""
import csv
import os

from paper import config
from paper.run_dl_loso_sweep import load_baseline_dict
from paper.stats_analysis import holm_bonferroni, paired_comparison_from_dicts, save_results

RESULTS_DIR = config.RESULTS_DIR


def load_fold(path, restrict_to=None):
    return load_baseline_dict(path, restrict_to=restrict_to)


def load_e4_nested_ensemble(path):
    with open(path, newline="") as f:
        return {int(r["held_out_subject"]): float(r["ensemble_nested_accuracy"])
                for r in csv.DictReader(f)}


def main():
    R = lambda name: os.path.join(RESULTS_DIR, name)
    results = []

    print("=" * 10, "EEGNetv4+EA (seed 42, default window) -- exact variant identification", "=" * 10)
    bci2a_ea = load_fold(R("e1_dlloso_eegnetv4_ea_default_seed42_bci2a.csv"))
    bci2a_noea = load_fold(R("e1_dlloso_eegnetv4_none_default_seed42_bci2a.csv"))
    pn_ea = load_fold(R("e1_dlloso_eegnetv4_ea_default_seed42_physionet30.csv"))
    pn_noea = load_fold(R("e1_dlloso_eegnetv4_none_default_seed42_physionet30.csv"))
    print(f"BCI-IV 2a: n={len(bci2a_ea)} subjects, seed 42 only, EA-aligned, window=[0,3]s")
    print(f"PhysioNet: n={len(pn_ea)} subjects (fixed seed-42-sampled 30-subject subset, "
          f"NOT the full 109), seed 42 only, EA-aligned, window=[0,3]s")

    print("\n" + "=" * 10, "Family 1: EEGNetv4+EA vs. classical/DA baselines", "=" * 10)
    bci2a_baselines = {
        "EA+FgMDM": R("factorial_fgmdm_t0.0_3.0_lwf.csv"),
        "MEKT": R("zero_shot_mekt.csv"),
        "SPDNet": R("zero_shot_spdnet.csv"),
    }
    pn_baselines = {
        "EA+FgMDM": R("physionet_comprehensive_ea_fgmdm.csv"),
        "MEKT": R("physionet_comprehensive_mekt_default.csv"),
        "SPDNet": R("physionet_loso_spdnet.csv"),
    }
    family1_bci2a, family1_pn = [], []
    for name, path in bci2a_baselines.items():
        b = load_fold(path)
        r = paired_comparison_from_dicts(f"[Family1-BCI2a] {name} -> EEGNetv4+EA", b, bci2a_ea)
        if r:
            r.update({"family": "eegnet_vs_classical", "dataset": "bci2a", "baseline": name})
            family1_bci2a.append(r)
    for name, path in pn_baselines.items():
        b = load_fold(path, restrict_to=set(pn_ea.keys()))
        r = paired_comparison_from_dicts(f"[Family1-PhysioNet30] {name} -> EEGNetv4+EA", b, pn_ea)
        if r:
            r.update({"family": "eegnet_vs_classical", "dataset": "physionet30", "baseline": name})
            family1_pn.append(r)
    family1_bci2a = holm_bonferroni(family1_bci2a)
    family1_pn = holm_bonferroni(family1_pn)
    results += family1_bci2a + family1_pn

    print("\n" + "=" * 10,
          "Family 2: EEGNetv4+EA vs. TASK-E4 NESTED-SELECTED ensemble (not in-sample)", "=" * 10)
    e4_path = R("e4_nested_selection_bci2a_reducedgrid.csv")
    family2 = []
    if os.path.exists(e4_path):
        nested_ens = load_e4_nested_ensemble(e4_path)
        r = paired_comparison_from_dicts("[Family2-BCI2a] nested-ensemble -> EEGNetv4+EA",
                                          nested_ens, bci2a_ea)
        if r:
            r.update({"family": "eegnet_vs_nested_ensemble", "dataset": "bci2a",
                       "baseline": "E4 nested-selected ensemble"})
            family2.append(r)
    else:
        print("[Family2] SKIPPED -- e4_nested_selection_bci2a_reducedgrid.csv not found")
    pn_e4_path = R("e4_nested_selection_physionet30_reducedgrid.csv")
    if os.path.exists(pn_e4_path):
        nested_ens_pn = load_e4_nested_ensemble(pn_e4_path)
        r = paired_comparison_from_dicts("[Family2-PhysioNet30] nested-ensemble -> EEGNetv4+EA",
                                          nested_ens_pn, pn_ea)
        if r:
            r.update({"family": "eegnet_vs_nested_ensemble", "dataset": "physionet30",
                       "baseline": "E4 nested-selected ensemble"})
            family2.append(r)
    else:
        print("[Family2-PhysioNet30] SKIPPED -- E4's PhysioNet repeat has not landed yet "
              "(task instruction: do not block on it, rerun this script once it does)")
    family2 = holm_bonferroni(family2)
    results += family2

    print("\n" + "=" * 10, "Family 3: EEGNetv4 EA on/off (alignment effect on this architecture)",
          "=" * 10)
    family3 = []
    r = paired_comparison_from_dicts("[Family3-BCI2a] EEGNetv4(no EA) -> EEGNetv4(EA)",
                                      bci2a_noea, bci2a_ea)
    if r:
        r.update({"family": "eegnet_alignment_effect", "dataset": "bci2a", "baseline": "no-EA"})
        family3.append(r)
    r = paired_comparison_from_dicts("[Family3-PhysioNet30] EEGNetv4(no EA) -> EEGNetv4(EA)",
                                      pn_noea, pn_ea)
    if r:
        r.update({"family": "eegnet_alignment_effect", "dataset": "physionet30", "baseline": "no-EA"})
        family3.append(r)
    family3 = holm_bonferroni(family3)
    results += family3

    save_results(results, "stats_eegnet_comprehensive.csv")
    print(f"\nWrote {len(results)} comparisons to "
          f"{os.path.join(RESULTS_DIR, 'stats_eegnet_comprehensive.csv')}")
    return results


if __name__ == "__main__":
    main()
