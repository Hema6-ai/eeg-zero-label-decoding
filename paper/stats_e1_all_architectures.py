"""TMLR revision round 2, Phase-5 follow-up: extends paper/stats_eegnet_comprehensive.py's
methodology to ShallowFBCSPNet and ATCNet (E1's other two architectures), on both datasets,
against the same baseline family (EA+FgMDM, MEKT-default, SPDNet, and -- where available -- the
E4 nested-selected ensemble). stats_eegnet_comprehensive.py already covers EEGNetv4; this script
exists because the ShallowFBCSPNet/ATCNet PhysioNet comparisons were previously reported in
PHASE_2_PROGRESS.md as "computed ad hoc," with no saved script+CSV backing them -- a violation of
this project's own rule. Regenerates them properly, saved to a real CSV.

    python -m paper.stats_e1_all_architectures
"""
import csv
import os

from paper import config
from paper.run_dl_loso_sweep import load_baseline_dict
from paper.stats_analysis import holm_bonferroni, paired_comparison_from_dicts, save_results

RESULTS_DIR = config.RESULTS_DIR
ARCHS = ["shallowfbcspnet", "atcnet"]


def load_fold(path, restrict_to=None):
    return load_baseline_dict(path, restrict_to=restrict_to)


def load_e4_nested_ensemble(path):
    with open(path, newline="") as f:
        return {int(r["held_out_subject"]): float(r["ensemble_nested_accuracy"])
                for r in csv.DictReader(f)}


def main():
    R = lambda name: os.path.join(RESULTS_DIR, name)
    results = []

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
    e4_bci2a_path = R("e4_nested_selection_bci2a_reducedgrid.csv")
    e4_pn_path = R("e4_nested_selection_physionet30_reducedgrid.csv")

    for arch in ARCHS:
        bci2a_ea = load_fold(R(f"e1_dlloso_{arch}_ea_default_seed42_bci2a.csv"))
        pn_ea = load_fold(R(f"e1_dlloso_{arch}_ea_default_seed42_physionet30.csv"))
        print(f"\n{'='*10} {arch}+EA (seed 42, default window) {'='*10}")
        print(f"BCI-IV 2a: n={len(bci2a_ea)}; PhysioNet30: n={len(pn_ea)} "
              f"(fixed seed-42-sampled 30-subject subset, NOT full 109)")

        family_bci2a, family_pn = [], []
        for name, path in bci2a_baselines.items():
            b = load_fold(path)
            r = paired_comparison_from_dicts(f"[{arch}-BCI2a] {name} -> {arch}+EA", b, bci2a_ea)
            if r:
                r.update({"family": f"{arch}_vs_classical", "dataset": "bci2a", "baseline": name})
                family_bci2a.append(r)
        for name, path in pn_baselines.items():
            b = load_fold(path, restrict_to=set(pn_ea.keys()))
            r = paired_comparison_from_dicts(f"[{arch}-PhysioNet30] {name} -> {arch}+EA", b, pn_ea)
            if r:
                r.update({"family": f"{arch}_vs_classical", "dataset": "physionet30", "baseline": name})
                family_pn.append(r)
        family_bci2a = holm_bonferroni(family_bci2a)
        family_pn = holm_bonferroni(family_pn)
        results += family_bci2a + family_pn

        family_nested = []
        if os.path.exists(e4_bci2a_path):
            nested_ens = load_e4_nested_ensemble(e4_bci2a_path)
            r = paired_comparison_from_dicts(f"[{arch}-BCI2a] nested-ensemble -> {arch}+EA",
                                              nested_ens, bci2a_ea)
            if r:
                r.update({"family": f"{arch}_vs_nested_ensemble", "dataset": "bci2a",
                           "baseline": "E4 nested-selected ensemble"})
                family_nested.append(r)
        if os.path.exists(e4_pn_path):
            nested_ens_pn = load_e4_nested_ensemble(e4_pn_path)
            r = paired_comparison_from_dicts(f"[{arch}-PhysioNet30] nested-ensemble -> {arch}+EA",
                                              nested_ens_pn, pn_ea)
            if r:
                r.update({"family": f"{arch}_vs_nested_ensemble", "dataset": "physionet30",
                           "baseline": "E4 nested-selected ensemble"})
                family_nested.append(r)
        family_nested = holm_bonferroni(family_nested)
        results += family_nested

    save_results(results, "stats_e1_all_architectures.csv")
    print(f"\nWrote {len(results)} comparisons to "
          f"{os.path.join(RESULTS_DIR, 'stats_e1_all_architectures.csv')}")
    return results


if __name__ == "__main__":
    main()
