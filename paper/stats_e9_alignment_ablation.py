"""TMLR revision round 2, Phase-5 follow-up item 1(c): paired Wilcoxon (Holm within family),
bootstrap CI, and effect size for the EA on/off ablation (paper/eval_e9_alignment_ablation.py)
on FgMDM and MEKT-default, full PhysioNet population and Cho2017.

    python -m paper.stats_e9_alignment_ablation
"""
import csv
import os

from paper import config
from paper.stats_analysis import paired_comparison_from_dicts, holm_bonferroni, save_results

RESULTS_DIR = config.RESULTS_DIR


def load(name):
    with open(os.path.join(RESULTS_DIR, f"{name}.csv"), newline="") as f:
        return {int(r["held_out_subject"]): float(r["accuracy"]) for r in csv.DictReader(f)}


def main():
    pairs = [
        ("PhysioNet FgMDM: no-EA -> EA", "physionet_comprehensive_noea_fgmdm", "physionet_comprehensive_ea_fgmdm"),
        ("PhysioNet MEKT: no-EA -> EA", "physionet_comprehensive_noea_mekt_default", "physionet_comprehensive_mekt_default"),
        ("Cho2017 FgMDM: no-EA -> EA", "e5_cho2017_noea_fgmdm", "e5_cho2017_fgmdm"),
        ("Cho2017 MEKT: no-EA -> EA", "e5_cho2017_noea_mekt_default", "e5_cho2017_mekt_default"),
    ]
    results = []
    for name, noea_file, ea_file in pairs:
        noea = load(noea_file)
        ea = load(ea_file)
        common = set(noea) & set(ea)
        r = paired_comparison_from_dicts(name, {s: noea[s] for s in common}, {s: ea[s] for s in common})
        if r is not None:
            results.append(r)

    results = holm_bonferroni(results)
    save_results(results, "stats_e9_alignment_ablation.csv")
    print()
    for r in results:
        print(f"{r['comparison']}: n={r['n_subjects']} {r['mean_diff_pp']:+.2f}pp "
              f"95%CI=[{r['ci_lo_pp']:+.2f},{r['ci_hi_pp']:+.2f}]pp p={r['wilcoxon_p']:.4f} "
              f"Holm-sig={r['holm_significant']}")
    return results


if __name__ == "__main__":
    main()
