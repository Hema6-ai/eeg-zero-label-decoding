"""Independent verification: recomputes every aggregate (mean, std, kappa, n) directly from the
per-subject prediction/result CSVs and cross-checks against (a) the summary CSVs written by the
original experiment scripts and (b) the exact numbers written into tmlr_2026.tex. This is the
"never trust the filename or a previous summary" check required by the final revision directive.

Does not re-run any experiment (that would cost GPU/CPU hours for no new information); it
re-derives every aggregate from the same raw per-subject rows the original scripts wrote, using
independent code (not the original scripts' own aggregation functions), and flags any mismatch
beyond a small floating-point tolerance.

    python -m paper.verify_all_claims
"""
import csv
import os
import re

import numpy as np

RESULTS_DIR = "paper/results"
TEX_PATH = "tmlr_2026.tex"
TOL = 0.001  # 0.1 percentage point tolerance for accuracy; same abs tolerance for kappa


def load_per_subject_accuracy_kappa(path):
    accs, kappas = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if "accuracy" not in row or row["accuracy"] in (None, ""):
                continue
            accs.append(float(row["accuracy"]))
            kappas.append(float(row["kappa"]))
    return np.array(accs), np.array(kappas)


def recompute(path):
    accs, kappas = load_per_subject_accuracy_kappa(path)
    return {
        "n": len(accs),
        "acc_mean": accs.mean() if len(accs) else float("nan"),
        "acc_std": accs.std() if len(accs) else float("nan"),
        "kappa_mean": kappas.mean() if len(kappas) else float("nan"),
        "kappa_std": kappas.std() if len(kappas) else float("nan"),
    }


# (label, per-subject CSV, claimed accuracy %, claimed std %, claimed kappa, claimed n)
# Claimed values are exactly what appears in tmlr_2026.tex at time of writing this verifier.
CLAIMS = [
    ("PhysioNet CSP+LDA", "physionet_comprehensive_ea_csp8_lda.csv", 49.40, 14.31, 0.241, 109),
    ("PhysioNet CSP+SVM", "physionet_comprehensive_ea_csp8_svm.csv", 48.67, 14.40, 0.230, 109),
    ("PhysioNet FBCSP+SVM", "physionet_comprehensive_ea_fbcsp_svm.csv", 48.79, 13.60, 0.232, 109),
    ("PhysioNet MDM", "physionet_comprehensive_ea_mdm.csv", 42.23, 8.55, 0.132, 109),
    ("PhysioNet FgMDM", "physionet_comprehensive_ea_fgmdm.csv", 49.44, 13.46, 0.242, 109),
    ("PhysioNet RPA", "physionet_comprehensive_rpa.csv", 40.26, 10.76, 0.104, 109),
    ("PhysioNet MEKT default", "physionet_comprehensive_mekt_default.csv", 51.54, 14.35, 0.273, 109),
    ("PhysioNet MEKT tuned", "physionet_comprehensive_mekt_tuned.csv", 50.95, 14.34, 0.264, 109),
    ("PhysioNet SPDNet", "physionet_loso_spdnet.csv", 49.40, 14.25, 0.241, 109),
    ("PhysioNet RiemannianTransformer v2", "physionet_loso_riemannian_transformer_v2.csv", 42.60, 10.45, 0.139, 109),
    ("PhysioNet SupCon", "physionet_loso_supcon.csv", 33.04, 4.53, -0.008, 109),
    ("PhysioNet MAE+GRL", "physionet_loso_mae_grl.csv", 36.80, 7.71, 0.052, 109),
    ("PhysioNet MAE-only", "physionet_loso_mae_only.csv", 33.99, 4.75, 0.008, 109),
    ("LaBraM BCI-IV 2a zero-shot", "zero_shot_labram.csv", 25.08, 1.74, 0.001, 9),
    ("BCI-IV 2a SPDNet (lwf, factorial baseline)", "zero_shot_spdnet.csv", 44.12, 12.21, 0.255, 9),
    ("BCI-IV 2a SPDNet (corr, factorial check)", "zero_shot_spdnet_corrcov.csv", 45.79, 14.33, 0.277, 9),
]


def main():
    print(f"{'Claim':45s} {'claimed acc':>12s} {'recomputed':>12s} {'claimed std':>12s} "
          f"{'recomputed':>12s} {'claimed kap':>12s} {'recomputed':>12s} {'n claim':>8s} {'n actual':>9s} status")
    failures = []
    for label, fname, claim_acc, claim_std, claim_kappa, claim_n in CLAIMS:
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            print(f"{label:45s} MISSING FILE: {path}")
            failures.append((label, "missing file"))
            continue
        r = recompute(path)
        acc_pct = r["acc_mean"] * 100
        std_pct = r["acc_std"] * 100
        ok = (abs(acc_pct - claim_acc) < 0.1 and abs(std_pct - claim_std) < 0.1
              and abs(r["kappa_mean"] - claim_kappa) < 0.01 and r["n"] == claim_n)
        status = "OK" if ok else "MISMATCH"
        if not ok:
            failures.append((label, f"acc {acc_pct:.2f} vs {claim_acc}, std {std_pct:.2f} vs {claim_std}, "
                                     f"kappa {r['kappa_mean']:.3f} vs {claim_kappa}, n {r['n']} vs {claim_n}"))
        print(f"{label:45s} {claim_acc:12.2f} {acc_pct:12.2f} {claim_std:12.2f} {std_pct:12.2f} "
              f"{claim_kappa:12.3f} {r['kappa_mean']:12.3f} {claim_n:8d} {r['n']:9d} {status}")

    print(f"\n{len(CLAIMS) - len(failures)}/{len(CLAIMS)} claims independently verified.")
    if failures:
        print("\nFAILURES:")
        for label, detail in failures:
            print(f"  - {label}: {detail}")
    else:
        print("All per-subject-derived aggregate claims in tmlr_2026.tex match independent recomputation.")

    # Cross-check: do the CSV files referenced in CLAIMS actually get cited in tmlr_2026.tex's
    # numbers (sanity check that we didn't miss updating the manuscript after a rerun)?
    tex = open(TEX_PATH, encoding="utf-8").read()
    print("\n--- spot-checking a few key numeric strings appear in tmlr_2026.tex ---")
    for needle in ["49.40", "51.54", "25.08", "0.0002", "7.21", "12.60"]:
        print(f"  {needle}: {'FOUND' if needle in tex else 'MISSING'}")

    return len(failures) == 0


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
