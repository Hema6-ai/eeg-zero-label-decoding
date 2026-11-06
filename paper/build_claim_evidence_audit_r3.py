"""Extends reports/claim_evidence_audit.csv with every quantitative claim added for Reviewer
c1um's matched-protocol SPDNet fairness check (TMLR revision round 3, Section 13.3). Appends to
the existing CSV rather than replacing it. Every row's numbers are read directly from
paper/results/stats_spdnet_matched_protocol.csv or the raw per-subject CSVs, not typed by hand.

    python -m paper.build_claim_evidence_audit_r3
"""
import csv
import os

import numpy as np

RESULTS_DIR = "paper/results"
OUT_PATH = "reports/claim_evidence_audit.csv"
STATS_PATH = os.path.join(RESULTS_DIR, "stats_spdnet_matched_protocol.csv")
FIELDNAMES = ["claim_id", "manuscript_claim", "section", "result_file", "script", "config",
              "prediction_artifact", "stats_script", "verification_status", "discrepancy_or_limitation"]

SECTION = "Section 13.3 (\"A fairness check: SPDNet under the matched protocol\"), response to Reviewer c1um"
SCRIPT = "paper/eval_spdnet_matched_protocol.py"
STATS_SCRIPT = "paper/stats_spdnet_matched_protocol.py"

ROWS = []

# --- Seed-averaged matched-protocol SPDNet accuracy at the full 109-subject PhysioNet scale ---
seed_files = [
    "e1_dlloso_spdnet_matched_ea_default_seed42_physionet_full.csv",
    "e1_dlloso_spdnet_matched_ea_default_seed0_physionet_full.csv",
    "e1_dlloso_spdnet_matched_ea_default_seed123_physionet_full.csv",
]
per_seed_accs = []
for fname in seed_files:
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, newline="") as f:
        per_seed_accs.append([float(r["accuracy"]) for r in csv.DictReader(f)])
subj_means = np.mean(per_seed_accs, axis=0)
acc_pct, std_pct, n = subj_means.mean() * 100, subj_means.std() * 100, len(subj_means)
claim_acc, claim_std, claim_n = 47.18, 12.16, 109
ok = abs(acc_pct - claim_acc) < 0.1 and abs(std_pct - claim_std) < 0.1 and n == claim_n
ROWS.append({
    "claim_id": "matched_spdnet_physionet_full_seedavg",
    "manuscript_claim": f"Matched-protocol SPDNet, PhysioNet-full (seed-avg 42/0/123): {claim_acc}% +/- {claim_std}%, n={claim_n}",
    "section": SECTION,
    "result_file": ", ".join(f"paper/results/{f}" for f in seed_files),
    "script": SCRIPT,
    "config": "ea=ea, window=default, tmin_tmax=(0.0,3.0), seeds=[42,0,123], subjects=PRETRAIN_SUBJECTS (109)",
    "prediction_artifact": ", ".join(f"paper/results/{f}" for f in seed_files),
    "stats_script": STATS_SCRIPT,
    "verification_status": "VERIFIED" if ok else "MISMATCH",
    "discrepancy_or_limitation": "" if ok else f"recomputed acc={acc_pct:.2f} std={std_pct:.2f} n={n}",
})

# --- Paired comparisons (diff, Wilcoxon p, Holm significance) read directly from the stats CSV,
# matched against the exact numbers in the Section 13.3 prose. ---
CLAIMED_COMPARISONS = {
    "[bci2a] matched SPDNet vs EEGNetv4": (-1.14, 0.5703),
    "[bci2a] matched SPDNet vs EA+FgMDM": (-1.81, 0.2500),
    "[bci2a] matched SPDNet vs MEKT": (-2.74, 0.0742),
    "[bci2a] matched SPDNet vs nested ensemble": (-4.30, 0.0977),
    "[physionet30] matched SPDNet vs EEGNetv4": (-0.11, 0.9053),
    "[physionet30] matched SPDNet vs EA+FgMDM": (-2.73, 0.0478),
    "[physionet30] matched SPDNet vs MEKT": (-3.62, 0.0157),
    "[physionet30] matched SPDNet vs full ensemble": (-4.96, 0.0293),
    "[physionet_full] matched SPDNet vs EEGNetv4": (-5.81, 0.0001),  # manuscript rounds to p<0.0001
    "[physionet_full] matched SPDNet vs EA+FgMDM": (-2.26, 0.0006),
    "[physionet_full] matched SPDNet vs MEKT": (-4.36, 0.0001),
    "[physionet_full] matched SPDNet vs full ensemble": (-4.77, 0.0001),
    "[bci2a] matched SPDNet vs original-protocol SPDNet": (1.52, 0.2626),
    "[physionet30] matched SPDNet vs original-protocol SPDNet": (-2.19, 0.1512),
    "[physionet_full] matched SPDNet vs original-protocol SPDNet": (-2.22, 0.0010),
}

with open(STATS_PATH, newline="") as f:
    stats_rows = {r["comparison"]: r for r in csv.DictReader(f)}

for comparison, (claim_diff, claim_p) in CLAIMED_COMPARISONS.items():
    row = stats_rows.get(comparison)
    status = "FAIL: comparison missing from stats CSV"
    discrepancy = ""
    if row is not None:
        actual_diff = float(row["mean_diff_pp"])
        actual_p = float(row["wilcoxon_p"])
        diff_ok = abs(actual_diff - claim_diff) < 0.05
        # For p<0.0001-style claims, only check the actual p is below that bound.
        p_ok = (actual_p < 0.0001) if claim_p == 0.0001 else (abs(actual_p - claim_p) < 0.0005)
        ok = diff_ok and p_ok
        status = "VERIFIED" if ok else "MISMATCH"
        if not ok:
            discrepancy = f"recomputed diff={actual_diff:.2f}pp p={actual_p:.4f} vs claimed {claim_diff}pp/{claim_p}"
    ROWS.append({
        "claim_id": f"matched_spdnet_stat_{comparison}",
        "manuscript_claim": f"{comparison}: diff={claim_diff:+.2f}pp, p={claim_p}",
        "section": SECTION,
        "result_file": "paper/results/stats_spdnet_matched_protocol.csv",
        "script": SCRIPT,
        "config": "Holm-corrected within each dataset's 4-comparison family (paired tests vs. original-protocol SPDNet are single comparisons, not Holm-corrected)",
        "prediction_artifact": "paper/results/stats_spdnet_matched_protocol.csv",
        "stats_script": STATS_SCRIPT,
        "verification_status": status,
        "discrepancy_or_limitation": discrepancy,
    })


def main():
    if not os.path.exists(OUT_PATH):
        raise FileNotFoundError(
            f"{OUT_PATH} does not exist -- run `python -m paper.build_claim_evidence_audit` first; "
            f"this script appends to that one, it does not replace it.")
    with open(OUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerows(ROWS)
    n_total = len(ROWS)
    n_failed = sum(1 for r in ROWS if not r["verification_status"].startswith("VERIFIED"))
    print(f"Appended {n_total} Phase-5-round-3 (Reviewer c1um) claim rows to {OUT_PATH}")
    print(f"{n_total - n_failed}/{n_total} VERIFIED; {n_failed} FAILED/MISMATCH")
    if n_failed:
        for r in ROWS:
            if not r["verification_status"].startswith("VERIFIED"):
                print(f"  - {r['claim_id']}: {r['verification_status']} :: {r['discrepancy_or_limitation']}")


if __name__ == "__main__":
    main()
