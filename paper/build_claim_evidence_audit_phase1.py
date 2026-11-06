"""TMLR final-pass Phase 1 robustness check: extends reports/claim_evidence_audit.csv with every
quantitative claim in the "Robustness to hyperparameter tuning" paragraph (Section 13.3) --
the tuned matched-protocol SPDNet comparisons and the tuned-vs-untuned/tuned-vs-original
protocol-effect checks. Appends to the existing CSV; every number is read directly from
paper/results/stats_spdnet_tuned_protocol.csv, not typed by hand.

    python -m paper.build_claim_evidence_audit_phase1
"""
import csv
import os

RESULTS_DIR = "paper/results"
OUT_PATH = "reports/claim_evidence_audit.csv"
STATS_PATH = os.path.join(RESULTS_DIR, "stats_spdnet_tuned_protocol.csv")
FIELDNAMES = ["claim_id", "manuscript_claim", "section", "result_file", "script", "config",
              "prediction_artifact", "stats_script", "verification_status", "discrepancy_or_limitation"]

SECTION = "Section 13.3 (\"Robustness to hyperparameter tuning\"), Phase 1 final-pass check"
SCRIPT = "paper/tune_spdnet_matched_protocol.py, paper/eval_spdnet_matched_protocol_tuned.py"
STATS_SCRIPT = "paper/stats_spdnet_tuned_protocol.py"

CLAIMED_COMPARISONS = {
    "[physionet_full] tuned SPDNet vs EEGNetv4": (-3.82, 0.0001),
    "[physionet_full] tuned SPDNet vs EA+FgMDM": (-0.26, 0.4665),
    "[physionet_full] tuned SPDNet vs MEKT": (-2.37, 0.0001),
    "[physionet_full] tuned SPDNet vs full ensemble": (-2.78, 0.0001),
    "[physionet_full] tuned SPDNet vs untuned matched-protocol SPDNet": (1.99, 0.0002),
    "[bci2a] tuned SPDNet vs EEGNetv4": (0.98, 0.6523),
    "[physionet30] tuned SPDNet vs untuned matched-protocol SPDNet": (-1.44, 0.4236),
}

with open(STATS_PATH, newline="") as f:
    stats_rows = {r["comparison"]: r for r in csv.DictReader(f)}

ROWS = []
for comparison, (claim_diff, claim_p) in CLAIMED_COMPARISONS.items():
    row = stats_rows.get(comparison)
    status = "FAIL: comparison missing from stats CSV"
    discrepancy = ""
    if row is not None:
        actual_diff = float(row["mean_diff_pp"])
        actual_p = float(row["wilcoxon_p"])
        diff_ok = abs(actual_diff - claim_diff) < 0.05
        p_ok = (actual_p < 0.0001) if claim_p == 0.0001 else (abs(actual_p - claim_p) < 0.0005)
        ok = diff_ok and p_ok
        status = "VERIFIED" if ok else "MISMATCH"
        if not ok:
            discrepancy = f"recomputed diff={actual_diff:.2f}pp p={actual_p:.4f} vs claimed {claim_diff}pp/{claim_p}"
    ROWS.append({
        "claim_id": f"tuned_spdnet_stat_{comparison}",
        "manuscript_claim": f"{comparison}: diff={claim_diff:+.2f}pp, p={claim_p}",
        "section": SECTION,
        "result_file": "paper/results/stats_spdnet_tuned_protocol.csv",
        "script": SCRIPT,
        "config": "dims=(22,22), lr=1e-3 (selected by paper/tune_spdnet_matched_protocol.py on PhysioNet30 source-validation accuracy)",
        "prediction_artifact": "paper/results/stats_spdnet_tuned_protocol.csv",
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
    print(f"Appended {n_total} Phase 1 (hyperparameter robustness check) claim rows to {OUT_PATH}")
    print(f"{n_total - n_failed}/{n_total} VERIFIED; {n_failed} FAILED/MISMATCH")
    if n_failed:
        for r in ROWS:
            if not r["verification_status"].startswith("VERIFIED"):
                print(f"  - {r['claim_id']}: {r['verification_status']} :: {r['discrepancy_or_limitation']}")


if __name__ == "__main__":
    main()
