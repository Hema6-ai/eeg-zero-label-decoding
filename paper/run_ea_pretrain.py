"""Tests whether alignment-aware pretraining (not just post-hoc alignment on a frozen encoder)
closes the gap the classical Riemannian baseline showed: MAE-only pretrained on Euclidean-Aligned
PhysioNet, evaluated zero-shot on Euclidean-Aligned BCI-IV 2a (LOSO, no calibration data).

    python -m paper.run_ea_pretrain
"""
import csv
import os

from paper import config
from paper.eval_zero_shot import run as run_zero_shot
from paper.train_pretrain import run as run_pretrain


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("=== Pretraining: mae_only + EA ===")
    run_pretrain("mae_only", align="ea")

    print("=== Zero-shot LOSO eval: mae_only + EA ===")
    summary, _ = run_zero_shot("mae_only", align="ea")

    path = os.path.join(config.RESULTS_DIR, "final_results_table.csv")
    row = {
        "model": "MAE-only + EA (zero-shot)",
        "accuracy_mean": summary["accuracy_mean"],
        "accuracy_std": summary["accuracy_std"],
        "kappa_mean": summary["kappa_mean"],
        "kappa_std": summary["kappa_std"],
        "subject_probe_acc": "",
        "notes": "LOSO on BCI-IV 2a; Euclidean Alignment applied per-subject (unsupervised) to "
                 "both PhysioNet pretraining and BNCI zero-shot data; linear head only",
    }
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)

    print(f"\nAppended to {path}")
    print(row)
    print(f"\nCROSSED 50%? {'YES' if summary['accuracy_mean'] > 0.50 else 'NO'} "
          f"-- CROSSED 60%? {'YES' if summary['accuracy_mean'] > 0.60 else 'NO'} "
          f"-- exact accuracy: {summary['accuracy_mean']*100:.2f}%")


if __name__ == "__main__":
    main()
