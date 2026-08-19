"""Runs the full SupCon pipeline (pretrain -> subject probe -> zero-shot eval) and appends the
result to the results table.

    python -m paper.run_supcon
"""
import csv
import os

from paper import config
from paper.eval_zero_shot_supcon import run as run_zero_shot
from paper.train_subject_probe_supcon import run as run_subject_probe
from paper.train_supcon import run as run_pretrain


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("=== Pretraining: SupCon ===")
    run_pretrain()

    print("=== Subject probe: SupCon ===")
    probe_acc = run_subject_probe()

    print("=== Zero-shot LOSO eval: SupCon ===")
    summary, _ = run_zero_shot()

    path = os.path.join(config.RESULTS_DIR, "final_results_table.csv")
    row = {
        "model": "SupCon (zero-shot)",
        "accuracy_mean": summary["accuracy_mean"],
        "accuracy_std": summary["accuracy_std"],
        "kappa_mean": summary["kappa_mean"],
        "kappa_std": summary["kappa_std"],
        "subject_probe_acc": probe_acc,
        "notes": "LOSO on BCI-IV 2a; encoder frozen, MLP head; supervised contrastive "
                 "pretraining on PhysioNet (feet/left_hand/right_hand only, no tongue class available)",
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
