"""Runs the full MAE+CLUB+SCIN pipeline (proposed method) and appends row 4 to the results table.

    python -m paper.run_club_scin
"""
import csv
import os

from paper import config
from paper.eval_zero_shot_club_scin import run as run_zero_shot
from paper.train_club_scin import run as run_pretrain
from paper.train_subject_probe_club_scin import run as run_subject_probe


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("=== Pretraining: mae_club_scin ===")
    run_pretrain()

    print("=== Subject probe: mae_club_scin ===")
    probe_acc = run_subject_probe()

    print("=== Zero-shot LOSO eval: mae_club_scin ===")
    summary, _ = run_zero_shot()
    summary["subject_probe_acc"] = probe_acc

    path = os.path.join(config.RESULTS_DIR, "final_results_table.csv")
    row = {
        "model": "MAE+CLUB+SCIN (zero-shot)",
        "accuracy_mean": summary["accuracy_mean"],
        "accuracy_std": summary["accuracy_std"],
        "kappa_mean": summary["kappa_mean"],
        "kappa_std": summary["kappa_std"],
        "subject_probe_acc": probe_acc,
        "notes": "LOSO on BCI-IV 2a; encoder frozen, MLP head ensembled x5; SCIN+CLUB+VIB+augmentation",
    }
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nAppended row 4 to {path}")
    print(row)
    print(f"\nCROSSED 60%? {'YES' if summary['accuracy_mean'] > 0.60 else 'NO'} "
          f"-- exact accuracy: {summary['accuracy_mean']*100:.2f}%")


if __name__ == "__main__":
    main()
