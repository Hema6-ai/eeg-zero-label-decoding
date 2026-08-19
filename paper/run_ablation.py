"""Full ablation pipeline: [MAE only] vs [MAE + GRL] vs [per-subject EEGNet baseline].

    python -m paper.run_ablation

Runs, in order: PhysioNet pretraining (both variants) -> post-hoc subject probe (both
variants) -> zero-shot LOSO eval on BNCI2014_001 (both variants) -> EEGNet baseline ->
aggregated results table.
"""
import csv
import os

from paper import config
from paper.eval_zero_shot import run as run_zero_shot
from paper.train_eegnet_baseline import run as run_eegnet_baseline
from paper.train_pretrain import run as run_pretrain
from paper.train_subject_probe import run as run_subject_probe


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    rows = []

    for variant in ("mae_only", "mae_grl"):
        print(f"\n=== Pretraining: {variant} ===")
        run_pretrain(variant)

        print(f"\n=== Subject probe: {variant} ===")
        probe_acc = run_subject_probe(variant)

        print(f"\n=== Zero-shot LOSO eval: {variant} ===")
        summary, _ = run_zero_shot(variant)
        summary["subject_probe_acc"] = probe_acc
        rows.append(summary)

    print("\n=== EEGNet per-subject baseline ===")
    eegnet_summary, _ = run_eegnet_baseline()
    eegnet_summary["variant"] = "eegnet_baseline"
    rows.append(eegnet_summary)

    path = os.path.join(config.RESULTS_DIR, "ablation_summary.csv")
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved ablation summary to {path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
