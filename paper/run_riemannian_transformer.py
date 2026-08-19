"""Full Riemannian Transformer pipeline: Stage 1 (PhysioNet pretrain) -> Stage 2 (BCI-IV 2a
zero-shot LOSO eval) -> append result to the results table.

    python -m paper.run_riemannian_transformer
"""
import csv
import os

from paper import config
from paper.eval_zero_shot_riemannian_transformer import run as run_zero_shot
from paper.train_riemannian_transformer import run as run_pretrain


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("=== Stage 1: Pretraining Riemannian Transformer on PhysioNet ===")
    run_pretrain()

    print("=== Stage 2: Zero-shot LOSO eval on BCI-IV 2a ===")
    summary, fold_results = run_zero_shot()

    path = os.path.join(config.RESULTS_DIR, "final_results_table.csv")
    row = {
        "model": "Riemannian Transformer (zero-shot)",
        "accuracy_mean": summary["accuracy_mean"],
        "accuracy_std": summary["accuracy_std"],
        "kappa_mean": summary["kappa_mean"],
        "kappa_std": summary["kappa_std"],
        "subject_probe_acc": "",
        "notes": "LOSO on BCI-IV 2a; EA + Covariances(lwf) + TangentSpace(fit on source only) "
                 "+ frozen 4-layer transformer encoder (pretrained on PhysioNet, 3 classes) + MLP head",
    }
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)

    print(f"\nAppended to {path}")
    print(row)
    acc = summary["accuracy_mean"]
    print(f"\nBeat FgMDM (47.45%)? {'YES' if acc > 0.4745 else 'NO'}")
    print(f"Beat CSP+LR (53.24%)? {'YES' if acc > 0.5324 else 'NO'}")
    print(f"CROSSED 60%? {'YES' if acc > 0.60 else 'NO'} -- exact accuracy: {acc*100:.2f}%")
    print("\nPer-fold breakdown:")
    for r in fold_results:
        print(f"  subject {r['held_out_subject']}: acc={r['accuracy']:.4f} kappa={r['kappa']:.4f}")


if __name__ == "__main__":
    main()
