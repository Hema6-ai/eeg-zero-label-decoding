"""Multi-seed retraining of every Family 1/2 (deep-learning) method, for reviewer-facing
robustness validation. Runs all 6 methods (MAE-only, MAE+GRL, MAE+CLUB+SCIN+VIB, Riemannian
Transformer v1, Riemannian Transformer v2, SPDNet) back-to-back for ONE seed per invocation, so
PhysioNet's download/cache is shared across all 6 within a single Kaggle session instead of paying
the download cost 6 times. Seed 42 was already run individually earlier in this project; this
script is meant to be run once per ADDITIONAL seed (e.g. 0, then 123) on Kaggle.

Unlike MEKT (fully deterministic), these are real neural network trainings -- weight
initialization, data shuffling, dropout, and (for MAE) mask sampling are all genuinely
seed-dependent, so config.SEED here is a real source of variation, not a no-op.

    python -m paper.run_multiseed_dl_sweep --seed 0
"""
import argparse
import csv
import os
import traceback

from paper import config


def run_one(seed: int):
    config.SEED = seed
    config.CHECKPOINT_DIR = f"/kaggle/working/checkpoints_seed{seed}" if os.path.exists("/kaggle") else f"paper/checkpoints_seed{seed}"
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    rows = []

    def safe_run(name, fn):
        try:
            print(f"\n{'='*20} {name} (seed={seed}) {'='*20}")
            result = fn()
            print(f"[{name}] seed={seed}: SUCCESS -> {result}")
            return result
        except Exception as e:
            print(f"[{name}] seed={seed}: FAILED -- {e}")
            traceback.print_exc()
            return None

    # --- MAE-only / MAE+GRL ---
    from paper.eval_zero_shot import run as run_zero_shot
    from paper.train_pretrain import run as run_pretrain

    for variant in ("mae_only", "mae_grl"):
        def _fn(v=variant):
            run_pretrain(v)
            summary, _ = run_zero_shot(v)
            return summary
        result = safe_run(f"pretrain+eval_{variant}", _fn)
        if result:
            rows.append({"method": variant, "seed": seed, **result})

    # --- MAE+CLUB+SCIN+VIB ---
    from paper.eval_zero_shot_club_scin import run as run_zero_shot_club
    from paper.train_club_scin import run as run_train_club

    def _fn():
        run_train_club()
        summary, _ = run_zero_shot_club()
        return summary
    result = safe_run("mae_club_scin_vib", _fn)
    if result:
        rows.append({"method": "mae_club_scin_vib", "seed": seed, **result})

    # --- Riemannian Transformer v1 ---
    from paper.eval_zero_shot_riemannian_transformer import run as run_zero_shot_rt1
    from paper.train_riemannian_transformer import run as run_train_rt1

    def _fn():
        run_train_rt1()
        summary, _ = run_zero_shot_rt1()
        return summary
    result = safe_run("riemannian_transformer_v1", _fn)
    if result:
        rows.append({"method": "riemannian_transformer_v1", "seed": seed, **result})

    # --- Riemannian Transformer v2 ---
    from paper.eval_zero_shot_riemannian_transformer_v2 import run as run_zero_shot_rt2
    from paper.train_riemannian_transformer_v2 import run as run_train_rt2

    def _fn():
        run_train_rt2()
        summary, _ = run_zero_shot_rt2()
        return summary
    result = safe_run("riemannian_transformer_v2", _fn)
    if result:
        rows.append({"method": "riemannian_transformer_v2", "seed": seed, **result})

    # --- SPDNet ---
    from paper.eval_zero_shot_spdnet import run as run_zero_shot_spdnet
    from paper.train_spdnet import run as run_train_spdnet

    def _fn():
        run_train_spdnet()
        summary, _ = run_zero_shot_spdnet()
        return summary
    result = safe_run("spdnet", _fn)
    if result:
        rows.append({"method": "spdnet", "seed": seed, **result})

    out_path = os.path.join(config.RESULTS_DIR, f"multiseed_dl_seed{seed}.csv")
    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nSaved {len(rows)}/6 method results for seed={seed} to {out_path}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run_one(args.seed)
