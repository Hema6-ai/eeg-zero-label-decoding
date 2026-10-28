"""Orchestrates the PhysioNet-comprehensive deep-learning comparison added in TMLR review response
(Critical Issue B): pretrains each Family 1/2 method on BCI-IV 2a (source="bnci", the reversed
direction from this project's default setup -- see paper/data.py::load_bnci_pretrain docstring for
why this avoids leakage) and then zero-shot LOSO-evaluates the frozen encoder across PhysioNet's
full subject population. Modeled directly on paper/run_multiseed_dl_sweep.py's
pretrain-then-evaluate-with-error-handling pattern, so a single method's failure does not abort the
rest, and partial progress is always saved.

Designed to run standalone on Kaggle (GPU, internet enabled for the MOABB/PhysioNet download) --
see kaggle_kernel/physionet_deep_sweep_push/run_kernel.py.

    python -m paper.run_physionet_deep_sweep
"""
import csv
import os
import time
import traceback

from paper import config


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    rows = []

    def safe_run(name, fn):
        try:
            print(f"\n{'=' * 20} {name} {'=' * 20}", flush=True)
            t0 = time.time()
            result = fn()
            elapsed = time.time() - t0
            print(f"[{name}]: SUCCESS -> {result} ({elapsed:.1f}s)", flush=True)
            return result, elapsed, None
        except Exception as e:
            print(f"[{name}]: FAILED -- {e}", flush=True)
            traceback.print_exc()
            return None, None, str(e)

    def record(method, result, elapsed, error):
        row = {"method": method, "status": "ok" if result else "failed", "error": error or ""}
        if result:
            row.update(result)
        if elapsed is not None:
            row["elapsed_seconds"] = elapsed
        rows.append(row)
        _flush()

    def _flush():
        out_path = os.path.join(config.RESULTS_DIR, "physionet_loso_deep_summary.csv")
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # --- MAE-only / MAE+GRL ---
    from paper.eval_physionet_loso_deep import run_mae, run_riemannian_transformer_v2, run_spdnet, run_supcon
    from paper.train_pretrain import run as run_pretrain
    from paper.train_riemannian_transformer_v2 import run as run_train_rt2
    from paper.train_spdnet import run as run_train_spdnet
    from paper.train_supcon import run as run_train_supcon

    for variant in ("mae_only", "mae_grl"):
        def _fn(v=variant):
            run_pretrain(v, source="bnci")
            summary, _ = run_mae(v)
            return summary
        result, elapsed, error = safe_run(f"pretrain(bnci)+physionet_loso_{variant}", _fn)
        record(variant, result, elapsed, error)

    # --- SupCon ---
    def _fn():
        run_train_supcon(source="bnci")
        summary, _ = run_supcon()
        return summary
    result, elapsed, error = safe_run("pretrain(bnci)+physionet_loso_supcon", _fn)
    record("supcon", result, elapsed, error)

    # --- SPDNet ---
    def _fn():
        run_train_spdnet(source="bnci")
        summary, _ = run_spdnet()
        return summary
    result, elapsed, error = safe_run("pretrain(bnci)+physionet_loso_spdnet", _fn)
    record("spdnet", result, elapsed, error)

    # --- Riemannian Transformer v2 ---
    def _fn():
        run_train_rt2(source="bnci")
        summary, _ = run_riemannian_transformer_v2()
        return summary
    result, elapsed, error = safe_run("pretrain(bnci)+physionet_loso_riemannian_transformer_v2", _fn)
    record("riemannian_transformer_v2", result, elapsed, error)

    print(f"\nSaved {len(rows)}/5 method results to "
          f"{os.path.join(config.RESULTS_DIR, 'physionet_loso_deep_summary.csv')}")
    return rows


if __name__ == "__main__":
    main()
