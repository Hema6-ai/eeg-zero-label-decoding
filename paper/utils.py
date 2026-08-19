"""Metrics, checkpointing, and epoch logging shared across training scripts."""
import csv
import os

import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score


def classification_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }


class EpochLogger:
    """Appends one CSV row per epoch; creates the header from the first row's keys."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._header_written = os.path.exists(path)

    def log(self, **fields) -> None:
        write_header = not self._header_written
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(fields.keys()))
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(fields)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in fields.items()))


def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def maybe_save_periodic_and_best(model, optimizer, epoch: int, metric: float,
                                  best_metric: float, checkpoint_dir: str,
                                  run_name: str, checkpoint_every: int, higher_is_better: bool = False):
    """Saves every `checkpoint_every` epochs and separately tracks the best metric checkpoint.

    For MAE pretraining, `metric` is typically the reconstruction loss (lower is better).
    """
    is_best = metric < best_metric if not higher_is_better else metric > best_metric
    if is_best:
        best_metric = metric
        save_checkpoint(
            {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "metric": metric},
            os.path.join(checkpoint_dir, f"{run_name}_best.pt"),
        )
    if epoch % checkpoint_every == 0:
        save_checkpoint(
            {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "metric": metric},
            os.path.join(checkpoint_dir, f"{run_name}_epoch{epoch}.pt"),
        )
    return best_metric
