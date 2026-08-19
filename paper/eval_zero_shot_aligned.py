"""Tests whether EA/RA alignment improves zero-shot transfer through our EXISTING frozen MAE
encoders (no retraining) -- a secondary check alongside the classical alignment baseline.

    python -m paper.eval_zero_shot_aligned
"""
import numpy as np
import torch
import torch.nn.functional as F

from paper import config
from paper.align import align_per_subject
from paper.data import load_bnci_all
from paper.eval_zero_shot import load_frozen_encoder, extract_cls_features, train_linear_head
from paper.utils import classification_metrics


def run(variant: str, subjects=None, device: str = None):
    config.set_seed(config.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    subjects = subjects or config.EVAL_SUBJECTS

    per_subject, label_encoder = load_bnci_all(subjects)
    n_channels, n_times = next(iter(per_subject.values()))[0].shape[1:]
    n_classes = len(label_encoder.classes_)
    encoder = load_frozen_encoder(variant, n_channels, n_times, device)

    results = {}
    for method in ("none", "ea", "ra"):
        X_only = {s: X for s, (X, y) in per_subject.items()}
        aligned_X = align_per_subject(X_only, method)
        aligned = {s: (aligned_X[s].astype(np.float32), per_subject[s][1]) for s in per_subject}

        fold_results = []
        for held_out in subjects:
            source_subjects = [s for s in subjects if s != held_out]
            source_X = np.concatenate([aligned[s][0] for s in source_subjects], axis=0)
            source_y = np.concatenate([aligned[s][1] for s in source_subjects], axis=0)
            target_X, target_y = aligned[held_out]

            source_feats = extract_cls_features(encoder, source_X, device)
            target_feats = extract_cls_features(encoder, target_X, device)

            head = train_linear_head(source_feats, torch.from_numpy(source_y).long(), n_classes, device)
            head.eval()
            with torch.no_grad():
                pred = head(target_feats.to(device)).argmax(-1).cpu().numpy()
            metrics = classification_metrics(target_y, pred)
            fold_results.append(metrics)

        accs = np.array([r["accuracy"] for r in fold_results])
        kappas = np.array([r["kappa"] for r in fold_results])
        results[method] = {"accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
                            "kappa_mean": kappas.mean(), "kappa_std": kappas.std()}
        print(f"[{variant}/{method}] acc={accs.mean()*100:.2f}% +/- {accs.std()*100:.2f}%  "
              f"kappa={kappas.mean():.3f} +/- {kappas.std():.3f}")

    return results


if __name__ == "__main__":
    for variant in ("mae_only", "mae_grl"):
        run(variant)
