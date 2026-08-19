"""Diagnostic 1: does the frozen MAE representation carry class-discriminative information at
all, with no cross-subject transfer involved? For each BCI-IV 2a subject, train a linear
classifier (LogisticRegression) on that subject's own frozen CLS features (training session) and
evaluate on that same subject's held-out trials (test session) -- no other subject's data is
used anywhere, isolating representation quality from the zero-shot transfer mechanism.

    python -m paper.diagnose_in_distribution_probe
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from paper import config
from paper.eval_zero_shot import extract_cls_features, load_frozen_encoder
from paper.eval_zero_shot_club_scin import load_frozen_model
from paper.train_eegnet_baseline import load_subject_sessions
from paper.utils import classification_metrics


def run_variant(variant: str, subjects=None, device: str = "cpu"):
    subjects = subjects or config.EVAL_SUBJECTS

    fold_results = []
    encoder = None
    club_model = None
    for subject in subjects:
        X_train, y_train, X_test, y_test, label_encoder = load_subject_sessions(subject)
        n_channels, n_times = X_train.shape[1], X_train.shape[2]

        if variant == "mae_club_scin":
            if club_model is None:
                club_model = load_frozen_model(n_channels, n_times, len(config.PRETRAIN_SUBJECTS), device)
            train_feats = club_model.encode_zero_shot(torch.from_numpy(X_train).to(device)).detach().numpy()
            test_feats = club_model.encode_zero_shot(torch.from_numpy(X_test).to(device)).detach().numpy()
        else:
            if encoder is None:
                encoder = load_frozen_encoder(variant, n_channels, n_times, device)
            train_feats = extract_cls_features(encoder, X_train, device).numpy()
            test_feats = extract_cls_features(encoder, X_test, device).numpy()

        clf = LogisticRegression(max_iter=1000)
        clf.fit(train_feats, y_train)
        pred = clf.predict(test_feats)

        metrics = classification_metrics(y_test, pred)
        metrics["subject"] = subject
        fold_results.append(metrics)
        print(f"[{variant}] subject {subject} (in-distribution): "
              f"acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

    accs = np.array([r["accuracy"] for r in fold_results])
    kappas = np.array([r["kappa"] for r in fold_results])
    print(f"[{variant}] in-distribution accuracy: {accs.mean():.4f} +/- {accs.std():.4f}  "
          f"kappa: {kappas.mean():.4f} +/- {kappas.std():.4f}\n")
    return {"accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
            "kappa_mean": kappas.mean(), "kappa_std": kappas.std()}, fold_results


if __name__ == "__main__":
    for variant in ("mae_only", "mae_grl", "mae_club_scin"):
        run_variant(variant)
