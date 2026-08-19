"""Per-subject fine-tuned EEGNet baseline on BNCI2014_001 (within-subject, uses calibration data).

Trains EEGNet from scratch on each subject's own 'train' session and evaluates on that
subject's own 'test' session -- this is the calibrated baseline the zero-shot EEG-MAE targets
are compared against (reference: ~0.68 accuracy).

    python -m paper.train_eegnet_baseline
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from paper import config
from paper.data import get_paradigm, zscore_per_trial
from paper.models.eegnet import EEGNet
from paper.utils import classification_metrics, EpochLogger
from moabb.datasets import BNCI2014_001
from sklearn.preprocessing import LabelEncoder


def load_subject_sessions(subject: int):
    paradigm = get_paradigm()
    X, y, metadata = paradigm.get_data(dataset=BNCI2014_001(), subjects=[subject])
    X = zscore_per_trial(X.astype(np.float32))

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    is_train = metadata["session"].str.contains("train", case=False)
    return X[is_train.values], y[is_train.values], X[~is_train.values], y[~is_train.values], label_encoder


def run(subjects=None, epochs: int = 100, device: str = None):
    config.set_seed(config.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    subjects = subjects or config.EVAL_SUBJECTS

    logger = EpochLogger(os.path.join(config.RESULTS_DIR, "eegnet_baseline.csv"))
    fold_results = []

    for subject in subjects:
        X_train, y_train, X_test, y_test, label_encoder = load_subject_sessions(subject)
        n_channels, n_times = X_train.shape[1], X_train.shape[2]
        n_classes = len(label_encoder.classes_)

        model = EEGNet(n_channels=n_channels, n_times=n_times, n_classes=n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long()),
            batch_size=32, shuffle=True,
        )
        for _ in range(epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(X_test).to(device)).argmax(-1).cpu().numpy()
        metrics = classification_metrics(y_test, pred)
        metrics["subject"] = subject
        fold_results.append(metrics)
        logger.log(subject=subject, accuracy=metrics["accuracy"], kappa=metrics["kappa"])

    accs = np.array([r["accuracy"] for r in fold_results])
    kappas = np.array([r["kappa"] for r in fold_results])
    print(f"[eegnet_baseline] accuracy: {accs.mean():.4f} +/- {accs.std():.4f}  "
          f"kappa: {kappas.mean():.4f} +/- {kappas.std():.4f}")
    return {"accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
            "kappa_mean": kappas.mean(), "kappa_std": kappas.std()}, fold_results


if __name__ == "__main__":
    run()
