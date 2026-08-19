"""Stage 1: supervised pretraining of the Riemannian Transformer on PhysioNet.

Pipeline per trial: raw EEG -> EA (per-subject) -> Covariances(lwf) -> TangentSpace(riemann,
fit on all pretraining trials) -> (253,) vector -> TangentSpaceTransformer -> CrossEntropy on
PhysioNet's 3 shared classes (feet/left_hand/right_hand; no 'tongue' data exists in PhysioNet).

    python -m paper.train_riemannian_transformer
"""
import os

import numpy as np
import torch
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from torch.utils.data import DataLoader, TensorDataset, random_split

from paper import config
from paper.align import align_per_subject
from paper.data import load_physionet_supcon
from paper.models.riemannian_transformer import RiemannianTransformerModel
from paper.utils import EpochLogger, maybe_save_periodic_and_best


def run(subjects=None, epochs: int = None, device: str = None):
    config.set_seed(config.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epochs or config.RT_EPOCHS

    X, y, subject_ids, label_encoder, subject_encoder = load_physionet_supcon(subjects)
    print(f"Riemannian Transformer pretrain classes: {list(label_encoder.classes_)}, "
          f"n_trials={len(X)}, n_subjects={len(subject_encoder.classes_)}")

    per_subject = {s: X[subject_ids == s] for s in np.unique(subject_ids)}
    aligned = align_per_subject(per_subject, "ea")
    X_aligned = np.empty_like(X)
    for s, aX in aligned.items():
        X_aligned[subject_ids == s] = aX

    covs = Covariances(estimator="lwf").fit_transform(X_aligned.astype(np.float64))
    ts = TangentSpace(metric="riemann")
    tangent_vecs = ts.fit_transform(covs).astype(np.float32)
    print(f"Tangent vectors shape: {tangent_vecs.shape}")

    dataset = TensorDataset(torch.from_numpy(tangent_vecs), torch.from_numpy(y).long())
    n_val = max(1, int(0.1 * len(dataset)))
    train_set, val_set = random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(config.SEED),
    )
    train_loader = DataLoader(train_set, batch_size=config.RT_BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config.RT_BATCH_SIZE, shuffle=False)

    n_classes = len(label_encoder.classes_)
    model = RiemannianTransformerModel(
        tangent_dim=config.RT_TANGENT_DIM, embed_dim=config.RT_EMBED_DIM,
        depth=config.RT_DEPTH, heads=config.RT_HEADS, n_classes=n_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.RT_LR, weight_decay=config.RT_WEIGHT_DECAY)
    logger = EpochLogger(os.path.join(config.RESULTS_DIR, "pretrain_riemannian_transformer.csv"))

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_n = 0.0, 0
        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_n += x.size(0)

        model.eval()
        val_loss, val_correct, val_n = 0.0, 0, 0
        with torch.no_grad():
            for x, labels in val_loader:
                x, labels = x.to(device), labels.to(device)
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                val_loss += loss.item() * x.size(0)
                val_correct += (logits.argmax(-1) == labels).sum().item()
                val_n += x.size(0)
        val_loss /= val_n
        val_acc = val_correct / val_n

        logger.log(epoch=epoch, train_loss=train_loss / train_n, val_loss=val_loss, val_acc=val_acc)

        best_val_loss = maybe_save_periodic_and_best(
            model, optimizer, epoch, val_loss, best_val_loss,
            config.CHECKPOINT_DIR, "pretrain_riemannian_transformer", config.RT_CHECKPOINT_EVERY,
        )

    return model, subject_encoder


if __name__ == "__main__":
    run()
