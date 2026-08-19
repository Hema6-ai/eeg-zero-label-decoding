"""Stage 1 (v2): pretrain the trial-attention Riemannian Transformer on PhysioNet.

Each forward pass processes N=32 trials from ONE subject as a set; the transformer attends
across trials (not within a single trial, which v1 had nothing to attend to). EA is applied
per-subject before covariance estimation; TangentSpace is fit PER SUBJECT independently (each
subject's own reference), matching how the zero-shot eval fits TangentSpace per-domain (source
pool) rather than reusing a global reference from a different subject population.

Train/val split is at the SUBJECT level (entire subjects held out for validation), not the trial
level, since each "sample" here is a set of trials from one subject.

    python -m paper.train_riemannian_transformer_v2
"""
import os

import numpy as np
import torch
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from paper import config
from paper.align import align_per_subject
from paper.data import load_physionet_supcon
from paper.models.riemannian_transformer import TrialAttentionModel
from paper.utils import EpochLogger, maybe_save_periodic_and_best


def build_scheduler(optimizer, warmup_epochs, total_epochs):
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs))
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def prepare_per_subject_tangent_vecs(subjects=None):
    """EA per subject -> Covariances(lwf) -> TangentSpace fit PER SUBJECT independently.
    Returns {subject_id: (tangent_vecs (M,253), labels (M,))}."""
    X, y, subject_ids, label_encoder, subject_encoder = load_physionet_supcon(subjects)
    per_subject_X = {s: X[subject_ids == s] for s in np.unique(subject_ids)}
    per_subject_y = {s: y[subject_ids == s] for s in np.unique(subject_ids)}
    aligned = align_per_subject(per_subject_X, "ea")

    out = {}
    for s in aligned:
        covs = Covariances(estimator="lwf").fit_transform(aligned[s].astype(np.float64))
        ts = TangentSpace(metric="riemann")
        vecs = ts.fit_transform(covs).astype(np.float32)
        out[s] = (vecs, per_subject_y[s])
    return out, label_encoder, subject_encoder


def make_chunks(vecs, labels, n_trials, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(vecs))
    n_chunks = len(vecs) // n_trials
    chunks = []
    for i in range(n_chunks):
        sel = idx[i * n_trials:(i + 1) * n_trials]
        chunks.append((vecs[sel], labels[sel]))
    return chunks


def run(subjects=None, epochs: int = None, device: str = None):
    config.set_seed(config.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epochs or config.RTV2_EPOCHS
    n_trials = config.RTV2_N_TRIALS

    per_subject, label_encoder, subject_encoder = prepare_per_subject_tangent_vecs(subjects)
    print(f"Trial-attention pretrain classes: {list(label_encoder.classes_)}, "
          f"n_subjects={len(per_subject)}")

    all_subjects = sorted(per_subject.keys())
    n_val_subjects = max(1, int(0.1 * len(all_subjects)))
    rng = np.random.RandomState(config.SEED)
    val_subjects = set(rng.choice(all_subjects, n_val_subjects, replace=False))
    train_subjects = [s for s in all_subjects if s not in val_subjects]

    n_classes = len(label_encoder.classes_)
    model = TrialAttentionModel(
        tangent_dim=config.RT_TANGENT_DIM, embed_dim=config.RT_EMBED_DIM,
        depth=config.RTV2_DEPTH, heads=config.RT_HEADS, n_classes=n_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.RTV2_LR, weight_decay=config.RTV2_WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, config.RTV2_WARMUP_EPOCHS, epochs)
    logger = EpochLogger(os.path.join(config.RESULTS_DIR, "pretrain_riemannian_transformer_v2.csv"))

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_n_chunks = 0.0, 0
        train_chunks = []
        for s in train_subjects:
            vecs, labels = per_subject[s]
            train_chunks.extend(make_chunks(vecs, labels, n_trials, seed=config.SEED + epoch + s))
        rng.shuffle(train_chunks)

        for vecs, labels in train_chunks:
            x = torch.from_numpy(vecs).unsqueeze(0).to(device)       # (1, N, 253)
            y = torch.from_numpy(labels).long().unsqueeze(0).to(device)  # (1, N)
            optimizer.zero_grad()
            logits = model(x)  # (1, N, n_classes)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, n_classes), y.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_n_chunks += 1
        scheduler.step()

        model.eval()
        val_loss, val_correct, val_total, val_n_chunks = 0.0, 0, 0, 0
        with torch.no_grad():
            for s in val_subjects:
                vecs, labels = per_subject[s]
                for vc, lb in make_chunks(vecs, labels, n_trials, seed=1000 + s):
                    x = torch.from_numpy(vc).unsqueeze(0).to(device)
                    y = torch.from_numpy(lb).long().unsqueeze(0).to(device)
                    logits = model(x)
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, n_classes), y.view(-1))
                    val_loss += loss.item()
                    val_correct += (logits.argmax(-1) == y).sum().item()
                    val_total += y.numel()
                    val_n_chunks += 1

        train_loss /= max(1, train_n_chunks)
        val_loss /= max(1, val_n_chunks)
        val_acc = val_correct / max(1, val_total)

        logger.log(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_acc=val_acc)

        best_val_loss = maybe_save_periodic_and_best(
            model, optimizer, epoch, val_loss, best_val_loss,
            config.CHECKPOINT_DIR, "pretrain_riemannian_transformer_v2", config.RTV2_CHECKPOINT_EVERY,
        )

    return model, subject_encoder


if __name__ == "__main__":
    run()
