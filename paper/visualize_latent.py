"""Diagnostic 2: t-SNE visualization of the frozen MAE CLS representation on BCI-IV 2a trials,
colored by class and separately by subject. If class clusters are not visually separable even
within a single subject's own trials, that is direct evidence the representation itself -- not
just its cross-subject transfer -- is the bottleneck.

    python -m paper.visualize_latent
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from paper import config
from paper.data import load_bnci_all
from paper.eval_zero_shot import extract_cls_features, load_frozen_encoder


def run(variant: str = "mae_only", subjects=None, device: str = "cpu",
        out_dir: str = "paper/writeup/figures"):
    os.makedirs(out_dir, exist_ok=True)
    subjects = subjects or config.EVAL_SUBJECTS

    per_subject, label_encoder = load_bnci_all(subjects)
    n_channels, n_times = next(iter(per_subject.values()))[0].shape[1:]
    encoder = load_frozen_encoder(variant, n_channels, n_times, device)

    all_feats, all_labels, all_subjects = [], [], []
    for subject in subjects:
        X, y = per_subject[subject]
        feats = extract_cls_features(encoder, X, device).numpy()
        all_feats.append(feats)
        all_labels.append(y)
        all_subjects.append(np.full(len(y), subject))

    feats = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    subj_ids = np.concatenate(all_subjects, axis=0)

    embedding = TSNE(n_components=2, random_state=config.SEED, perplexity=30, init="pca").fit_transform(feats)

    class_names = label_encoder.classes_
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(class_names):
        mask = labels == i
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=8, alpha=0.6, label=name)
    ax.set_title(f"{variant}: t-SNE by MI class", fontsize=11, wrap=True)
    ax.legend(markerscale=2, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    class_path = os.path.join(out_dir, f"tsne_{variant}_by_class.png")
    fig.savefig(class_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for subject in subjects:
        mask = subj_ids == subject
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=8, alpha=0.6, label=f"S{subject}")
    ax.set_title(f"{variant}: t-SNE by subject", fontsize=11, wrap=True)
    ax.legend(markerscale=2, fontsize=7, ncol=2)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    subj_path = os.path.join(out_dir, f"tsne_{variant}_by_subject.png")
    fig.savefig(subj_path, dpi=150)
    plt.close(fig)

    print(f"Saved {class_path}")
    print(f"Saved {subj_path}")
    return class_path, subj_path


if __name__ == "__main__":
    run("mae_only")
