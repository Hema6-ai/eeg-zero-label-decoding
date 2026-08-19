"""Diagnostic visualizations and feature statistics comparing frozen MAE representations against
CSP, using only existing checkpoints -- no new training. Run:

    python -m paper.representational_analysis
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from mne.decoding import CSP
from sklearn.manifold import TSNE

from paper import config
from paper.data import load_bnci_all
from paper.eval_zero_shot import extract_cls_features, load_frozen_encoder
from paper.eval_zero_shot_club_scin import load_frozen_model
from paper.train_eegnet_baseline import load_subject_sessions

FIGURES_DIR = "paper/writeup/figures"
RESULTS_DIR = "paper/results"
CLASS_CMAP = "tab10"
SUBJECT_CMAP = "tab20"


def get_mae_features(variant, subjects, device="cpu"):
    per_subject, label_encoder = load_bnci_all(subjects)
    n_channels, n_times = next(iter(per_subject.values()))[0].shape[1:]

    if variant == "mae_club_scin":
        model = load_frozen_model(n_channels, n_times, len(config.PRETRAIN_SUBJECTS), device)
        extract = lambda X: model.encode_zero_shot(torch.from_numpy(X).to(device)).detach().numpy()
    else:
        encoder = load_frozen_encoder(variant, n_channels, n_times, device)
        extract = lambda X: extract_cls_features(encoder, X, device).numpy()

    all_feats, all_labels, all_subj = [], [], []
    for s in subjects:
        X, y = per_subject[s]
        all_feats.append(extract(X))
        all_labels.append(y)
        all_subj.append(np.full(len(y), s))
    return (np.concatenate(all_feats), np.concatenate(all_labels),
            np.concatenate(all_subj), label_encoder)


def get_csp_features(subjects, n_components=6):
    all_feats, all_labels, all_subj = [], [], []
    label_encoder = None
    for s in subjects:
        X_train, y_train, X_test, y_test, le = load_subject_sessions(s)
        label_encoder = le
        csp = CSP(n_components=n_components, reg="ledoit_wolf", log=True)
        csp.fit(X_train.astype(np.float64), y_train)
        feats = np.concatenate([csp.transform(X_train.astype(np.float64)),
                                 csp.transform(X_test.astype(np.float64))])
        labels = np.concatenate([y_train, y_test])
        all_feats.append(feats)
        all_labels.append(labels)
        all_subj.append(np.full(len(labels), s))
    return (np.concatenate(all_feats), np.concatenate(all_labels),
            np.concatenate(all_subj), label_encoder)


def fisher_and_similarity(feats, labels):
    """Returns (between_class_scatter, within_class_scatter, fisher_ratio, mean_cosine_sim)."""
    classes = np.unique(labels)
    centroids = np.array([feats[labels == c].mean(axis=0) for c in classes])

    between_dists = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            between_dists.append(np.linalg.norm(centroids[i] - centroids[j]))
    between = np.mean(between_dists)

    within_dists = []
    for c in classes:
        members = feats[labels == c]
        if len(members) > 200:  # subsample for speed on large classes
            idx = np.random.RandomState(42).choice(len(members), 200, replace=False)
            members = members[idx]
        n = len(members)
        if n < 2:
            continue
        d = np.linalg.norm(members[:, None, :] - members[None, :, :], axis=-1)
        within_dists.append(d[np.triu_indices(n, k=1)].mean())
    within = np.mean(within_dists)

    fisher = between / (within + 1e-12)

    feats_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    idx = np.random.RandomState(42).choice(len(feats_norm), min(500, len(feats_norm)), replace=False)
    sample = feats_norm[idx]
    sim = sample @ sample.T
    mean_cos_sim = sim[np.triu_indices(len(sample), k=1)].mean()

    return between, within, fisher, mean_cos_sim, centroids


def plot_tsne_comparison(mae_feats, mae_color, csp_feats, csp_color, cmap, labels_names,
                          title_suffix, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, feats, color, name in [(axes[0], mae_feats, mae_color, "MAE-only"),
                                     (axes[1], csp_feats, csp_color, "CSP")]:
        emb = TSNE(n_components=2, perplexity=30, max_iter=1000, init="pca",
                   random_state=config.SEED).fit_transform(feats)
        scatter = ax.scatter(emb[:, 0], emb[:, 1], c=color, cmap=cmap, s=8, alpha=0.7)
        ax.set_title(f"{name}: t-SNE {title_suffix}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    if labels_names is not None:
        handles, _ = scatter.legend_elements()
        axes[1].legend(handles, labels_names, markerscale=1.5, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_centroid_heatmap(mae_centroids, csp_centroids, class_names, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, centroids, name in [(axes[0], mae_centroids, "MAE-only"), (axes[1], csp_centroids, "CSP")]:
        n = len(centroids)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist[i, j] = np.linalg.norm(centroids[i] - centroids[j])
        sns.heatmap(dist, annot=True, fmt=".1f", cmap="viridis", ax=ax,
                    xticklabels=class_names, yticklabels=class_names, cbar=True)
        ax.set_title(f"{name}: class centroid L2 distance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_embedding_variance(variance_dict, out_path):
    fig, axes = plt.subplots(1, len(variance_dict), figsize=(4 * len(variance_dict), 4))
    for ax, (name, var) in zip(axes, variance_dict.items()):
        ax.hist(var, bins=30, color="steelblue", edgecolor="white")
        ax.set_title(f"{name}\n(mean std={np.sqrt(var).mean():.3f})", fontsize=10)
        ax.set_xlabel("per-dimension variance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_singular_values(sv_dict, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, sv in sv_dict.items():
        sv_norm = sv / sv[0]
        ax.plot(np.arange(1, len(sv_norm) + 1), sv_norm, marker="o", markersize=3, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("normalized singular value (log scale)")
    ax.set_title("Feature matrix singular value spectrum")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def run(subjects=None):
    subjects = subjects or config.EVAL_SUBJECTS
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Extracting features...")
    mae_only_feats, mae_only_labels, mae_only_subj, label_encoder = get_mae_features("mae_only", subjects)
    mae_grl_feats, mae_grl_labels, _, _ = get_mae_features("mae_grl", subjects)
    mae_club_feats, mae_club_labels, _, _ = get_mae_features("mae_club_scin", subjects)
    csp_feats, csp_labels, csp_subj, csp_label_encoder = get_csp_features(subjects)

    class_names = list(label_encoder.classes_)

    # --- 1 & 2: t-SNE comparisons ---
    plot_tsne_comparison(mae_only_feats, mae_only_labels, csp_feats, csp_labels, CLASS_CMAP,
                          class_names, "by MI class", os.path.join(FIGURES_DIR, "tsne_class_comparison.png"))
    plot_tsne_comparison(mae_only_feats, mae_only_subj, csp_feats, csp_subj, SUBJECT_CMAP,
                          None, "by subject", os.path.join(FIGURES_DIR, "tsne_subject_comparison.png"))

    # --- 3: Feature statistics ---
    stats_rows = []
    for name, feats, labels in [
        ("MAE-only", mae_only_feats, mae_only_labels),
        ("MAE+GRL", mae_grl_feats, mae_grl_labels),
        ("MAE+CLUB+SCIN+VIB", mae_club_feats, mae_club_labels),
        ("CSP", csp_feats, csp_labels),
    ]:
        between, within, fisher, cos_sim, centroids = fisher_and_similarity(feats, labels)
        stats_rows.append({
            "method": name, "between_class_scatter": between, "within_class_scatter": within,
            "fisher_ratio": fisher, "mean_pairwise_cosine_similarity": cos_sim,
        })
        print(f"[{name}] between={between:.4f} within={within:.4f} fisher={fisher:.4f} "
              f"mean_cos_sim={cos_sim:.4f}")
        if name == "MAE-only":
            mae_only_centroids = centroids
        if name == "CSP":
            csp_centroids = centroids

    with open(os.path.join(RESULTS_DIR, "feature_statistics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stats_rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    methods = [r["method"] for r in stats_rows]
    for ax, key, title in [(axes[0], "fisher_ratio", "Fisher discriminability ratio (higher=better)"),
                            (axes[1], "mean_pairwise_cosine_similarity", "Mean pairwise cosine similarity\n(lower=more diverse)"),
                            (axes[2], "between_class_scatter", "Between-class centroid distance")]:
        values = [r[key] for r in stats_rows]
        colors = ["#4C72B0", "#4C72B0", "#4C72B0", "#DD8452"]
        ax.bar(methods, values, color=colors)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "feature_statistics.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {os.path.join(FIGURES_DIR, 'feature_statistics.png')}")

    # --- 4: Centroid heatmap ---
    plot_centroid_heatmap(mae_only_centroids, csp_centroids, class_names,
                           os.path.join(FIGURES_DIR, "centroid_heatmap.png"))

    # --- Embedding variance ---
    variance_dict = {
        "MAE-only": mae_only_feats.var(axis=0),
        "CSP": csp_feats.var(axis=0),
    }
    plot_embedding_variance(variance_dict, os.path.join(FIGURES_DIR, "embedding_variance.png"))

    # --- Singular values / effective rank ---
    sv_dict = {}
    for name, feats in [("MAE-only", mae_only_feats), ("CSP", csp_feats)]:
        centered = feats - feats.mean(axis=0, keepdims=True)
        _, sv, _ = np.linalg.svd(centered, full_matrices=False)
        sv_dict[name] = sv
    plot_singular_values(sv_dict, os.path.join(FIGURES_DIR, "singular_values.png"))

    print("\n=== Representational analysis complete ===")
    return stats_rows


if __name__ == "__main__":
    run()
