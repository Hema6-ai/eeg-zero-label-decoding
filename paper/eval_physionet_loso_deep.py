"""Zero-shot LOSO evaluation of deep-learning (Family 1/2) methods WITHIN PhysioNet's full subject
population, addressing TMLR review Critical Issue B (comprehensive method-family comparison was
not reproduced on the larger dataset) for the self-supervised / geometric-deep-learning families.

Protocol note (see paper/data.py::load_bnci_pretrain docstring and reports/data_leakage_audit.md
for the full discussion): the manuscript's existing checkpoints were pretrained on ALL 109
PhysioNet subjects, so they cannot be validly zero-shot evaluated WITHIN PhysioNet (each held-out
subject's own unlabeled data would already be in its own encoder's pretraining corpus -- the
original project explicitly identified and avoided exactly this, see
paper/results/final_results_table.csv row 27). This script therefore evaluates encoders pretrained
in the REVERSED direction instead: on BCI-IV 2a's 9 subjects (source="bnci", checkpoints suffixed
"_srcbnci"), then zero-shot LOSO across PhysioNet's full population -- a clean, leakage-free
population swap using the exact same frozen-encoder + linear/MLP-probe protocol as every other
zero-shot eval in this project. The PhysioNet-pretrained checkpoints (used for the manuscript's
BCI-IV 2a headline numbers) are untouched by this script.

Run `paper.run_physionet_deep_sweep` first to produce the "_srcbnci" checkpoints this script reads.

    python -m paper.eval_physionet_loso_deep --variant mae_only
"""
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from torch.utils.data import DataLoader, TensorDataset

from paper import config
from paper.data import load_physionet_zeroshot
from paper.models.encoder import EEGTransformerEncoder
from paper.models.heads import LinearProbe, MLPProbe
from paper.models.riemannian_transformer import TrialAttentionTransformer
from paper.models.spdnet import SPDNetEncoder
from paper.utils import classification_metrics

RESULTS_DIR = config.RESULTS_DIR


def _fold_csv_path(name):
    return os.path.join(RESULTS_DIR, f"physionet_loso_{name}.csv")


def _load_done_folds(name):
    """Resume support: returns {held_out_subject: {accuracy, kappa}} already written for this
    variant, so an interrupted Kaggle session only loses the in-progress fold, not the whole run
    (same convention as kaggle_kernel/physionet_full_pipeline_push/run_kernel.py)."""
    path = _fold_csv_path(name)
    done = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                done[int(row["held_out_subject"])] = {
                    "accuracy": float(row["accuracy"]), "kappa": float(row["kappa"])}
    return done


def _append_fold_csv(name, held_out, accuracy, kappa):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = _fold_csv_path(name)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "accuracy", "kappa"])
        if write_header:
            writer.writeheader()
        writer.writerow({"held_out_subject": held_out, "accuracy": accuracy, "kappa": kappa})


def _summarize(name, fold_results):
    accs = np.array([r["accuracy"] for r in fold_results])
    kappas = np.array([r["kappa"] for r in fold_results])
    summary = {"variant": name, "n_subjects": len(fold_results),
               "accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
               "kappa_mean": kappas.mean(), "kappa_std": kappas.std()}
    print(f"[{name}] n={summary['n_subjects']} LOSO accuracy: {summary['accuracy_mean']:.4f} +/- "
          f"{summary['accuracy_std']:.4f}  kappa: {summary['kappa_mean']:.4f} +/- {summary['kappa_std']:.4f}")
    return summary


# --- MAE-only / MAE+GRL: raw-EEG transformer encoder, CLS token, linear probe, no EA ---

def run_mae(variant: str, subjects=None, device: str = None):
    assert variant in ("mae_only", "mae_grl")
    config.set_seed(config.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    per_subject, label_encoder, good_subjects = load_physionet_zeroshot(subjects, align="none")
    n_classes = len(label_encoder.classes_)
    n_channels, n_times = next(iter(per_subject.values()))[0].shape[1:]

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"pretrain_{variant}_srcbnci_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder = EEGTransformerEncoder(
        n_channels=n_channels, n_times=n_times, patch_size=config.PATCH_SIZE,
        embed_dim=config.EMBED_DIM, depth=config.ENCODER_DEPTH, heads=config.ENCODER_HEADS,
    ).to(device)
    encoder_state = {k[len("encoder."):]: v for k, v in ckpt["model"].items() if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    def extract(X, batch_size=256):
        feats = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.from_numpy(X[i:i + batch_size]).to(device)
                feats.append(encoder.cls_representation(batch).cpu())
        return torch.cat(feats, dim=0)

    def train_head(feats, labels):
        head = LinearProbe(config.EMBED_DIM, n_classes).to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=config.PROBE_LR)
        loader = DataLoader(TensorDataset(feats, labels), batch_size=64, shuffle=True)
        head.train()
        for _ in range(config.PROBE_EPOCHS):
            for feat, label in loader:
                feat, label = feat.to(device), label.to(device)
                optimizer.zero_grad()
                loss = F.cross_entropy(head(feat), label)
                loss.backward()
                optimizer.step()
        return head

    done = _load_done_folds(variant)
    fold_results = [{"held_out_subject": s, **r} for s, r in done.items()]
    for held_out in good_subjects:
        if held_out in done:
            continue
        source_subjects = [s for s in good_subjects if s != held_out]
        source_X = np.concatenate([per_subject[s][0] for s in source_subjects], axis=0)
        source_y = np.concatenate([per_subject[s][1] for s in source_subjects], axis=0)
        target_X, target_y = per_subject[held_out]

        source_feats = extract(source_X)
        target_feats = extract(target_X)
        head = train_head(source_feats, torch.from_numpy(source_y).long())
        head.eval()
        with torch.no_grad():
            pred = head(target_feats.to(device)).argmax(-1).cpu().numpy()

        metrics = classification_metrics(target_y, pred)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        _append_fold_csv(variant, held_out, metrics["accuracy"], metrics["kappa"])
        print(f"[{variant}] held-out subject {held_out}: acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

    return _summarize(variant, fold_results), fold_results


# --- SupCon: same encoder architecture as MAE, CLS token, linear... actually MLP probe (per
# original eval_zero_shot_supcon.py convention) ---

def run_supcon(subjects=None, device: str = None):
    config.set_seed(config.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    per_subject, label_encoder, good_subjects = load_physionet_zeroshot(subjects, align="none")
    n_classes = len(label_encoder.classes_)
    n_channels, n_times = next(iter(per_subject.values()))[0].shape[1:]

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "pretrain_supcon_srcbnci_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder = EEGTransformerEncoder(
        n_channels=n_channels, n_times=n_times, patch_size=config.PATCH_SIZE,
        embed_dim=config.EMBED_DIM, depth=config.ENCODER_DEPTH, heads=config.ENCODER_HEADS,
    ).to(device)
    encoder_state = {k[len("encoder."):]: v for k, v in ckpt["model"].items() if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    def extract(X, batch_size=256):
        feats = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.from_numpy(X[i:i + batch_size]).to(device)
                feats.append(encoder.cls_representation(batch).cpu())
        return torch.cat(feats, dim=0)

    def train_head(feats, labels):
        head = MLPProbe(config.EMBED_DIM, n_classes, config.SUPCON_MLP_HIDDEN, config.SUPCON_MLP_DROPOUT).to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=config.PROBE_LR)
        loader = DataLoader(TensorDataset(feats, labels), batch_size=64, shuffle=True)
        head.train()
        for _ in range(config.SUPCON_HEAD_EPOCHS):
            for feat, label in loader:
                feat, label = feat.to(device), label.to(device)
                optimizer.zero_grad()
                loss = F.cross_entropy(head(feat), label)
                loss.backward()
                optimizer.step()
        return head

    done = _load_done_folds("supcon")
    fold_results = [{"held_out_subject": s, **r} for s, r in done.items()]
    for held_out in good_subjects:
        if held_out in done:
            continue
        source_subjects = [s for s in good_subjects if s != held_out]
        source_X = np.concatenate([per_subject[s][0] for s in source_subjects], axis=0)
        source_y = np.concatenate([per_subject[s][1] for s in source_subjects], axis=0)
        target_X, target_y = per_subject[held_out]

        source_feats = extract(source_X)
        target_feats = extract(target_X)
        head = train_head(source_feats, torch.from_numpy(source_y).long())
        head.eval()
        with torch.no_grad():
            pred = head(target_feats.to(device)).argmax(-1).cpu().numpy()

        metrics = classification_metrics(target_y, pred)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        _append_fold_csv("supcon", held_out, metrics["accuracy"], metrics["kappa"])
        print(f"[supcon] held-out subject {held_out}: acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

    return _summarize("supcon", fold_results), fold_results


# --- SPDNet: EA + Covariances(lwf) + frozen BiMap/ReEig/LogEig encoder + MLP probe ---

def run_spdnet(subjects=None, device: str = None):
    config.set_seed(config.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    per_subject, label_encoder, good_subjects = load_physionet_zeroshot(subjects, align="ea")
    n_classes = len(label_encoder.classes_)

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "pretrain_spdnet_srcbnci_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder = SPDNetEncoder(dims=config.SPDNET_DIMS).to(device)
    encoder_state = {k[len("encoder."):]: v for k, v in ckpt["model"].items() if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    cov = Covariances(estimator="lwf")

    def embed(covs, batch_size=128):
        embeds = []
        with torch.no_grad():
            for i in range(0, len(covs), batch_size):
                batch = torch.from_numpy(covs[i:i + batch_size]).to(device)
                embeds.append(encoder(batch).cpu())
        return torch.cat(embeds, dim=0)

    def train_head(feats, labels):
        head = MLPProbe(feats.shape[1], n_classes, config.SPDNET_HEAD_HIDDEN, config.SPDNET_HEAD_DROPOUT).to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=config.PROBE_LR)
        loader = DataLoader(TensorDataset(feats, labels), batch_size=64, shuffle=True)
        head.train()
        for _ in range(config.SPDNET_HEAD_EPOCHS):
            for feat, label in loader:
                feat, label = feat.to(device), label.to(device)
                optimizer.zero_grad()
                loss = F.cross_entropy(head(feat), label)
                loss.backward()
                optimizer.step()
        return head

    done = _load_done_folds("spdnet")
    fold_results = [{"held_out_subject": s, **r} for s, r in done.items()]
    for held_out in good_subjects:
        if held_out in done:
            continue
        source_subjects = [s for s in good_subjects if s != held_out]
        source_X = np.concatenate([per_subject[s][0] for s in source_subjects], axis=0).astype(np.float64)
        source_y = np.concatenate([per_subject[s][1] for s in source_subjects], axis=0)
        target_X, target_y = per_subject[held_out]
        target_X = target_X.astype(np.float64)

        source_covs = cov.fit_transform(source_X).astype(np.float32)
        target_covs = cov.transform(target_X).astype(np.float32)
        source_feats = embed(source_covs)
        target_feats = embed(target_covs)

        head = train_head(source_feats, torch.from_numpy(source_y).long())
        head.eval()
        with torch.no_grad():
            pred = head(target_feats.to(device)).argmax(-1).cpu().numpy()

        metrics = classification_metrics(target_y, pred)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        _append_fold_csv("spdnet", held_out, metrics["accuracy"], metrics["kappa"])
        print(f"[spdnet] held-out subject {held_out}: acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

    return _summarize("spdnet", fold_results), fold_results


# --- Riemannian Transformer v2: EA + Covariances(lwf) + TangentSpace(source-fit) + frozen
# trial-attention transformer (N=32 trial chunks) + MLP probe ---

def run_riemannian_transformer_v2(subjects=None, device: str = None):
    config.set_seed(config.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_trials = config.RTV2_N_TRIALS

    per_subject, label_encoder, good_subjects = load_physionet_zeroshot(subjects, align="ea")
    n_classes = len(label_encoder.classes_)

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "pretrain_riemannian_transformer_v2_srcbnci_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder = TrialAttentionTransformer(
        tangent_dim=config.RT_TANGENT_DIM, embed_dim=config.RT_EMBED_DIM,
        depth=config.RTV2_DEPTH, heads=config.RT_HEADS,
    ).to(device)
    encoder_state = {k[len("encoder."):]: v for k, v in ckpt["model"].items() if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    def embed_all_trials(vecs):
        m = len(vecs)
        n_chunks = int(np.ceil(m / n_trials))
        padded_len = n_chunks * n_trials
        idx = np.arange(padded_len) % m
        all_embeds = np.empty((padded_len, encoder.embed_dim), dtype=np.float32)
        with torch.no_grad():
            for c in range(n_chunks):
                sel = idx[c * n_trials:(c + 1) * n_trials]
                x = torch.from_numpy(vecs[sel]).unsqueeze(0).to(device)
                _, trial_out = encoder(x)
                all_embeds[c * n_trials:(c + 1) * n_trials] = trial_out.squeeze(0).cpu().numpy()
        result = np.empty((m, encoder.embed_dim), dtype=np.float32)
        seen = np.zeros(m, dtype=bool)
        for pos, orig in enumerate(idx):
            if not seen[orig]:
                result[orig] = all_embeds[pos]
                seen[orig] = True
        return result

    def train_head(feats, labels):
        head = MLPProbe(config.RT_EMBED_DIM, n_classes, config.RT_HEAD_HIDDEN, config.RT_HEAD_DROPOUT).to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=config.PROBE_LR)
        loader = DataLoader(TensorDataset(feats, labels), batch_size=64, shuffle=True)
        head.train()
        for _ in range(config.RT_HEAD_EPOCHS):
            for feat, label in loader:
                feat, label = feat.to(device), label.to(device)
                optimizer.zero_grad()
                loss = F.cross_entropy(head(feat), label)
                loss.backward()
                optimizer.step()
        return head

    done = _load_done_folds("riemannian_transformer_v2")
    fold_results = [{"held_out_subject": s, **r} for s, r in done.items()]
    for held_out in good_subjects:
        if held_out in done:
            continue
        source_subjects = [s for s in good_subjects if s != held_out]
        source_X = np.concatenate([per_subject[s][0] for s in source_subjects], axis=0).astype(np.float64)
        source_y = np.concatenate([per_subject[s][1] for s in source_subjects], axis=0)
        target_X, target_y = per_subject[held_out]
        target_X = target_X.astype(np.float64)

        cov = Covariances(estimator="lwf")
        source_covs = cov.fit_transform(source_X)
        target_covs = cov.transform(target_X)
        ts = TangentSpace(metric="riemann")
        source_vecs = ts.fit_transform(source_covs).astype(np.float32)
        target_vecs = ts.transform(target_covs).astype(np.float32)

        source_feats = torch.from_numpy(embed_all_trials(source_vecs))
        target_feats = torch.from_numpy(embed_all_trials(target_vecs))
        head = train_head(source_feats, torch.from_numpy(source_y).long())
        head.eval()
        with torch.no_grad():
            pred = head(target_feats.to(device)).argmax(-1).cpu().numpy()

        metrics = classification_metrics(target_y, pred)
        metrics["held_out_subject"] = held_out
        fold_results.append(metrics)
        _append_fold_csv("riemannian_transformer_v2", held_out, metrics["accuracy"], metrics["kappa"])
        print(f"[riemannian_transformer_v2] held-out subject {held_out}: "
              f"acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

    return _summarize("riemannian_transformer_v2", fold_results), fold_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                         choices=["mae_only", "mae_grl", "supcon", "spdnet", "riemannian_transformer_v2"])
    args = parser.parse_args()
    if args.variant in ("mae_only", "mae_grl"):
        run_mae(args.variant)
    elif args.variant == "supcon":
        run_supcon()
    elif args.variant == "spdnet":
        run_spdnet()
    else:
        run_riemannian_transformer_v2()
