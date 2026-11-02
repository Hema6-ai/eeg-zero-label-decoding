"""TMLR revision round 2, task E2(a): LaBraM positive-control diagnosis. The manuscript reports
LaBraM at exactly chance (25.08% +/- 1.74%, 4-class) under this paper's zero-shot LOSO protocol
(paper/eval_foundation_model_labram.py). Before reporting that as a finding about the model, this
script checks for a pipeline bug via 4 controls, per the task:

  (i)   In-distribution same-subject linear probe: train AND test on the SAME subject's own
        frozen features (BCI-IV 2a's native train/eval session split, no cross-subject transfer
        at all). If this is also at chance, the bug is upstream of cross-subject transfer.
  (ii)  Input amplitude/scale convention check. FINDING (verified against the live LaBraM repo,
        github.com/935963004/LaBraM, README.md + dataset_maker/make_h5dataset_for_pretrain.py,
        this session): LaBraM's documented preprocessing sets the input unit to raw microvolts
        (no z-scoring, no explicit extra multiplier found in the repo's own data-prep script)
        after bandpass/notch/resample. paper/eval_foundation_model_labram.py's
        `load_bnci_200hz` instead z-scores each trial to zero-mean/unit-variance BEFORE handing
        it to the frozen LaBraM backbone -- a real, mechanistically plausible scale mismatch: a
        pretrained patch-embedding linear layer calibrated to real EEG microvolt magnitudes
        (verified empirically this session: raw MOABB BCI-IV-2a amplitudes are already in a
        microvolt-like range, std~5, range [-31,+36]) receiving unit-variance-normalized input
        instead could plausibly produce uninformative features regardless of probe quality. This
        variant tests z-scored vs. raw (MOABB-native-scale, mean-centered but NOT variance-
        normalized) input side by side.
  (iii) Intermediate-layer probing: extracts the CLS token after each of the model's 12 transformer
        blocks (not just the final layer) to check whether informative features exist at some
        depth even if the final-layer representation is not useful for this task.
  (iv)  EA-aligned input: per-subject Euclidean Alignment (this paper's standard unsupervised
        preprocessing, paper/align.py) applied before the chosen best scale convention.

If the in-distribution probe (i) is still at chance after controlling for (ii)-(iv), that is
reported honestly as "could not establish a working positive control," not papered over.

    python -m paper.diagnose_labram_positive_control
"""
import csv
import os

import numpy as np
import torch

import moabb.datasets as _moabb_datasets
if not hasattr(_moabb_datasets, "BNCI2014001"):
    # See paper/run_dl_loso_sweep.py for why this shim is needed (braindecode 1.2.0 vs moabb 1.5.0
    # import-time incompatibility, pure compatibility alias, changes nothing else).
    _moabb_datasets.BNCI2014001 = _moabb_datasets.BNCI2014_001
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from paper import config
from paper.align import euclidean_alignment
from paper.eval_foundation_model_labram import (
    LABRAM_CHANNELS, LABRAM_SFREQ, LABRAM_PATCH_SIZE, LABRAM_TMIN, LABRAM_TMAX,
    get_input_chans, load_frozen_labram,
)
from paper.utils import classification_metrics

RESULTS_DIR = config.RESULTS_DIR


def load_subject_sessions_labram(subject, zscore: bool):
    """BCI-IV 2a at LaBraM's native 200Hz/20-channel convention, split by its native train/eval
    session labels (same split paper/train_eegnet_baseline.py::load_subject_sessions uses, just
    at this experiment's sampling rate/channel subset). zscore=False keeps MOABB's native
    amplitude scale (mean-centered per trial, but NOT divided by its own std) -- see module
    docstring finding (ii)."""
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    paradigm = MotorImagery(channels=LABRAM_CHANNELS, resample=LABRAM_SFREQ,
                             tmin=LABRAM_TMIN, tmax=LABRAM_TMAX)
    X, y, metadata = paradigm.get_data(dataset=BNCI2014_001(), subjects=[subject])
    n_times_exact = (X.shape[-1] // LABRAM_PATCH_SIZE) * LABRAM_PATCH_SIZE
    X = X[..., :n_times_exact].astype(np.float32)

    mean = X.mean(axis=-1, keepdims=True)
    if zscore:
        std = X.std(axis=-1, keepdims=True)
        X = (X - mean) / (std + 1e-6)
    else:
        X = X - mean  # mean-center only, keep MOABB's native (microvolt-like) amplitude scale

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    is_train = metadata["session"].str.contains("train", case=False).values
    return X[is_train], y[is_train], X[~is_train], y[~is_train], label_encoder


def extract_cls_at_depth(model, X, device, depth, batch_size=64):
    """Replicates braindecode.models.Labram.forward_features up through exactly `depth` of the
    12 transformer blocks (depth=None = all blocks, matching the original script's convention),
    returning the CLS token (not the mean-pooled patch tokens the original script's fc_norm path
    uses) -- CLS token is well-defined at every depth, unlike fc_norm's patch-token pooling."""
    n_blocks = len(model.blocks) if depth is None else depth
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).to(device)
            x = model.patch_embed(batch)
            cls_tokens = model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            pos_embed = model._adj_position_embedding(pos_embed_used=model.position_embedding,
                                                        batch_size=x.shape[0])
            x = x + pos_embed
            time_embed = model._adj_temporal_embedding(num_ch=model.n_chans, batch_size=x.shape[0],
                                                         dim_embed=x.shape[-1])
            x[:, 1:, :] = x[:, 1:, :] + time_embed
            x = model.pos_drop(x)
            for blk in model.blocks[:n_blocks]:
                x = blk(x)
            x = model.norm(x)
            feats.append(x[:, 0].cpu())  # CLS token
    return torch.cat(feats, dim=0).numpy()


def run_variant(name, subjects, device, zscore, align, depth):
    rows = []
    encoder = None
    for subject in subjects:
        Xtr, ytr, Xte, yte, _ = load_subject_sessions_labram(subject, zscore=zscore)
        if align:
            Xtr = euclidean_alignment(Xtr.astype(np.float64)).astype(np.float32)
            Xte = euclidean_alignment(Xte.astype(np.float64)).astype(np.float32)

        n_chans, n_times = Xtr.shape[1], Xtr.shape[2]
        if encoder is None:
            encoder = load_frozen_labram(n_chans, n_times, device)

        train_feats = extract_cls_at_depth(encoder, Xtr, device, depth)
        test_feats = extract_cls_at_depth(encoder, Xte, device, depth)

        clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
        clf.fit(train_feats, ytr)
        pred = clf.predict(test_feats)
        metrics = classification_metrics(yte, pred)
        metrics["subject"] = subject
        rows.append(metrics)
        print(f"[{name}] subject={subject} acc={metrics['accuracy']*100:.2f}% kappa={metrics['kappa']:.3f}",
              flush=True)

    accs = np.array([r["accuracy"] for r in rows])
    kappas = np.array([r["kappa"] for r in rows])
    print(f"=== [{name}] in-distribution probe: {accs.mean()*100:.2f}% +/- {accs.std()*100:.2f}% "
          f"kappa={kappas.mean():.3f} (chance=25%) ===\n", flush=True)
    return rows, {"variant": name, "accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
                  "kappa_mean": kappas.mean(), "kappa_std": kappas.std(), "n_subjects": len(rows)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subjects = config.EVAL_SUBJECTS
    os.makedirs(RESULTS_DIR, exist_ok=True)

    variants = [
        ("zscore_final", dict(zscore=True, align=False, depth=None)),
        ("raw_scale_final", dict(zscore=False, align=False, depth=None)),
        ("raw_scale_ea_final", dict(zscore=False, align=True, depth=None)),
        ("zscore_ea_final", dict(zscore=True, align=True, depth=None)),
    ]
    summaries = []
    best_so_far = ("zscore_final", dict(zscore=True, align=False, depth=None))
    for name, kwargs in variants:
        _, summary = run_variant(name, subjects, device, **kwargs)
        summaries.append(summary)
        if summary["accuracy_mean"] > [s for s in summaries if s["variant"] == best_so_far[0]][0]["accuracy_mean"]:
            best_so_far = (name, kwargs)

    # Intermediate-layer probe at the best-performing scale/alignment convention found above.
    best_name, best_kwargs = best_so_far
    n_blocks_total = None
    for depth in [2, 4, 6, 8, 10, None]:
        kwargs = dict(best_kwargs)
        kwargs["depth"] = depth
        depth_label = depth if depth is not None else "all"
        _, summary = run_variant(f"{best_name}_depth{depth_label}", subjects, device, **kwargs)
        summaries.append(summary)

    out_path = os.path.join(RESULTS_DIR, "e2_labram_positive_control.csv")
    with open(out_path, "w", newline="") as f:
        fieldnames = list(summaries[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nWrote {len(summaries)} variant summaries to {out_path}")

    best_overall = max(summaries, key=lambda s: s["accuracy_mean"])
    print(f"\nBest variant overall: {best_overall['variant']} -> "
          f"{best_overall['accuracy_mean']*100:.2f}% (chance=25%)")
    if best_overall["accuracy_mean"] < 0.30:
        print("WARNING: even the best in-distribution, same-subject control remains near chance "
              "(<30%). This should be reported as 'could not establish a working positive "
              "control for this checkpoint/protocol' rather than a confirmed cross-subject "
              "transfer failure specifically.")
    return summaries


if __name__ == "__main__":
    main()
