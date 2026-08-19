"""3-way learned stacked ensemble: FgMDM + MEKT-lite + SPDNet. Same leak-free inner-LOSO-over-
source-subjects methodology as eval_stacked_ensemble.py, extended with SPDNet's frozen-encoder
probabilities as a third meta-feature block. SPDNet is individually weaker than FgMDM/MEKT
(44.12% vs 47.45%/48.38%), but a stacked ensemble can still benefit from a weaker method if its
errors are sufficiently uncorrelated with the other two -- this tests that directly rather than
assuming it from the individual accuracy numbers.

    python -m paper.eval_stacked_ensemble_3way
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

from paper import config
from paper.data import load_bnci_all
from paper.eval_stacked_ensemble import _fgmdm_probs, _mekt_probs
from paper.eval_zero_shot_spdnet import embed_covs, load_frozen_encoder
from paper.models.heads import MLPProbe
from paper.utils import classification_metrics
from pyriemann.estimation import Covariances


def _spdnet_probs(source_X, source_y, eval_X, classes, encoder, device):
    cov = Covariances(estimator="lwf")
    source_covs = cov.fit_transform(source_X).astype(np.float32)
    eval_covs = cov.transform(eval_X).astype(np.float32)
    source_feats = embed_covs(encoder, source_covs, device)
    eval_feats = embed_covs(encoder, eval_covs, device)

    head = MLPProbe(source_feats.shape[1], len(classes), config.SPDNET_HEAD_HIDDEN, config.SPDNET_HEAD_DROPOUT).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=config.PROBE_LR)
    loader = DataLoader(TensorDataset(source_feats, torch.from_numpy(source_y).long()), batch_size=64, shuffle=True)
    head.train()
    for _ in range(config.SPDNET_HEAD_EPOCHS):
        for feat, label in loader:
            feat, label = feat.to(device), label.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(head(feat), label)
            loss.backward()
            optimizer.step()
    head.eval()
    with torch.no_grad():
        probs = F.softmax(head(eval_feats.to(device)), dim=-1).cpu().numpy()
    return probs


def run(subjects=None, device: str = None):
    config.set_seed(config.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    subjects = subjects or config.EVAL_SUBJECTS
    per_subject, label_encoder = load_bnci_all(subjects, align="ea")
    classes = np.arange(len(label_encoder.classes_))
    spdnet_encoder = load_frozen_encoder(device)

    fold_results = {"fgmdm": [], "mekt": [], "spdnet": [], "stacked3": []}
    for held_out in subjects:
        source_subjects = [s for s in subjects if s != held_out]

        meta_X, meta_y = [], []
        for inner_held_out in source_subjects:
            inner_source = [s for s in source_subjects if s != inner_held_out]
            inner_source_X = np.concatenate([per_subject[s][0] for s in inner_source], axis=0).astype(np.float64)
            inner_source_y = np.concatenate([per_subject[s][1] for s in inner_source], axis=0)
            inner_eval_X, inner_eval_y = per_subject[inner_held_out]
            inner_eval_X = inner_eval_X.astype(np.float64)

            p_fgmdm = _fgmdm_probs(inner_source_X, inner_source_y, inner_eval_X, classes)
            p_mekt = _mekt_probs(inner_source_X, inner_source_y, inner_eval_X, classes)
            p_spd = _spdnet_probs(inner_source_X, inner_source_y, inner_eval_X, classes, spdnet_encoder, device)
            meta_X.append(np.concatenate([p_fgmdm, p_mekt, p_spd], axis=1))
            meta_y.append(inner_eval_y)

        meta_X = np.concatenate(meta_X, axis=0)
        meta_y = np.concatenate(meta_y, axis=0)
        meta_clf = LogisticRegression(max_iter=1000, C=1.0)
        meta_clf.fit(meta_X, meta_y)

        source_X = np.concatenate([per_subject[s][0] for s in source_subjects], axis=0).astype(np.float64)
        source_y = np.concatenate([per_subject[s][1] for s in source_subjects], axis=0)
        target_X, target_y = per_subject[held_out]
        target_X = target_X.astype(np.float64)

        p_fgmdm_t = _fgmdm_probs(source_X, source_y, target_X, classes)
        p_mekt_t = _mekt_probs(source_X, source_y, target_X, classes)
        p_spd_t = _spdnet_probs(source_X, source_y, target_X, classes, spdnet_encoder, device)
        meta_features = np.concatenate([p_fgmdm_t, p_mekt_t, p_spd_t], axis=1)
        pred_stacked = meta_clf.predict(meta_features)

        for key, pred in [("fgmdm", p_fgmdm_t.argmax(1)), ("mekt", p_mekt_t.argmax(1)),
                          ("spdnet", p_spd_t.argmax(1)), ("stacked3", pred_stacked)]:
            m = classification_metrics(target_y, pred)
            m["held_out_subject"] = held_out
            fold_results[key].append(m)

        print(f"[3way] held-out subject {held_out}: "
              f"fgmdm={fold_results['fgmdm'][-1]['accuracy']:.4f} "
              f"mekt={fold_results['mekt'][-1]['accuracy']:.4f} "
              f"spdnet={fold_results['spdnet'][-1]['accuracy']:.4f} "
              f"STACKED3={fold_results['stacked3'][-1]['accuracy']:.4f}")

    summaries = {}
    for key, results in fold_results.items():
        accs = np.array([r["accuracy"] for r in results])
        kappas = np.array([r["kappa"] for r in results])
        summaries[key] = {"accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
                           "kappa_mean": kappas.mean(), "kappa_std": kappas.std()}
        print(f"[{key}] LOSO accuracy: {summaries[key]['accuracy_mean']:.4f} +/- "
              f"{summaries[key]['accuracy_std']:.4f}  kappa: {summaries[key]['kappa_mean']:.4f} +/- "
              f"{summaries[key]['kappa_std']:.4f}")

    return summaries, fold_results


if __name__ == "__main__":
    run()
