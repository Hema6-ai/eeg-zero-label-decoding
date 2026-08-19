"""MOABB paradigm-based data loading. No custom dataset parsing -- MOABB/MNE handle everything."""
import numpy as np
from moabb.datasets import BNCI2014_001, PhysionetMI
from moabb.paradigms import MotorImagery
from sklearn.preprocessing import LabelEncoder

from paper import config
from paper.align import align_per_subject


def _align_by_subject_ids(X: np.ndarray, subject_ids: np.ndarray, align: str) -> np.ndarray:
    """Applies per-subject alignment to a flat (n_trials, ...) array grouped by subject_ids,
    then reassembles it in the original trial order."""
    if align == "none":
        return X
    per_subject = {s: X[subject_ids == s] for s in np.unique(subject_ids)}
    aligned = align_per_subject(per_subject, align)
    out = np.empty_like(X)
    for s, aligned_X in aligned.items():
        out[subject_ids == s] = aligned_X
    return out


def get_paradigm() -> MotorImagery:
    return MotorImagery(
        channels=config.COMMON_CHANNELS,
        resample=config.RESAMPLE_HZ,
        tmin=config.TMIN,
        tmax=config.TMAX,
    )


def zscore_per_trial(X: np.ndarray) -> np.ndarray:
    """Per-channel, per-trial z-score: each (trial, channel) time series independently normalized."""
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    return (X - mean) / (std + 1e-6)


def load_physionet_pretrain(subjects=None, align: str = "none"):
    """Returns (X, subject_ids) for MAE(+GRL) pretraining. subject_ids are 0-indexed and contiguous.

    align: "none" | "ea" | "ra" -- applies per-subject, unsupervised alignment (each subject's
    own trial covariances only) after the existing per-trial z-score.
    """
    subjects = subjects or config.PRETRAIN_SUBJECTS
    paradigm = get_paradigm()
    dataset = PhysionetMI()

    X, _, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
    X = zscore_per_trial(X.astype(np.float32))

    subject_encoder = LabelEncoder()
    subject_ids = subject_encoder.fit_transform(metadata["subject"].values)
    X = _align_by_subject_ids(X, subject_ids, align).astype(np.float32)
    return X, subject_ids, subject_encoder


def load_physionet_supcon(subjects=None, classes=("feet", "left_hand", "right_hand")):
    """Returns (X, y, subject_ids) for SupCon pretraining, restricted to the classes PhysioNet
    actually shares with BCI-IV 2a. PhysioNet via MOABB has no 'tongue' class at all (its labels
    are feet/hands/left_hand/rest/right_hand) -- 'hands' (bilateral) and 'rest' are dropped since
    they have no counterpart in the BCI-IV 2a zero-shot label space, and 'tongue' is simply never
    seen during pretraining (no PhysioNet data exists for it).
    """
    subjects = subjects or config.PRETRAIN_SUBJECTS
    paradigm = get_paradigm()
    dataset = PhysionetMI()

    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
    keep = np.isin(y, classes)
    X, y, metadata = X[keep], y[keep], metadata[keep].reset_index(drop=True)

    X = zscore_per_trial(X.astype(np.float32))

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    subject_encoder = LabelEncoder()
    subject_ids = subject_encoder.fit_transform(metadata["subject"].values)
    return X, y_encoded, subject_ids, label_encoder, subject_encoder


def load_bnci_subject(subject: int):
    """Returns (X, y) for one BNCI2014_001 subject, for zero-shot LOSO eval."""
    paradigm = get_paradigm()
    dataset = BNCI2014_001()

    X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
    X = zscore_per_trial(X.astype(np.float32))

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return X, y, label_encoder


def load_physionet_zeroshot(subjects=None, classes=("feet", "left_hand", "right_hand"), align: str = "none"):
    """Returns per-subject (X, y) dict plus a shared label encoder, for LOSO zero-shot evaluation
    WITHIN PhysionetMI itself (as opposed to load_physionet_supcon/_pretrain, which prepare
    PhysionetMI purely as an upstream pretraining source for a different eval dataset). Restricted
    to the 3 classes PhysioNet actually has (no 'tongue'). A few PhysionetMI subjects are known to
    have inconsistent recordings (e.g. wrong sampling rate) and can fail to load via MOABB/MNE;
    those are skipped with a printed warning rather than aborting the whole batch load.

    align: "none" | "ea" | "ra" -- applied per-subject (unsupervised, no label leakage).
    """
    subjects = subjects or config.PRETRAIN_SUBJECTS
    paradigm = get_paradigm()
    dataset = PhysionetMI()

    per_subject = {}
    label_encoder = LabelEncoder()
    label_encoder.fit(list(classes))
    good_subjects, skipped_subjects = [], []

    for subject in subjects:
        try:
            X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
        except Exception as e:
            skipped_subjects.append((subject, str(e)))
            continue
        keep = np.isin(y, classes)
        X, y = X[keep], y[keep]
        if len(X) == 0:
            skipped_subjects.append((subject, "no trials in the 3 shared classes"))
            continue
        X = zscore_per_trial(X.astype(np.float32))
        y_encoded = label_encoder.transform(y)
        per_subject[subject] = (X, y_encoded)
        good_subjects.append(subject)

    if skipped_subjects:
        print(f"load_physionet_zeroshot: skipped {len(skipped_subjects)} subjects: {skipped_subjects}")

    if align != "none":
        X_only = {s: X for s, (X, y) in per_subject.items()}
        aligned_X = align_per_subject(X_only, align)
        per_subject = {s: (aligned_X[s].astype(np.float32), y) for s, (_, y) in per_subject.items()}

    return per_subject, label_encoder, good_subjects


def load_bnci_all(subjects=None, align: str = "none"):
    """Returns per-subject (X, y) dict plus a shared label encoder, for LOSO evaluation.

    align: "none" | "ea" | "ra" -- applied per-subject (unsupervised, no label leakage), so it
    is valid even for the held-out zero-shot subject in a LOSO fold.
    """
    subjects = subjects or config.EVAL_SUBJECTS
    paradigm = get_paradigm()
    dataset = BNCI2014_001()

    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
    X = zscore_per_trial(X.astype(np.float32))

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    per_subject = {}
    for subject in subjects:
        idx = metadata["subject"].values == subject
        per_subject[subject] = (X[idx], y[idx])

    if align != "none":
        X_only = {s: X for s, (X, y) in per_subject.items()}
        aligned_X = align_per_subject(X_only, align)
        per_subject = {s: (aligned_X[s].astype(np.float32), y) for s, (_, y) in per_subject.items()}
    return per_subject, label_encoder
