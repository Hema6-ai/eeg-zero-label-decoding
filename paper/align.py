"""Euclidean Alignment (He & Wu, 2020) and Riemannian Alignment (Zanini et al., 2018): unsupervised
per-subject preprocessing that whitens each subject's trials by the inverse square root of a
reference covariance, to test whether cross-subject distribution shift -- not representation
quality -- is the bottleneck on zero-shot transfer. Both use only that subject's own trial
statistics (no labels), so they're valid at zero-shot eval time with no calibration-data leakage.
"""
import numpy as np
from pyriemann.geometry.base import invsqrtm
from pyriemann.geometry.mean import mean_covariance


def _trial_covariances(X: np.ndarray) -> np.ndarray:
    """X: (n_trials, n_channels, n_times) -> (n_trials, n_channels, n_channels)."""
    return np.array([x @ x.T / x.shape[-1] for x in X])


def euclidean_alignment(X: np.ndarray) -> np.ndarray:
    """Reference covariance = arithmetic mean of per-trial covariances (He & Wu, 2020)."""
    covs = _trial_covariances(X)
    ref = covs.mean(axis=0)
    whitening = invsqrtm(ref)
    return np.array([whitening @ x for x in X])


def riemannian_alignment(X: np.ndarray) -> np.ndarray:
    """Reference covariance = Riemannian (affine-invariant geometric) mean (Zanini et al., 2018)."""
    covs = _trial_covariances(X)
    ref = mean_covariance(covs, metric="riemann")
    whitening = invsqrtm(ref)
    return np.array([whitening @ x for x in X])


def align_per_subject(per_subject_X: dict, method: str) -> dict:
    """Applies alignment independently to each subject's own trials (unsupervised, per-subject --
    valid for the held-out zero-shot subject too, since no labels or other subjects' data are used).

    per_subject_X: {subject_id: X} where X is (n_trials, n_channels, n_times).
    method: "none" | "ea" | "ra".
    """
    if method == "none":
        return per_subject_X
    fn = {"ea": euclidean_alignment, "ra": riemannian_alignment}[method]
    return {subj: fn(X) for subj, X in per_subject_X.items()}
