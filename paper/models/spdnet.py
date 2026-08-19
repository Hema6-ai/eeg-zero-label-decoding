"""SPDNet (Huang & Van Gool, 2017, "A Riemannian Network for SPD Matrix Learning"): unlike every
other model tried in this project, this never flattens covariances into a Euclidean tangent-space
vector until the very last layer. BiMap/ReEig layers operate directly on SPD matrices, preserving
the manifold structure throughout the learned transformation instead of linearizing it up front
(which is what TangentSpace + transformer/MLP does everywhere else in this project). Motivation:
every method that stays close to the native covariance geometry (CSP, FgMDM, TS+LR+EA) has beaten
every method that either uses raw EEG or flattens to tangent space early (MAE variants, SupCon,
both Riemannian Transformer versions) -- SPDNet is the one remaining architecture class that keeps
the whole pipeline on-manifold.

    BiMap:  X -> W^T X W        (learned dimensionality reduction, W semi-orthogonal)
    ReEig:  X -> U max(eps, S) U^T   (eigenvalue floor, X = U S U^T; nonlinearity analogous to ReLU)
    LogEig: X -> U log(S) U^T, then flatten upper triangle  (final Euclidean embedding, for a
            standard linear/MLP classifier -- this is the ONLY point in the whole network the
            output stops being SPD)
"""
import torch
import torch.nn as nn


class BiMap(nn.Module):
    """W is re-orthonormalized via QR every forward pass (differentiable), so it's always exactly
    semi-orthogonal without needing a separate Riemannian/Stiefel-manifold optimizer -- any
    standard optimizer (Adam/AdamW) works directly on the raw (unconstrained) parameter."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        assert d_out <= d_in, "BiMap only reduces dimensionality (d_out <= d_in)"
        self.weight = nn.Parameter(torch.empty(d_in, d_out))
        nn.init.orthogonal_(self.weight)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (B, d_in, d_in) SPD -> (B, d_out, d_out) SPD."""
        W, _ = torch.linalg.qr(self.weight)          # (d_in, d_out), orthonormal columns
        return W.T.unsqueeze(0) @ X @ W.unsqueeze(0)


class ReEig(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        eigvals, eigvecs = torch.linalg.eigh(X)
        eigvals = torch.clamp(eigvals, min=self.eps)
        return eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)


class LogEig(nn.Module):
    """Matrix logarithm, then flatten the (symmetric) upper triangle to a vector -- the tangent
    space of the manifold at this final reduced dimension, analogous to pyriemann's TangentSpace
    but applied after the learned BiMap/ReEig stack rather than on the raw input covariance."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        eigvals, eigvecs = torch.linalg.eigh(X)
        eigvals = torch.clamp(eigvals, min=self.eps)
        log_X = eigvecs @ torch.diag_embed(torch.log(eigvals)) @ eigvecs.transpose(-1, -2)
        d = log_X.shape[-1]
        idx = torch.triu_indices(d, d, device=X.device)
        return log_X[..., idx[0], idx[1]]


class SPDNetEncoder(nn.Module):
    """dims: e.g. [22, 16, 10] -- input channel count down to a smaller SPD dimension via two
    BiMap+ReEig stages. Output is a Euclidean embedding of size dims[-1]*(dims[-1]+1)/2."""

    def __init__(self, dims=(22, 16, 10)):
        super().__init__()
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(BiMap(d_in, d_out))
            layers.append(ReEig())
        self.spd_layers = nn.Sequential(*layers)
        self.logeig = LogEig()
        self.embed_dim = dims[-1] * (dims[-1] + 1) // 2

    def forward(self, cov: torch.Tensor) -> torch.Tensor:
        """cov: (B, C, C) SPD covariance matrices -> (B, embed_dim) Euclidean embedding."""
        return self.logeig(self.spd_layers(cov))


class SPDNetModel(nn.Module):
    """Bundles the SPDNet encoder + a linear classification head for pretraining; only the
    encoder is used downstream (frozen zero-shot eval discards this pretrain head entirely)."""

    def __init__(self, dims=(22, 16, 10), n_classes=3):
        super().__init__()
        self.encoder = SPDNetEncoder(dims)
        self.head = nn.Linear(self.encoder.embed_dim, n_classes)

    def forward(self, cov: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(cov))
