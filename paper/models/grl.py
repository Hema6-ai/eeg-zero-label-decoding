"""Gradient Reversal Layer -- identity forward, negated (scaled) gradient backward."""
import torch
import torch.nn as nn


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps `_GradReverse` with a mutable `lambd` so callers can schedule it (e.g. linear warm-up)."""

    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradReverse.apply(x, self.lambd)


class SubjectAdversary(nn.Module):
    """GRL -> linear subject classifier, applied to the pooled CLS token only.

    Design choice under test: adversarial disentanglement acts on the global
    representation (CLS), not on individual patch tokens.
    """

    def __init__(self, embed_dim: int, n_subjects: int, lambd: float = 1.0):
        super().__init__()
        self.grl = GradientReversalLayer(lambd)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, n_subjects),
        )

    def set_lambda(self, lambd: float) -> None:
        self.grl.lambd = lambd

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.grl(cls_token))
