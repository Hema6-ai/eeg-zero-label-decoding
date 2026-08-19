"""Projection head for contrastive pretraining (SimCLR, Khosla et al.'s own SupCon setup): the
contrastive loss is computed on this head's L2-normalized output, not on the backbone
representation directly. The head is discarded after pretraining -- downstream (frozen zero-shot
eval, subject probe) uses the backbone's raw CLS token.

Optimizing a transformer's raw output directly against a sharp-temperature contrastive loss is a
known way to trigger representation collapse (the backbone's CLS output starts fairly homogeneous
across inputs before training teaches it otherwise, and the resulting gradients near that
homogeneous point can be too small to escape it). The projection head adds capacity specifically
for the contrastive objective, decoupling it from the representation quality of the backbone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, embed_dim=256, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)
