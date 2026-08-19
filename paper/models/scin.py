"""Subject-Conditional Instance Normalization: removes per-subject amplitude/scale quirks
from the raw EEG before the encoder ever sees it, applied on top of the existing per-trial
z-score (which is per-channel-per-trial; SCIN adds a learned per-subject affine correction).
"""
import torch
import torch.nn as nn


class SCIN(nn.Module):
    def __init__(self, n_channels: int, n_subjects: int):
        super().__init__()
        self.gamma = nn.Embedding(n_subjects, n_channels)
        self.beta = nn.Embedding(n_subjects, n_channels)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-6
        x_norm = (x - mean) / std
        gamma = self.gamma(subject_ids).unsqueeze(-1)
        beta = self.beta(subject_ids).unsqueeze(-1)
        return gamma * x_norm + beta

    def forward_mean(self, x: torch.Tensor) -> torch.Tensor:
        """For zero-shot subjects with no learned embedding row: use the mean gamma/beta
        across all pretraining subjects as a generic, subject-agnostic affine correction."""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-6
        x_norm = (x - mean) / std
        gamma = self.gamma.weight.mean(dim=0).view(1, -1, 1)
        beta = self.beta.weight.mean(dim=0).view(1, -1, 1)
        return gamma * x_norm + beta
