"""Splits (channels, time) EEG epochs into flattened time patches and embeds them."""
import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbedding(nn.Module):
    """(B, C, T) -> patch tokens (B, N, embed_dim), zero-padding T up to a multiple of patch_size.

    Each patch spans all channels for `patch_size` consecutive time samples, flattened to
    a (C * patch_size) vector before the linear projection -- this keeps spatial (channel)
    structure inside each token rather than treating channels as separate token streams.
    """

    def __init__(self, n_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.proj = nn.Linear(n_channels * patch_size, embed_dim)

    def n_patches(self, n_times: int) -> int:
        return (n_times + self.patch_size - 1) // self.patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        b, c, t = x.shape
        n_patches = self.n_patches(t)
        pad = n_patches * self.patch_size - t
        if pad > 0:
            x = nn.functional.pad(x, (0, pad))
        x = rearrange(x, "b c (n p) -> b n (c p)", p=self.patch_size)
        return self.proj(x)
