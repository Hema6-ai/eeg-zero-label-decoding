"""EEG transformer encoder with CLS token and MAE-style random patch masking."""
import torch
import torch.nn as nn

from paper.models.patch_embed import PatchEmbedding


def random_masking(x: torch.Tensor, mask_ratio: float):
    """Per-sample random masking via shuffle/restore indices (He et al., MAE).

    Returns:
        x_visible: (B, N_visible, D) kept tokens
        mask: (B, N) binary, 1 = masked/removed, 0 = kept
        ids_restore: (B, N) indices to unshuffle back to original patch order
    """
    b, n, d = x.shape
    n_keep = max(1, int(n * (1 - mask_ratio)))

    noise = torch.rand(b, n, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :n_keep]
    x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, d))

    mask = torch.ones(b, n, device=x.device)
    mask[:, :n_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_visible, mask, ids_restore


class EEGTransformerEncoder(nn.Module):
    def __init__(self, n_channels: int, n_times: int, patch_size: int = 50,
                 embed_dim: int = 256, depth: int = 6, heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(n_channels, patch_size, embed_dim)
        n_patches = self.patch_embed.n_patches(n_times)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self.embed_dim = embed_dim
        self.n_patches = n_patches

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.0):
        """x: (B, C, T). If mask_ratio > 0, returns (encoded, mask, ids_restore); else returns encoded only.

        `encoded` always has shape (B, 1 + N_visible, D) with the CLS token at position 0.
        """
        tokens = self.patch_embed(x) + self.pos_embed[:, 1:, :]

        mask = ids_restore = None
        if mask_ratio > 0:
            tokens, mask, ids_restore = random_masking(tokens, mask_ratio)

        cls = self.cls_token.expand(tokens.shape[0], -1, -1) + self.pos_embed[:, :1, :]
        tokens = torch.cat([cls, tokens], dim=1)

        encoded = self.norm(self.blocks(tokens))
        if mask_ratio > 0:
            return encoded, mask, ids_restore
        return encoded

    def cls_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward (no masking) -> pooled CLS representation, for downstream/frozen use."""
        return self.forward(x, mask_ratio=0.0)[:, 0, :]
