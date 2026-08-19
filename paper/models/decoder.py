"""Lightweight MAE decoder: reconstructs raw patch values at masked positions."""
import torch
import torch.nn as nn


class MAEDecoder(nn.Module):
    def __init__(self, n_channels: int, patch_size: int, n_patches: int,
                 encoder_dim: int = 256, decoder_dim: int = 128, depth: int = 4, heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.n_patches = n_patches
        self.patch_dim = n_channels * patch_size

        self.embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=heads, dim_feedforward=int(decoder_dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, self.patch_dim)

    def forward(self, encoded_visible: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """encoded_visible: (B, 1 + N_visible, D_enc) with CLS at position 0.

        Returns reconstructed patches for ALL patch positions: (B, n_patches, patch_dim).
        """
        x = self.embed(encoded_visible)
        cls, visible = x[:, :1, :], x[:, 1:, :]

        b, n_visible, d = visible.shape
        n_masked = self.n_patches - n_visible
        mask_tokens = self.mask_token.expand(b, n_masked, -1)
        x_ = torch.cat([visible, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, d))

        x = torch.cat([cls, x_], dim=1) + self.pos_embed
        x = self.norm(self.blocks(x))
        return self.pred(x[:, 1:, :])
