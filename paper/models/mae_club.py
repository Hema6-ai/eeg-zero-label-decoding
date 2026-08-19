"""EEG-MAE + SCIN (input-level subject normalization) + CLUB (nonlinear MI minimization on
the CLS token) + VIB (information-bottleneck compression of the CLS token, stacked on top of
CLUB for a stronger combined constraint).

This is a separate model class from EEGMAE (mae.py) rather than an extension of it: CLUB
requires its own optimizer and a two-step (MLE fit, then upper-bound minimization) training
procedure per batch, which doesn't fit the single `total_loss.backward()` pattern the
GRL-based EEGMAE uses. Keeping this self-contained also leaves the already-completed
MAE-only / MAE+GRL results untouched and reproducible.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from paper.models.club import CLUBEstimator
from paper.models.decoder import MAEDecoder
from paper.models.encoder import EEGTransformerEncoder
from paper.models.scin import SCIN
from paper.models.vib import VIBBottleneck


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    b, c, t = x.shape
    n_patches = (t + patch_size - 1) // patch_size
    pad = n_patches * patch_size - t
    if pad > 0:
        x = F.pad(x, (0, pad))
    return rearrange(x, "b c (n p) -> b n (c p)", p=patch_size)


class EEGMAEClubScin(nn.Module):
    def __init__(self, n_channels: int, n_times: int, n_subjects: int, patch_size: int = 50,
                 embed_dim: int = 256, encoder_depth: int = 6, encoder_heads: int = 8,
                 decoder_dim: int = 128, decoder_depth: int = 4, decoder_heads: int = 8,
                 mask_ratio: float = 0.75, club_lambda: float = 0.01, vib_beta: float = 1e-3,
                 norm_pix_loss: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.club_lambda = club_lambda
        self.vib_beta = vib_beta
        self.norm_pix_loss = norm_pix_loss

        self.scin = SCIN(n_channels, n_subjects)
        self.encoder = EEGTransformerEncoder(
            n_channels, n_times, patch_size, embed_dim, encoder_depth, encoder_heads,
        )
        self.decoder = MAEDecoder(
            n_channels, patch_size, self.encoder.n_patches, embed_dim, decoder_dim,
            decoder_depth, decoder_heads,
        )
        self.vib = VIBBottleneck(embed_dim)
        self.club = CLUBEstimator(embed_dim, n_subjects)

    def encode_zero_shot(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen-encoder feature extraction for unseen (zero-shot) subjects: SCIN uses the
        mean gamma/beta (no per-subject embedding exists for these subjects), VIB uses its
        deterministic mean (eval mode)."""
        x = self.scin.forward_mean(x)
        cls_token = self.encoder.cls_representation(x)
        z, _ = self.vib(cls_token)
        return z

    def encode_known_subject(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        """Frozen-encoder feature extraction for subjects seen during pretraining (used by the
        post-hoc subject-identity probe): SCIN uses that subject's own learned embedding."""
        x = self.scin(x, subject_ids)
        cls_token = self.encoder.cls_representation(x)
        z, _ = self.vib(cls_token)
        return z

    def forward(self, x: torch.Tensor, subject_labels: torch.Tensor):
        # Reconstruct the ORIGINAL (pre-SCIN) signal, not the SCIN-transformed one: SCIN's
        # gamma/beta are unconstrained, so if the target were computed post-SCIN, the model
        # could trivially minimize reconstruction loss by driving gamma->0 (collapsing the
        # target toward a constant per-subject value) instead of learning real EEG structure.
        # Keeping the target fixed to the original signal removes that shortcut: a collapsed
        # gamma now starves the encoder of real input and hurts reconstruction instead.
        x_scin = self.scin(x, subject_labels)

        encoded, mask, ids_restore = self.encoder(x_scin, mask_ratio=self.mask_ratio)
        pred = self.decoder(encoded, ids_restore)

        target = patchify(x, self.patch_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            std = target.std(dim=-1, keepdim=True)
            target = (target - mean) / (std + 1e-6)
        loss_per_patch = (pred - target).pow(2).mean(dim=-1)
        recon_loss = (loss_per_patch * mask).sum() / mask.sum().clamp(min=1)

        cls_token = encoded[:, 0, :]
        z, vib_kl = self.vib(cls_token)

        club_mle_loss = self.club.learning_loss(z.detach(), subject_labels)
        club_bound = self.club(z, subject_labels)

        main_loss = recon_loss + self.vib_beta * vib_kl + self.club_lambda * club_bound

        return {
            "recon_loss": recon_loss,
            "vib_kl": vib_kl,
            "club_bound": club_bound,
            "club_mle_loss": club_mle_loss,
            "main_loss": main_loss,
        }
