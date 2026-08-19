"""EEG-MAE: masked-patch pretraining with an optional adversarial subject-disentanglement branch.

The adversarial branch (GRL -> subject classifier) is applied to the pooled CLS token only,
never to individual patch tokens -- this is the design choice under test in the ablation
(MAE-only vs MAE+GRL use the identical encoder/decoder and only toggle `use_grl`).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from paper.models.decoder import MAEDecoder
from paper.models.encoder import EEGTransformerEncoder
from paper.models.grl import SubjectAdversary


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, C, T) -> (B, n_patches, C * patch_size), zero-padded to match PatchEmbedding."""
    b, c, t = x.shape
    n_patches = (t + patch_size - 1) // patch_size
    pad = n_patches * patch_size - t
    if pad > 0:
        x = F.pad(x, (0, pad))
    return rearrange(x, "b c (n p) -> b n (c p)", p=patch_size)


class EEGMAE(nn.Module):
    def __init__(self, n_channels: int, n_times: int, n_subjects: int, patch_size: int = 50,
                 embed_dim: int = 256, encoder_depth: int = 6, encoder_heads: int = 8,
                 decoder_dim: int = 128, decoder_depth: int = 4, decoder_heads: int = 8,
                 mask_ratio: float = 0.75, use_grl: bool = True, grl_lambda: float = 1.0,
                 norm_pix_loss: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.use_grl = use_grl
        self.norm_pix_loss = norm_pix_loss

        self.encoder = EEGTransformerEncoder(
            n_channels, n_times, patch_size, embed_dim, encoder_depth, encoder_heads,
        )
        self.decoder = MAEDecoder(
            n_channels, patch_size, self.encoder.n_patches, embed_dim, decoder_dim,
            decoder_depth, decoder_heads,
        )
        if use_grl:
            self.subject_adversary = SubjectAdversary(embed_dim, n_subjects, grl_lambda)

    def set_grl_lambda(self, lambd: float) -> None:
        if self.use_grl:
            self.subject_adversary.set_lambda(lambd)

    def forward(self, x: torch.Tensor, subject_labels: torch.Tensor = None):
        encoded, mask, ids_restore = self.encoder(x, mask_ratio=self.mask_ratio)
        pred = self.decoder(encoded, ids_restore)

        target = patchify(x, self.patch_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            std = target.std(dim=-1, keepdim=True)
            target = (target - mean) / (std + 1e-6)

        loss_per_patch = (pred - target).pow(2).mean(dim=-1)  # (B, n_patches)
        recon_loss = (loss_per_patch * mask).sum() / mask.sum().clamp(min=1)

        losses = {"recon_loss": recon_loss}
        total_loss = recon_loss

        if self.use_grl:
            cls_token = encoded[:, 0, :]
            losses["subject_logits"] = self.subject_adversary(cls_token)
            if subject_labels is not None:
                adv_loss = F.cross_entropy(losses["subject_logits"], subject_labels)
                losses["adv_loss"] = adv_loss
                total_loss = total_loss + adv_loss

        losses["loss"] = total_loss
        return losses
