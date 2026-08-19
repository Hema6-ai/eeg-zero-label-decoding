"""On-the-fly EEG augmentation for pretraining only (never applied at eval/zero-shot time)."""
import torch
import torch.nn.functional as F


def time_warp(x: torch.Tensor, max_warp: float = 0.1) -> torch.Tensor:
    """Randomly stretches/compresses each trial's time axis by up to +/-max_warp, then
    resamples back to the original length via linear interpolation."""
    b, c, t = x.shape
    scale = 1.0 + (torch.rand(1).item() * 2 - 1) * max_warp
    warped_len = max(8, int(round(t * scale)))
    warped = F.interpolate(x, size=warped_len, mode="linear", align_corners=False)
    return F.interpolate(warped, size=t, mode="linear", align_corners=False)


def channel_dropout(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    """Zeros out a random subset of channels per-trial (spatial dropout)."""
    b, c, t = x.shape
    mask = (torch.rand(b, c, 1, device=x.device) > p).float()
    return x * mask


def augment(x: torch.Tensor, time_warp_prob: float = 0.5, max_warp: float = 0.1,
            channel_dropout_prob: float = 0.1) -> torch.Tensor:
    if torch.rand(1).item() < time_warp_prob:
        x = time_warp(x, max_warp)
    x = channel_dropout(x, channel_dropout_prob)
    return x
