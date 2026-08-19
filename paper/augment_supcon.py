"""Augmentations for SupCon pretraining: two independently-augmented views per trial."""
import torch


def gaussian_noise(x: torch.Tensor, scale: float = 0.05) -> torch.Tensor:
    return x + torch.randn_like(x) * scale


def time_shift(x: torch.Tensor, max_shift: int = 50) -> torch.Tensor:
    """Per-sample independent shift -- a single shared shift for the whole batch would apply the
    identical temporal offset to every trial, reducing augmentation diversity within a batch and
    potentially letting the encoder key off batch-shared positional artifacts."""
    b, c, t = x.shape
    out = torch.empty_like(x)
    shifts = torch.randint(-max_shift, max_shift + 1, (b,))
    for i in range(b):
        out[i] = torch.roll(x[i], shifts=int(shifts[i].item()), dims=-1)
    return out


def channel_dropout(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    b, c, t = x.shape
    mask = (torch.rand(b, c, 1, device=x.device) > p).float()
    return x * mask


def time_masking(x: torch.Tensor, mask_fraction: float = 0.1) -> torch.Tensor:
    """Per-sample independent mask window -- see time_shift's docstring for why a single
    batch-shared mask position is a genuine augmentation-diversity bug, not just a style choice."""
    b, c, t = x.shape
    mask_len = int(t * mask_fraction)
    if mask_len < 1:
        return x
    x = x.clone()
    starts = torch.randint(0, t - mask_len + 1, (b,))
    for i in range(b):
        x[i, :, starts[i]:starts[i] + mask_len] = 0
    return x


def make_view(x: torch.Tensor) -> torch.Tensor:
    """Applies all four augmentations, each independently at random, to produce one view."""
    x = gaussian_noise(x, scale=0.05)
    x = time_shift(x, max_shift=50)
    x = channel_dropout(x, p=0.1)
    x = time_masking(x, mask_fraction=0.1)
    return x


def make_two_views(x: torch.Tensor):
    """Returns two independently-augmented views of the same batch, for SupCon."""
    return make_view(x), make_view(x)
