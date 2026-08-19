"""Variational Information Bottleneck (Alemi et al., 2016) applied to the pooled CLS token.

Where CLUB explicitly targets I(z; subject), VIB instead bounds the *total* information
capacity of z via KL(q(z|x) || N(0, I)): z is resampled stochastically (reparameterization
trick) from a learned (mu, logvar) instead of being used deterministically. Compressing z's
total capacity is a strictly stronger constraint than removing one nuisance variable -- any
information the bottleneck squeezes out (including, but not limited to, subject identity)
cannot leak into z. This can be used standalone or stacked with CLUB (as done here): CLUB
explicitly pushes subject information out, VIB additionally caps how much of anything else
can pass through.
"""
import torch
import torch.nn as nn


class VIBBottleneck(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.mu = nn.Linear(embed_dim, embed_dim)
        self.logvar = nn.Linear(embed_dim, embed_dim)

    def forward(self, h: torch.Tensor):
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(min=-10, max=10)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
        return z, kl
