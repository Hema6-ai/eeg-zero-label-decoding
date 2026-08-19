"""EEGNet (Lawhern et al., 2018) -- per-subject supervised baseline for comparison."""
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, n_channels: int, n_times: int, n_classes: int,
                 f1: int = 8, depth_multiplier: int = 2, f2: int = 16,
                 kernel_length: int = 64, dropout: float = 0.5):
        super().__init__()
        f2 = f1 * depth_multiplier if f2 is None else f2

        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * depth_multiplier, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f1 * depth_multiplier, f1 * depth_multiplier, (1, 16),
                      padding=(0, 8), groups=f1 * depth_multiplier, bias=False),
            nn.Conv2d(f1 * depth_multiplier, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        flat_dim = f2 * (n_times // 32)
        self.classify = nn.Linear(flat_dim, n_classes)

    def forward(self, x):
        # x: (B, C, T) -> (B, 1, C, T)
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        return self.classify(x)
