"""Linear probes trained on a frozen EEG-MAE encoder's CLS representation."""
import torch.nn as nn


class LinearProbe(nn.Module):
    """Single linear layer on top of a frozen encoder's pooled CLS token."""

    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, cls_token):
        return self.fc(cls_token)


class MLPProbe(nn.Module):
    """Two-layer MLP head on top of a frozen encoder's pooled CLS token."""

    def __init__(self, embed_dim: int, n_classes: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, cls_token):
        return self.net(cls_token)
