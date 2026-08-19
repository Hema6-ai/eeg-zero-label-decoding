"""Riemannian Transformer: covariance -> tangent space -> transformer, instead of raw EEG ->
transformer (the MAE approach, which failed -- Section on representational analysis showed
covariance-space features (CSP, FgMDM) are far more class-discriminative than anything the raw-EEG
transformer learned). This model operates entirely on already-informative tangent-space features,
using the transformer only to refine/re-weight them, not to discover structure from raw signal.

Each trial becomes a single token (no time axis, so no positional embedding) plus a CLS token;
the "sequence" the transformer attends over is just [CLS, trial_token].
"""
import torch
import torch.nn as nn


class TangentSpaceTransformer(nn.Module):
    def __init__(self, tangent_dim=253, embed_dim=128, depth=4, heads=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(tangent_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def cls_representation(self, tangent_vec: torch.Tensor) -> torch.Tensor:
        """tangent_vec: (B, tangent_dim) -> (B, embed_dim) pooled CLS representation."""
        token = self.proj(tangent_vec).unsqueeze(1)          # (B, 1, embed_dim)
        cls = self.cls_token.expand(token.shape[0], -1, -1)  # (B, 1, embed_dim)
        seq = torch.cat([cls, token], dim=1)                 # (B, 2, embed_dim)
        encoded = self.norm(self.blocks(seq))
        return encoded[:, 0, :]


class PretrainHead(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


class ReZeroEncoderLayer(nn.Module):
    """Pre-LN self-attention block whose attention/FFN contributions are each gated by a
    learnable scalar initialized to 0 (ReZero, Bachlechner et al. 2020). At random init,
    self-attention logits are near-zero, so softmax attention is nearly uniform over tokens
    and acts as a pure averaging operator -- for a permutation-symmetric, un-ordered token
    set (trials, no positional embedding) this collapses every token to the same vector
    within a single layer (verified empirically: untrained plain TransformerEncoderLayer
    already gives ~0.93 mean pairwise cosine similarity between trial tokens, vs ~0.29 for
    the raw linear projection, and approaches ~0.995 by depth 6). Gating each sublayer's
    output to ~0 at init makes the block start as an identity function, preserving the
    healthy input diversity, and lets the network learn how much attention to use instead
    of being forced through an averaging operator from step one.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.alpha_attn = nn.Parameter(torch.zeros(1))
        self.alpha_ff = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed, need_weights=False)
        x = x + self.alpha_attn * self.dropout1(attn_out)
        x = x + self.alpha_ff * self.dropout2(self.ff(self.norm2(x)))
        return x


class TrialAttentionTransformer(nn.Module):
    """v2: attends ACROSS a set of N trials (from the same subject) instead of a single trial
    per forward pass. v1 gave the transformer a 2-token sequence (CLS + 1 trial) -- effectively
    nothing to attend to. Here the sequence is [CLS, trial_1, ..., trial_N], so self-attention
    can relate trials to each other; per-trial classification uses each trial's own (attended)
    token, not the CLS token, which instead aggregates subject-level context.
    """

    def __init__(self, tangent_dim=253, embed_dim=128, depth=6, heads=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(tangent_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            ReZeroEncoderLayer(d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim * 4, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tangent_vecs: torch.Tensor):
        """tangent_vecs: (B, N, tangent_dim) -- N trials, same subject, mixed classes.
        Returns (cls_out (B, embed_dim), trial_out (B, N, embed_dim))."""
        tokens = self.proj(tangent_vecs)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        for block in self.blocks:
            seq = block(seq)
        encoded = self.norm(seq)
        return encoded[:, 0, :], encoded[:, 1:, :]


class TrialAttentionModel(nn.Module):
    """Bundles the trial-attention encoder + a per-trial classification head for pretraining."""

    def __init__(self, tangent_dim=253, embed_dim=128, depth=6, heads=4, n_classes=3):
        super().__init__()
        self.encoder = TrialAttentionTransformer(tangent_dim, embed_dim, depth, heads)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, tangent_vecs):
        _, trial_out = self.encoder(tangent_vecs)
        return self.head(trial_out)  # (B, N, n_classes)


class RiemannianTransformerModel(nn.Module):
    """Bundles encoder + pretrain head so a single checkpoint holds both; only the encoder is
    used downstream (frozen zero-shot eval discards the pretrain head entirely)."""

    def __init__(self, tangent_dim=253, embed_dim=128, depth=4, heads=4, n_classes=3):
        super().__init__()
        self.encoder = TangentSpaceTransformer(tangent_dim, embed_dim, depth, heads)
        self.head = PretrainHead(embed_dim, n_classes)

    def forward(self, tangent_vec):
        return self.head(self.encoder.cls_representation(tangent_vec))
