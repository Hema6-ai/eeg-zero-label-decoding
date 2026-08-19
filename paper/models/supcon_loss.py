"""Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)."""
import torch
import torch.nn.functional as F


def supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """features: (N, D) raw (not yet normalized) embeddings. labels: (N,) int64, with the two
    augmented views of the same trial sharing the same label so they become positive pairs.
    """
    device = features.device
    features = F.normalize(features, dim=1)
    batch_size = features.shape[0]

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    anchor_dot_contrast = torch.matmul(features, features.T) / temperature
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    logits_mask = 1 - torch.eye(batch_size, device=device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    mask_pos_pairs = mask.sum(1)
    mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

    return -mean_log_prob_pos.mean()
