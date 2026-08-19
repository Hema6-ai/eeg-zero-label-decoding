"""Categorical CLUB (Cheng et al., ICML 2020): a variational upper bound on I(z; subject).

Unlike GRL, which reverses a classifier's gradient into the encoder (and can degrade to a
regime where the classifier and encoder both plateau near chance-level *loss* without the
representation actually becoming subject-uninformative -- exactly the failure mode observed
with GRL here, where post-hoc probe accuracy stayed ~8x above chance despite near-chance
in-training adversary loss), CLUB explicitly upper-bounds the mutual information via a
variational approximation q_theta(subject | z):

    I(z; subject) <= E_{p(z,s)}[log q(s|z)] - E_{p(z)p(s)}[log q(s|z)]

The right-hand side is estimated in-batch: the first term uses true (z, subject) pairs, the
second uses z paired with a random in-batch permutation of subject labels (an empirical draw
from the marginal p(s)). Subject identity is categorical here (not the continuous variable
CLUB's original Gaussian form assumes), so q_theta(s|z) is a standard softmax classifier and
"log q(s|z)" is its log-likelihood -- the same network shape as the GRL adversary, but used
differently.

Correct training requires TWO separate steps per batch (this is why CLUB cannot reuse the
single `total_loss.backward()` pattern GRL used):
  1. `learning_loss(z.detach(), subject)`: a plain cross-entropy MLE fit of q_theta to the
     *current* representation, optimized by its own optimizer. This must see z.detach() --
     if the encoder's gradient were allowed through here, the classifier could "help" the
     encoder hide information instead of honestly fitting p(subject | z), which would make
     the upper bound meaningless.
  2. `forward(z, subject)`: the upper-bound estimate above, added to the encoder's loss with
     its own coefficient. Gradients here flow into z (and hence the encoder) to *decrease*
     the bound; any incidental gradient into this module's own parameters is harmless since
     only the learning_loss step's optimizer.step() is ever applied to them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLUBEstimator(nn.Module):
    def __init__(self, embed_dim: int, n_subjects: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_subjects),
        )

    def learning_loss(self, z: torch.Tensor, subject_labels: torch.Tensor) -> torch.Tensor:
        """MLE fit of q_theta(subject | z). Call with z detached from the encoder graph."""
        logits = self.net(z)
        return F.cross_entropy(logits, subject_labels)

    def forward(self, z: torch.Tensor, subject_labels: torch.Tensor) -> torch.Tensor:
        """CLUB upper-bound estimate of I(z; subject); minimize this to reduce leakage."""
        log_probs = F.log_softmax(self.net(z), dim=-1)
        pos = log_probs.gather(1, subject_labels.unsqueeze(1)).squeeze(1).mean()

        shuffled = subject_labels[torch.randperm(subject_labels.size(0), device=z.device)]
        neg = log_probs.gather(1, shuffled.unsqueeze(1)).squeeze(1).mean()

        return pos - neg
