"""Phase 4 — Loss functions for student training.

Two pseudo-label loss variants (ablation) + multi-view consistency loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCENegSamplingLoss(nn.Module):
    """Variant A: BCE with explicit positive/negative sampling.

    Positives come from pseudo-labels (target=1). Negatives are randomly
    sampled from the complement (target=0). This avoids assuming all
    non-pseudo-label species are truly absent.
    """

    def __init__(self, num_classes: int = 7806, num_negatives: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.num_negatives = num_negatives

    def forward(
        self,
        logits: torch.Tensor,              # [B, C]
        pseudo_labels: list[torch.Tensor],  # list of [L_i] int tensors
    ) -> torch.Tensor:
        B, C = logits.shape
        device = logits.device

        target = torch.zeros(B, C, device=device)
        weight = torch.zeros(B, C, device=device)

        for i, labels in enumerate(pseudo_labels):
            pos = labels.to(device, dtype=torch.long)
            if len(pos) == 0:
                continue

            # Mark positives
            target[i, pos] = 1.0
            weight[i, pos] = 1.0

            # Sample negatives (excluding positives)
            neg_mask = torch.ones(C, dtype=torch.bool, device=device)
            neg_mask[pos] = False
            neg_pool = neg_mask.nonzero(as_tuple=True)[0]
            n_neg = min(self.num_negatives, len(neg_pool))
            perm = torch.randperm(len(neg_pool), device=device)[:n_neg]
            weight[i, neg_pool[perm]] = 1.0

        if weight.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = F.binary_cross_entropy_with_logits(
            logits, target, weight=weight, reduction="sum"
        )
        return loss / weight.sum()


class SoftDistillationLoss(nn.Module):
    """Variant B (recommended): KL-divergence via BCE with soft teacher targets.

    For independent Bernoulli outputs, KL(teacher || student) equals
    BCE(student_logits, teacher_probs) up to a constant (teacher entropy).
    Minimizing BCE with soft targets is equivalent to minimizing KL-divergence.

    This preserves the teacher's uncertainty across all 7806 classes —
    the student learns the full probability landscape, not just hard 0/1.
    """

    def forward(
        self,
        logits: torch.Tensor,         # [B, C]
        teacher_probs: torch.Tensor,  # [B, C] in [0, 1]
        topk_idx: torch.Tensor | None = None,  # [B, K] — focus on top-K classes
    ) -> torch.Tensor:
        if topk_idx is not None:
            logits = logits.gather(1, topk_idx)
            teacher_probs = teacher_probs.gather(1, topk_idx)
        return F.binary_cross_entropy_with_logits(
            logits, teacher_probs, reduction="mean"
        )


class ConsistencyLoss(nn.Module):
    """Multi-view consistency loss.

    Forces the student to predict the same species regardless of which
    geometric view (orig / flip / rot90) of the tiles it receives.

    Flip and rot90 predictions are stop-gradiented (MeanTeacher style):
    only the original view's logits receive gradients from this loss.

    Modes:
      logit_mse — MSE on raw logits (avoids sigmoid saturation)
      cosine    — 1 - cosine_similarity on logit vectors (scale-invariant)
      prob_mse  — MSE on sigmoid outputs (original; broken when logits saturate)
    """

    def __init__(self, mode: str = "logit_mse"):
        super().__init__()
        if mode not in ("logit_mse", "cosine", "prob_mse"):
            raise ValueError(f"Unknown consistency mode: {mode}")
        self.mode = mode

    def _pairwise(
        self, a: torch.Tensor, b: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "logit_mse":
            return F.mse_loss(a, b)
        if self.mode == "cosine":
            return (1.0 - F.cosine_similarity(a, b, dim=1)).mean()
        # prob_mse (original formulation)
        return F.mse_loss(torch.sigmoid(a), torch.sigmoid(b))

    def forward(
        self,
        logits_orig: torch.Tensor,   # [B, C]
        logits_flip: torch.Tensor,   # [B, C] — detached
        logits_rot90: torch.Tensor,  # [B, C] — detached
        topk_idx: torch.Tensor | None = None,  # [B, K] — focus on top-K classes
    ) -> torch.Tensor:
        if topk_idx is not None:
            logits_orig = logits_orig.gather(1, topk_idx)
            logits_flip = logits_flip.gather(1, topk_idx)
            logits_rot90 = logits_rot90.gather(1, topk_idx)
        loss_flip = self._pairwise(logits_orig, logits_flip)
        loss_rot90 = self._pairwise(logits_orig, logits_rot90)
        return (loss_flip + loss_rot90) / 2.0
