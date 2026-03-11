"""Phase 4 — Loss functions for student training.

Two pseudo-label loss variants (ablation) + multi-view consistency loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCENegSamplingLoss(nn.Module):
    """BCE with soft teacher targets and explicit negative sampling.

    Mirrors Exp-1's proven ``masked_distill_loss``:
      - Positives: classes where teacher_probs > 0 → trained with **soft**
        teacher probabilities (not hard 1.0)
      - Negatives: randomly sampled from the complement → target = 0
      - Zero-teacher entries are treated as *unknown*, not true negatives

    This avoids the mode-collapse caused by training on all 7806 classes
    (most of which are near-zero targets that overwhelm the positive signal).
    """

    def __init__(self, num_classes: int = 7806, num_negatives: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.num_negatives = num_negatives

    def forward(
        self,
        logits: torch.Tensor,         # [B, C]
        teacher_probs: torch.Tensor,   # [B, C] in [0, 1], sparse (mostly zeros)
    ) -> torch.Tensor:
        B, C = logits.shape
        device = logits.device

        pos_mask = teacher_probs > 0  # [B, C]
        pos_count = pos_mask.sum().item()

        # BCE on positives with soft teacher targets
        if pos_count > 0:
            pos_loss = F.binary_cross_entropy_with_logits(
                logits[pos_mask], teacher_probs[pos_mask], reduction="mean",
            )
        else:
            pos_loss = logits.sum() * 0.0

        # Random negative sampling from teacher == 0
        if self.num_negatives > 0:
            neg_mask = ~pos_mask
            neg_losses = []
            for i in range(B):
                zero_idx = neg_mask[i].nonzero(as_tuple=True)[0]
                if zero_idx.numel() == 0:
                    continue
                k = min(self.num_negatives, zero_idx.numel())
                sel = zero_idx[torch.randint(0, zero_idx.numel(), (k,), device=device)]
                neg_logits = logits[i, sel]
                neg_targets = torch.zeros_like(neg_logits)
                neg_losses.append(
                    F.binary_cross_entropy_with_logits(
                        neg_logits, neg_targets, reduction="mean",
                    )
                )
            neg_loss = torch.stack(neg_losses).mean() if neg_losses else logits.sum() * 0.0
        else:
            neg_loss = logits.sum() * 0.0

        return pos_loss + neg_loss


class SoftDistillationLoss(nn.Module):
    """KL-divergence via BCE with soft teacher targets.

    WARNING: This loss causes mode collapse (F1=0) because it trains on all
    7806 classes — the ~7742 near-zero targets overwhelm the ~64 positive
    signals. Use BCENegSamplingLoss instead.

    Kept for reference / ablation comparison only.
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
