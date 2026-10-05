"""Masked FP32 Flow-DPO, following MR-FlowDPO equation (2).

ACE-Step's time direction is data -> noise, so its target velocity is noise
minus data.  Reversing both the path and velocity relative to MelodyFlow does
not change the preference objective.  Rewards select/weight pairs; they are
not silently substituted for the denoising-error difference in the loss.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F


class FlowDPOResult(NamedTuple):
    loss: torch.Tensor
    dpo_loss: torch.Tensor
    chosen_fm: torch.Tensor
    rejected_fm: torch.Tensor
    reference_chosen_fm: torch.Tensor
    reference_rejected_fm: torch.Tensor
    preference_logits: torch.Tensor
    reference_penalty: torch.Tensor


def masked_flow_error(prediction: torch.Tensor, target: torch.Tensor,
                      mask: torch.Tensor | None = None) -> torch.Tensor:
    """Per-sample mean squared velocity error for [batch,time,channels]."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Flow predictions and targets must have identical [B,T,C] shapes")
    error = (prediction.float() - target.float()).square()
    if mask is None:
        return error.mean(dim=(1, 2))
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if mask.shape != prediction.shape[:2]:
        raise ValueError("attention mask must have shape [B,T]")
    mask = mask.to(device=prediction.device, dtype=torch.float32)
    if not torch.isfinite(mask).all() or (mask < 0).any():
        raise ValueError("attention mask must contain finite nonnegative values")
    denominator = mask.sum(dim=1) * prediction.shape[-1]
    if (denominator <= 0).any():
        raise ValueError("Every preference branch must contain at least one valid frame")
    # where avoids NaN/Inf from ignored padded targets contaminating valid data.
    error = torch.where(mask.unsqueeze(-1) > 0, error, 0.0)
    return (error * mask.unsqueeze(-1)).sum(dim=(1, 2)) / denominator


def flow_dpo_loss(
    chosen_prediction: torch.Tensor,
    rejected_prediction: torch.Tensor,
    chosen_reference: torch.Tensor,
    rejected_reference: torch.Tensor,
    chosen_target: torch.Tensor,
    rejected_target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    rejected_mask: torch.Tensor | None = None,
    beta: float = 2000.0,
    pair_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    fm_regularization: float = 0.0,
    reference_regularization: float = 0.0,
) -> FlowDPOResult:
    """Loss = softplus(beta * ((Ew-El) - (Erefw-Erefl))).

    The reference is detached defensively.  Optional smoothing, chosen-FM
    anchoring and reference-velocity anchoring are explicit extensions.
    """
    for name, value in (("beta", beta), ("fm_regularization", fm_regularization),
                        ("reference_regularization", reference_regularization)):
        if not math.isfinite(float(value)) or (value <= 0 if name == "beta" else value < 0):
            raise ValueError(f"{name} must be finite and {'positive' if name == 'beta' else 'nonnegative'}")
    if not math.isfinite(label_smoothing) or not 0 <= label_smoothing < 0.5:
        raise ValueError("label_smoothing must be in [0, 0.5)")
    rejected_mask = mask if rejected_mask is None else rejected_mask
    chosen_reference = chosen_reference.detach()
    rejected_reference = rejected_reference.detach()
    chosen_error = masked_flow_error(chosen_prediction, chosen_target, mask)
    rejected_error = masked_flow_error(rejected_prediction, rejected_target, rejected_mask)
    ref_chosen_error = masked_flow_error(chosen_reference, chosen_target, mask)
    ref_rejected_error = masked_flow_error(rejected_reference, rejected_target, rejected_mask)
    relative_gap = (chosen_error - rejected_error) - (ref_chosen_error - ref_rejected_error)
    logits = -float(beta) * relative_gap
    per_pair = ((1.0 - label_smoothing) * F.softplus(-logits)
                + label_smoothing * F.softplus(logits))
    reference_penalty = 0.5 * (
        masked_flow_error(chosen_prediction, chosen_reference, mask)
        + masked_flow_error(rejected_prediction, rejected_reference, rejected_mask))
    if pair_weight is None:
        weights = torch.ones_like(per_pair)
    else:
        weights = pair_weight.to(device=per_pair.device, dtype=torch.float32).reshape(-1)
        if weights.shape != per_pair.shape or not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("pair_weight must be finite, nonnegative, one per pair, with positive total")
    weights = weights / weights.sum()
    dpo = (weights * per_pair).sum()
    total = dpo + fm_regularization * (weights * chosen_error).sum()
    total = total + reference_regularization * (weights * reference_penalty).sum()
    return FlowDPOResult(total, dpo, chosen_error, rejected_error,
                         ref_chosen_error, ref_rejected_error, logits,
                         reference_penalty)
