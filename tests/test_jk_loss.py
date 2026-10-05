import math

import pytest
import torch

from jk_step.loss import flow_dpo_loss, masked_flow_error


def test_same_policy_and_reference_is_log_two():
    torch.manual_seed(3)
    chosen, rejected, target = [torch.randn(2, 4, 3) for _ in range(3)]
    result = flow_dpo_loss(chosen, rejected, chosen.clone(), rejected.clone(), target, target)
    assert result.loss.item() == pytest.approx(math.log(2), abs=1e-6)
    assert result.loss.dtype == torch.float32


def test_true_dpo_sign_favors_chosen_relative_to_reference():
    zero = torch.zeros(1, 2, 3)
    one = torch.ones_like(zero)
    good = flow_dpo_loss(zero, one, zero, zero, zero, zero, beta=2)
    bad = flow_dpo_loss(one, zero, zero, zero, zero, zero, beta=2)
    assert good.loss < math.log(2) < bad.loss
    assert good.preference_logits.item() == pytest.approx(2)


def test_reference_is_a_real_baseline_not_an_absolute_ranking():
    zero = torch.zeros(1, 2, 3)
    one = torch.ones_like(zero)
    result = flow_dpo_loss(zero, one, zero, one, zero, zero, beta=2000)
    assert result.loss.item() == pytest.approx(math.log(2), abs=1e-6)


def test_masks_mean_only_valid_frames_and_channels():
    prediction = torch.tensor([[[1.0, 3.0], [1000.0, 1000.0]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[1, 0]])
    assert masked_flow_error(prediction, target, mask).item() == 5.0


def test_gradients_reach_policy_but_never_reference():
    prediction = torch.ones(2, 3, 4, requires_grad=True)
    rejected = torch.zeros_like(prediction, requires_grad=True)
    ref = torch.zeros_like(prediction, requires_grad=True)
    target = torch.zeros_like(prediction)
    result = flow_dpo_loss(prediction, rejected, ref, ref, target, target, beta=2)
    result.loss.backward()
    assert prediction.grad is not None and prediction.grad.abs().sum() > 0
    assert ref.grad is None


def test_fp32_reduction_prevents_half_precision_overflow():
    prediction = torch.full((1, 2, 3), 500, dtype=torch.float16)
    target = torch.zeros_like(prediction)
    error = masked_flow_error(prediction, target)
    assert torch.isfinite(error).all()
    assert error.item() == 250000


def test_zero_weight_pair_does_not_change_the_weighted_loss():
    zero = torch.zeros(2, 2, 3)
    prediction = torch.stack([torch.zeros(2, 3), torch.ones(2, 3)])
    result = flow_dpo_loss(prediction, zero, zero, zero, zero, zero,
                           beta=2, pair_weight=torch.tensor([1.0, 0.0]))
    assert result.loss.item() == pytest.approx(math.log(2), abs=1e-6)


def test_regularizers_are_explicit_and_not_substitutes_for_dpo():
    zero = torch.zeros(1, 2, 3)
    one = torch.ones_like(zero)
    base = flow_dpo_loss(one, zero, zero, zero, zero, zero, beta=1)
    anchored = flow_dpo_loss(one, zero, zero, zero, zero, zero, beta=1,
                             fm_regularization=0.25, reference_regularization=0.5)
    assert anchored.dpo_loss == base.dpo_loss
    assert (anchored.loss - base.loss).item() == pytest.approx(0.5)


@pytest.mark.parametrize("option,value", [("beta", 0), ("beta", float("nan")),
                                         ("label_smoothing", 0.5), ("fm_regularization", -1)])
def test_invalid_objective_settings_fail_early(option, value):
    zero = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError):
        flow_dpo_loss(zero, zero, zero, zero, zero, zero, **{option: value})


def test_empty_mask_is_rejected():
    zero = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="valid frame"):
        masked_flow_error(zero, zero, torch.zeros(1, 2))
