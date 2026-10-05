"""ACE-Step decoder LoRA preference and supervised training for JK-Step.

The reference is the original frozen checkpoint, evaluated by disabling PEFT
adapters.  It is never a second copy of the XL model.  The actual objective is
the published velocity-error Flow-DPO objective, not supervised fine-tuning
renamed DPO.  Multi-reward selection belongs to the preference-data builder.
The optional sft objective uses a single target and ordinary masked flow
matching, with no rejected audio or frozen-reference forward.

ML imports are local to the entry points so CLI help/UI configuration remains
available on a fresh installation.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from jk_step.config import parse_layers, validate_config


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".writing")
    staged.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                allow_nan=False), encoding="utf-8")
    staged.replace(path)


def _hardware(config: dict) -> tuple[Any, str, Any]:
    import torch
    requested = str(config.get("device", "auto"))
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected but this PyTorch installation cannot use CUDA")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} does not exist")
        torch.cuda.set_device(index)
    if device.type not in ("cuda", "cpu", "mps", "xpu"):
        raise ValueError("Supported device types: cpu, cuda, mps, xpu")
    precision = config.get("precision", "auto")
    if precision == "auto":
        precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 mixed precision currently requires CUDA; use fp32 or bf16")
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("The selected GPU does not support bf16; select fp16 or fp32")
    dtype = {"fp32": torch.float32, "fp16": torch.float16,
             "bf16": torch.bfloat16}.get(precision)
    if dtype is None:
        raise ValueError("precision must be auto, fp32, fp16 or bf16")
    return device, precision, dtype


def load_base_model(config: dict) -> Any:
    """Load a full frozen ACE model from HF format or the original safetensors.

    For a single safetensors checkpoint, ``model_config_dir`` must contain the
    matching ACE architecture/configuration Python files.  Every parameter key
    and shape is checked before any weights are allocated.  Non-decoder modules
    can stay on CPU from the beginning.  This entry point also serves dataset
    preprocessing; pass offload_non_decoder=False to place all modules together.
    """
    import torch
    device, precision, dtype = _hardware(config)
    offload = bool(config.get("offload_non_decoder", True))
    checkpoint_file = str(config.get("checkpoint_file", "")).strip()
    if not checkpoint_file:
        from jk_engine.models.loader import load_decoder_for_training
        model = load_decoder_for_training(
            config["checkpoint_dir"], config.get("model_variant", "xl-sft"),
            device="cpu" if offload else str(device), precision=precision)
        if offload:
            model.decoder.to(device)
        return model

    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModel
    from jk_engine.models.acestep_remote_imports import _ensure_acestep_remote_imports
    _ensure_acestep_remote_imports()
    architecture_dir = Path(config.get("model_config_dir", "")).expanduser().resolve()
    if not (architecture_dir / "config.json").is_file():
        raise FileNotFoundError("model_config_dir must contain a matching config.json")
    model_config = AutoConfig.from_pretrained(
        str(architecture_dir), trust_remote_code=True, local_files_only=True)
    if getattr(model_config, "is_turbo", False):
        raise ValueError("JK-Step MR-FlowDPO requires a non-distilled SFT/base checkpoint")
    model_config.use_cache = False
    # Buffers remain real CPU tensors (including nonpersistent rotary inv_freq).
    with init_empty_weights(include_buffers=False):
        model = AutoModel.from_config(model_config, trust_remote_code=True,
                                     torch_dtype=dtype, attn_implementation="sdpa")
    expected = model.state_dict()
    prefixes = ("", "model.diffusion_model.", "diffusion_model.", "_orig_mod.")
    with safe_open(checkpoint_file, framework="pt", device="cpu") as checkpoint:
        original_keys = set(checkpoint.keys())
        selected_prefix = next((prefix for prefix in prefixes
                                if all(key.startswith(prefix) for key in original_keys)
                                and set(expected) == {key[len(prefix):] for key in original_keys
                                                     if key.startswith(prefix)}), None)
        if selected_prefix is None:
            missing = sorted(set(expected) - original_keys)[:8]
            extra = sorted(original_keys - set(expected))[:8]
            raise ValueError("Checkpoint architecture mismatch. Missing keys: "
                             f"{missing}; unexpected keys: {extra}. Use its matching model_config_dir.")
        for name, parameter in expected.items():
            shape = checkpoint.get_slice(selected_prefix + name).get_shape()
            if tuple(shape) != tuple(parameter.shape):
                raise ValueError(f"Checkpoint shape mismatch for {name}: {shape} versus {list(parameter.shape)}")
        for name in expected:
            tensor = checkpoint.get_tensor(selected_prefix + name)
            target_device = device if not offload or name.startswith("decoder.") else torch.device("cpu")
            set_module_tensor_to_device(model, name, target_device,
                                       value=tensor, dtype=dtype if tensor.is_floating_point() else None)
            del tensor
    del expected
    for name, buffer in model.named_buffers():
        if buffer.device.type == "meta":
            raise RuntimeError(f"Architecture produced an unmaterialized buffer {name}; use current matching ACE architecture files")
    model.decoder.to(device)
    if not offload:
        model.to(device)
    model.requires_grad_(False)
    model.eval()
    return model


def _target_names(decoder: Any, config: dict) -> list[str]:
    import torch
    layers = parse_layers(config["layers"])
    projections = config["target_modules"]
    mlp_names = ("gate_proj", "up_proj", "down_proj")
    selected = []
    for name, module in decoder.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        requested = any(name == suffix or name.endswith("." + suffix) for suffix in projections)
        is_mlp = name.rsplit(".", 1)[-1] in mlp_names and ".mlp." in name
        if not requested and not (config["target_mlp"] and is_mlp):
            continue
        if not is_mlp and config["attention_type"] != "both":
            if f"{config['attention_type']}_attn." not in name:
                continue
        if layers is not None:
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
            if match is None or int(match.group(1)) not in layers:
                continue
        selected.append(name)
    if not selected:
        raise ValueError("No decoder Linear layers matched target_modules/attention_type/layers")
    if layers is not None:
        found = {int(re.search(r"(?:^|\.)layers\.(\d+)\.", name).group(1)) for name in selected}
        if set(layers) - found:
            raise ValueError(f"Requested decoder layers do not match: {sorted(set(layers) - found)}")
    return selected


def _paired_dropout(probability: float) -> Any:
    import torch
    class PairedDropout(torch.nn.Module):
        """One Bernoulli mask per preference pair, reused for both branches."""
        def __init__(self, p):
            super().__init__()
            self.p = p

        def forward(self, tensor):
            if not self.training or self.p == 0:
                return tensor
            if tensor.shape[0] % 2:
                raise ValueError("Paired LoRA dropout requires [chosen, rejected] batches of size 2B")
            shape = (tensor.shape[0] // 2, *tensor.shape[1:])
            mask = torch.empty(shape, dtype=torch.float32, device=tensor.device)
            mask.bernoulli_(1.0 - self.p).div_(1.0 - self.p)
            return tensor * torch.cat((mask, mask), dim=0).to(tensor.dtype)
    return PairedDropout(probability)


def _inject_adapter(model: Any, config: dict) -> tuple[Any, list[Any]]:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    model.requires_grad_(False)
    model.eval()
    targets = _target_names(model.decoder, config)
    initial = config["resume_from"] or config["init_adapter"]
    if initial:
        existing = LoraConfig.from_pretrained(initial)
        settings = {"rank": existing.r, "alpha": existing.lora_alpha,
                    "dropout": existing.lora_dropout, "use_rslora": existing.use_rslora,
                    "rank_pattern": existing.rank_pattern, "alpha_pattern": existing.alpha_pattern}
        changed = [key for key, value in settings.items() if config[key] != value]
        if changed:
            raise ValueError("An existing adapter fixes these settings: " + ", ".join(changed)
                             + ". Match its adapter_config.json; creating a new rank requires a new adapter.")
        model.decoder = PeftModel.from_pretrained(model.decoder, initial, is_trainable=True,
                                                  autocast_adapter_dtype=True)
    else:
        # No language-model task type: the decoder takes ACE's hidden_states,
        # timestep and context tensors directly, not token IDs.
        adapter = LoraConfig(r=config["rank"], lora_alpha=config["alpha"],
                             lora_dropout=config["dropout"], target_modules=targets,
                             rank_pattern=config["rank_pattern"], alpha_pattern=config["alpha_pattern"],
                             bias="none", use_rslora=config["use_rslora"])
        model.decoder = get_peft_model(model.decoder, adapter, autocast_adapter_dtype=True)
    if config.get("objective", "flow_dpo") == "flow_dpo":
        for module in model.decoder.modules():
            if hasattr(module, "lora_dropout"):
                for adapter_name, dropout in module.lora_dropout.items():
                    probability = float(getattr(dropout, "p", 0.0))
                    module.lora_dropout[adapter_name] = _paired_dropout(probability)
    parameters = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name:
                raise RuntimeError(f"Unexpected non-adapter trainable parameter: {name}")
            parameter.data = parameter.data.float()
            parameters.append(parameter)
    if not parameters:
        raise RuntimeError("No trainable LoRA parameters were created")
    # Keep base attention dropout off, while checkpointing layers themselves
    # remain in training mode.  Adapter-only dropout is explicitly enabled.
    if config["gradient_checkpointing"]:
        base = model.decoder.get_base_model()
        if not hasattr(base, "gradient_checkpointing_enable"):
            raise ValueError("This decoder does not support gradient checkpointing")
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for module in model.decoder.modules():
        if hasattr(module, "config") and hasattr(module.config, "use_cache"):
            module.config.use_cache = False
    return model, parameters


def _policy_mode(decoder: Any, training: bool) -> None:
    decoder.eval()
    if training:
        # GradientCheckpointingLayer.__call__ checks its own training flag.
        # Its attention children remain in eval, disabling base dropout.
        for module in decoder.modules():
            if getattr(module, "gradient_checkpointing", False):
                module.training = True
            if hasattr(module, "lora_dropout"):
                for dropout in module.lora_dropout.values():
                    dropout.train()


def _autocast(device, dtype):
    import torch
    return (torch.autocast(device_type=device.type, dtype=dtype)
            if dtype != torch.float32 else contextlib.nullcontext())


def _prefix_lengths(mask: Any, name: str) -> list[int]:
    import torch
    if mask.ndim != 2 or not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError(f"{name} must be a binary [B,T] mask")
    lengths = mask.sum(dim=1).long()
    if (lengths <= 0).any():
        raise ValueError(f"{name} contains an empty sequence")
    expected = torch.arange(mask.shape[1], device=mask.device)[None] < lengths[:, None]
    if not torch.equal(expected, mask.bool()):
        raise ValueError(f"{name} must contain a contiguous valid prefix followed by padding")
    return lengths.tolist()


def _predict_pair(decoder: Any, chosen_xt: Any, rejected_xt: Any,
                  timestep: Any, batch: dict, *, require_input_grads: bool) -> tuple[Any, Any]:
    """Trim padding before ACE forwards; some ACE decoder revisions ignore masks.

    Group equal-length pairs together, ensuring padded frames/conditioning
    cannot influence real audio.  Re-padding occurs only after the network.
    """
    import torch
    import torch.nn.functional as F
    lengths = _prefix_lengths(batch["attention_mask"], "attention_mask")
    encoder_lengths = _prefix_lengths(batch["encoder_attention_mask"], "encoder_attention_mask")
    grouped = defaultdict(list)
    for index, key in enumerate(zip(lengths, encoder_lengths)):
        grouped[key].append(index)
    chosen_results, rejected_results = [None] * len(lengths), [None] * len(lengths)
    for (length, encoder_length), indices in grouped.items():
        chosen = chosen_xt[indices, :length]
        rejected = rejected_xt[indices, :length]
        hidden = torch.cat((chosen, rejected), dim=0)
        if require_input_grads:
            hidden = hidden.requires_grad_(True)
        cond = batch["encoder_hidden_states"][indices, :encoder_length]
        context = batch["context_latents"][indices, :length]
        times = timestep[indices]
        outputs = decoder(
            hidden_states=hidden, timestep=torch.cat((times, times)),
            timestep_r=torch.cat((times, times)),
            attention_mask=batch["attention_mask"][indices, :length].repeat(2, 1),
            encoder_hidden_states=cond.repeat(2, 1, 1),
            encoder_attention_mask=batch["encoder_attention_mask"][indices, :encoder_length].repeat(2, 1),
            context_latents=context.repeat(2, 1, 1), use_cache=False)
        prediction = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if hasattr(prediction, "sample"):
            prediction = prediction.sample
        if prediction.shape != hidden.shape:
            raise ValueError(f"Decoder returned {tuple(prediction.shape)}, expected {tuple(hidden.shape)}")
        predicted_chosen, predicted_rejected = prediction.chunk(2, dim=0)
        for local, index in enumerate(indices):
            pad = (0, 0, 0, chosen_xt.shape[1] - length)
            chosen_results[index] = F.pad(predicted_chosen[local], pad)
            rejected_results[index] = F.pad(predicted_rejected[local], pad)
    return torch.stack(chosen_results), torch.stack(rejected_results)


def preference_step(model: Any, batch: dict, config: dict, *, training: bool = True) -> Any:
    """One real preference microbatch, also exposed for small integration tests."""
    import torch
    from jk_step.loss import flow_dpo_loss
    device, _, dtype = _hardware(config)
    batch = {key: value.to(device, dtype=(torch.float32 if key in ("pair_weight", "attention_mask", "encoder_attention_mask")
                                        else dtype) if value.is_floating_point() else None,
                           non_blocking=device.type == "cuda")
             if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    chosen, rejected = batch["chosen_latents"], batch["rejected_latents"]
    if chosen.shape != rejected.shape or chosen.ndim != 3:
        raise ValueError("Chosen/rejected latents must be aligned [B,T,C] tensors")
    if not torch.isfinite(chosen).all() or not torch.isfinite(rejected).all():
        raise ValueError("Preference latents contain non-finite values")
    size = chosen.shape[0]
    if config["cfg_dropout"]:
        null = getattr(model, "null_condition_emb", None)
        if null is None:
            raise ValueError("CFG dropout requested but model has no null_condition_emb")
        drop = torch.rand(size, 1, 1, device=device) < config["cfg_dropout"]
        batch["encoder_hidden_states"] = torch.where(
            drop, null.to(device=device, dtype=dtype).expand_as(batch["encoder_hidden_states"]),
            batch["encoder_hidden_states"])
    if config["timestep_sampling"] == "uniform":
        timestep = torch.rand(size, device=device, dtype=torch.float32)
    else:
        # ACE's sample_t_r(use_meanflow=False): max of two logit-normal
        # draws, followed by r=t.  Do not accidentally train discrete Turbo t.
        draws = torch.randn(2, size, device=device, dtype=torch.float32)
        timestep = torch.sigmoid(draws * config["timestep_sigma"] + config["timestep_mu"]).amax(dim=0)
    timestep = timestep.clamp(config["timestep_min"], config["timestep_max"])
    noise = torch.randn(chosen.shape, device=device, dtype=torch.float32)
    times = timestep[:, None, None]
    chosen_xt = (times * noise + (1.0 - times) * chosen.float()).to(dtype)
    rejected_xt = (times * noise + (1.0 - times) * rejected.float()).to(dtype)
    chosen_target, rejected_target = noise - chosen.float(), noise - rejected.float()
    decoder = model.decoder
    # Reference first: no trainable gradients and absolutely no adapter/base
    # dropout.  The same t/noise/condition tensors are then reused by policy.
    decoder.eval()
    with torch.no_grad(), decoder.disable_adapter(), _autocast(device, dtype):
        chosen_ref, rejected_ref = _predict_pair(decoder, chosen_xt, rejected_xt, timestep,
                                                batch, require_input_grads=False)
    _policy_mode(decoder, training)
    with _autocast(device, dtype):
        chosen_prediction, rejected_prediction = _predict_pair(
            decoder, chosen_xt, rejected_xt, timestep, batch,
            require_input_grads=training and config["gradient_checkpointing"])
    weights = batch.get("pair_weight") if config["pair_weighting"] else None
    return flow_dpo_loss(chosen_prediction, rejected_prediction, chosen_ref, rejected_ref,
                         chosen_target, rejected_target, batch["attention_mask"],
                         beta=config["beta"], pair_weight=weights,
                         label_smoothing=config["label_smoothing"],
                         fm_regularization=config["fm_regularization"],
                         reference_regularization=config["reference_regularization"])


@dataclass
class SupervisedStepResult:
    loss: Any
    chosen_fm: Any


def _predict_supervised(decoder: Any, xt: Any, timestep: Any, batch: dict,
                        *, require_input_grads: bool) -> Any:
    """One branch per example, trimmed before XL's mask-ignoring decoder."""
    import torch
    import torch.nn.functional as F
    lengths = _prefix_lengths(batch["attention_mask"], "attention_mask")
    encoder_lengths = _prefix_lengths(batch["encoder_attention_mask"], "encoder_attention_mask")
    grouped = defaultdict(list)
    for index, shape in enumerate(zip(lengths, encoder_lengths)):
        grouped[shape].append(index)
    predictions = [None] * len(lengths)
    for (length, encoder_length), indices in grouped.items():
        hidden = xt[indices, :length]
        if require_input_grads:
            hidden = hidden.requires_grad_(True)
        times = timestep[indices]
        outputs = decoder(hidden_states=hidden, timestep=times, timestep_r=times,
            attention_mask=batch["attention_mask"][indices, :length],
            encoder_hidden_states=batch["encoder_hidden_states"][indices, :encoder_length],
            encoder_attention_mask=batch["encoder_attention_mask"][indices, :encoder_length],
            context_latents=batch["context_latents"][indices, :length], use_cache=False)
        prediction = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if hasattr(prediction, "sample"):
            prediction = prediction.sample
        if prediction.shape != hidden.shape:
            raise ValueError(f"Decoder returned {tuple(prediction.shape)}, expected {tuple(hidden.shape)}")
        for local, index in enumerate(indices):
            predictions[index] = F.pad(prediction[local], (0, 0, 0, xt.shape[1] - length))
    return torch.stack(predictions)


def supervised_step(model: Any, batch: dict, config: dict, *, training: bool = True) -> SupervisedStepResult:
    """Continuous single-target ACE flow matching; loss reduces in FP32.

    Preference coefficients (beta, label smoothing, pair weights and reference/
    chosen FM regularization) have no effect on this objective.
    """
    import torch
    device, _, dtype = _hardware(config)
    batch = {key: value.to(device, dtype=(torch.float32 if key in
                ("target_latents", "attention_mask", "encoder_attention_mask") else dtype)
                if value.is_floating_point() else None, non_blocking=device.type == "cuda")
             if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    target = batch["target_latents"]
    if target.ndim != 3 or target.shape[-1] != 64 or not torch.isfinite(target).all():
        raise ValueError("Supervised target_latents must be finite [B,T,64] tensors")
    size = target.shape[0]
    if config["cfg_dropout"]:
        null = getattr(model, "null_condition_emb", None)
        if null is None:
            raise ValueError("CFG dropout requested but model has no null_condition_emb")
        dropped = torch.rand(size, 1, 1, device=device) < config["cfg_dropout"]
        batch["encoder_hidden_states"] = torch.where(dropped,
            null.to(device=device, dtype=dtype).expand_as(batch["encoder_hidden_states"]),
            batch["encoder_hidden_states"])
    if config["timestep_sampling"] == "uniform":
        timestep = torch.rand(size, device=device, dtype=torch.float32)
    else:
        draws = torch.randn(2, size, device=device, dtype=torch.float32)
        timestep = torch.sigmoid(draws * config["timestep_sigma"] + config["timestep_mu"]).amax(dim=0)
    timestep = timestep.clamp(config["timestep_min"], config["timestep_max"])
    noise = torch.randn(target.shape, device=device, dtype=torch.float32)
    times = timestep[:, None, None]
    xt = (times * noise + (1 - times) * target.float()).to(dtype)
    velocity_target = noise - target.float()
    _policy_mode(model.decoder, training)
    with _autocast(device, dtype):
        prediction = _predict_supervised(model.decoder, xt, timestep, batch,
            require_input_grads=training and config["gradient_checkpointing"])
    # Reduce each example over valid frames/channels, then average examples.
    # Padding cannot affect the forward or the loss, even in variable batches.
    mask = batch["attention_mask"].float()
    squared = (prediction.float() - velocity_target).square().mean(dim=-1)
    per_sample = (squared * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return SupervisedStepResult(per_sample.mean(), per_sample)


def _training_step(model: Any, batch: dict, config: dict, *, training: bool = True) -> Any:
    function = supervised_step if config.get("objective", "flow_dpo") == "sft" else preference_step
    return function(model, batch, config, training=training)


def _step_metrics(result: Any, objective: str) -> dict:
    metrics = {"loss": result.loss, "chosen_fm": result.chosen_fm.mean()}
    if objective == "flow_dpo":
        metrics.update(dpo_loss=result.dpo_loss, rejected_fm=result.rejected_fm.mean(),
                       preference_margin=result.preference_logits.mean(),
                       preference_accuracy=(result.preference_logits > 0).float().mean())
    return metrics


def _rng_capture(device) -> dict:
    import torch
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    elif device.type == "xpu":
        state["xpu"] = torch.xpu.get_rng_state(device)
    return state


def _rng_restore(state: dict, device) -> None:
    import torch
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)
    elif device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"].cpu())
    elif device.type == "xpu" and "xpu" in state:
        torch.xpu.set_rng_state(state["xpu"].cpu(), device)


def _split_datasets(config: dict) -> tuple[Any, Any]:
    if config.get("objective", "flow_dpo") == "sft":
        from jk_step.supervised_data import split_supervised_datasets
        return split_supervised_datasets(config)
    from jk_step.preprocess import PreferenceTensorDataset
    from jk_step.pairs import read_manifest
    document, base = read_manifest(config["pairs_manifest"])
    pairs = [dict(pair) for pair in document.get("pairs", [])]
    if not pairs:
        raise ValueError("Preference manifest contains no pairs")
    # Resolve caches before passing in-memory split manifests to the dataset.
    for pair in pairs:
        path = Path(pair.get("tensor_path", ""))
        if not pair.get("tensor_path"):
            raise ValueError(f"Pair {pair.get('id', '?')} has no tensor_path; preprocess the pairs first")
        pair["tensor_path"] = str((path if path.is_absolute() else base / path).resolve())
    def group_key(pair):
        return str(pair.get("holdout_group") or pair.get("group_id") or pair.get("id"))
    explicit = any(pair.get("split") in ("validation", "val", "test") for pair in pairs)
    train_pairs, validation_pairs = [], []
    if explicit:
        grouped_splits = defaultdict(set)
        for pair in pairs:
            split = pair.get("split", "train")
            if split not in ("train", "validation", "val", "test"):
                raise ValueError(f"Unknown preference split {split!r}")
            canonical = "validation" if split in ("validation", "val") else split
            grouped_splits[group_key(pair)].add(canonical)
            if split == "test":
                continue
            (validation_pairs if canonical == "validation" else train_pairs).append(pair)
        if any(len(splits) > 1 for splits in grouped_splits.values()):
            raise ValueError("A recording group occurs in both train and validation; fix the split to avoid leakage")
    else:
        groups = sorted({group_key(pair) for pair in pairs})
        random.Random(config["seed"]).shuffle(groups)
        count = (min(len(groups) - 1, max(1, round(len(groups) * config["validation_fraction"])))
                 if config["validation_fraction"] > 0 and len(groups) > 1 else 0)
        holdout = set(groups[:count])
        for pair in pairs:
            group = group_key(pair)
            (validation_pairs if group in holdout else train_pairs).append(pair)
    if not train_pairs:
        raise ValueError("Preference training split is empty")
    def build(rows, split, repeats):
        for row in rows:
            row["split"] = split
        manifest = {**document, "pairs": rows}
        return PreferenceTensorDataset(
            manifest, split=split, seed=config["seed"], repeats=repeats,
            max_latent_length=config["max_latent_length"] or None,
            verify_checksums=config["verify_checksums"])
    training = build(train_pairs, "train", config["dataset_repeats"])
    validation = build(validation_pairs, "validation", 1) if validation_pairs else None
    return training, validation


def _optimizer(parameters: list, config: dict, device):
    import torch
    kwargs = {"lr": config["learning_rate"], "weight_decay": config["weight_decay"],
              "betas": (config["adam_beta1"], config["adam_beta2"]),
              "eps": config["adam_epsilon"]}
    if config["optimizer"] == "adamw8bit":
        if device.type != "cuda":
            raise ValueError("AdamW8bit requires a CUDA device")
        try:
            from bitsandbytes.optim import AdamW8bit
        except ImportError as exc:
            raise RuntimeError("Install bitsandbytes to select AdamW8bit, or use adamw") from exc
        return AdamW8bit(parameters, **kwargs)
    if config["optimizer"] == "adafactor":
        from transformers import Adafactor
        return Adafactor(parameters, lr=config["learning_rate"],
                         weight_decay=config["weight_decay"], relative_step=False,
                         scale_parameter=False, warmup_init=False)
    return torch.optim.AdamW(parameters, **kwargs)


def _scheduler(optimizer, config: dict, total_steps: int):
    from torch.optim.lr_scheduler import LambdaLR
    warmup = min(config["warmup_steps"], max(0, total_steps - 1))
    def factor(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        if config["scheduler"] == "constant":
            return 1.0
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup)))
        decay = 1.0 - progress if config["scheduler"] == "linear" else 0.5 * (1 + math.cos(math.pi * progress))
        return config["min_lr_ratio"] + (1.0 - config["min_lr_ratio"]) * decay
    return LambdaLR(optimizer, factor)


def _checkpoint_save(path: Path, decoder: Any, config: dict, optimizer: Any,
                     scheduler: Any, scaler: Any, runtime: dict, device,
                     *, rng: dict | None = None, rank_rng: list | None = None) -> Path:
    import torch
    if path.exists():
        path = path.with_name(path.name + f"-{runtime['global_step']}-{uuid.uuid4().hex[:6]}")
    staging = path.with_name("." + path.name + ".writing-" + uuid.uuid4().hex[:8])
    staging.mkdir(parents=True)
    decoder.save_pretrained(str(staging), safe_serialization=True)
    adapter_config = staging / "adapter_config.json"
    portable = json.loads(adapter_config.read_text(encoding="utf-8"))
    portable["base_model_name_or_path"] = ""
    _json_write(adapter_config, portable)
    state = {"format": "JK-Step MR-FlowDPO", "version": 1,
             "runtime": runtime, "config": config,
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "scaler": scaler.state_dict() if scaler else {},
             "rng": rng if rng is not None else _rng_capture(device)}
    if rank_rng is not None:
        state["rank_rng"] = rank_rng
    torch.save(state, staging / "training_state.pt")
    _json_write(staging / "training_config.json", config)
    staging.rename(path)
    return path


def _base_fingerprint(config: dict) -> str:
    """Detect changed local weights/config on resume without reading 20 GB.

    File size/mtime and configuration bytes are a change detector, not a
    cryptographic integrity assertion about all checkpoint tensor contents.
    """
    if config["checkpoint_file"]:
        sources = [Path(config["checkpoint_file"])]
        architecture = Path(config["model_config_dir"])
    else:
        mapping = {"xl-sft": "acestep-v15-xl-sft", "sft": "acestep-v15-sft",
                   "xl-base": "acestep-v15-xl-base", "base": "acestep-v15-base"}
        root = Path(config["checkpoint_dir"])
        architecture = root / mapping[config["model_variant"]]
        if not architecture.is_dir():
            architecture = root / config["model_variant"]
        sources = sorted(architecture.glob("*.safetensors")) + sorted(architecture.glob("*.bin"))
    digest = hashlib.sha256()
    for path in sources:
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    for path in sorted(architecture.glob("*.py")) + [architecture / "config.json"]:
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _prune_checkpoints(output: Path, limit: int) -> None:
    if limit <= 0:
        return
    candidates = sorted((path for path in output.iterdir()
                         if path.is_dir() and re.fullmatch(r"checkpoint-\d+", path.name)
                         and (path / "training_state.pt").is_file()), key=lambda p: int(p.name.split("-")[1]))
    for path in candidates[:-limit]:
        if path.resolve().parent != output.resolve():
            raise RuntimeError("Checkpoint retention target is outside the run directory")
        shutil.rmtree(path)


def export_adapter(adapter_dir: str | Path, output_path: str | Path | None = None,
                   *, target: str = "native", decoder: Any = None) -> dict:
    """ComfyUI export preserving PEFT's actual per-module/rsLoRA scaling."""
    import torch
    from safetensors.torch import load_file, save_file
    from jk_engine.core.comfyui_export import export_for_comfyui, resolve_target
    from peft import LoraConfig
    adapter_dir = Path(adapter_dir)
    result = export_for_comfyui(adapter_dir, str(output_path) if output_path else None,
                               target=target, verbose=False)
    if not result.get("ok"):
        raise RuntimeError(result.get("message", "ComfyUI export failed"))
    path = result.get("output_path")
    if not path:
        return result
    weights = load_file(path)
    prefix = resolve_target(target)
    if decoder is not None:
        scaling = {}
        for name, module in decoder.named_modules():
            if hasattr(module, "lora_A") and "default" in module.lora_A:
                raw = name.removeprefix("base_model.model.")
                rank = module.lora_A["default"].weight.shape[0]
                scaling[f"{prefix}.{raw}.alpha"] = float(module.scaling["default"]) * rank
        for key, alpha in scaling.items():
            if key in weights:
                weights[key] = torch.tensor(alpha, dtype=torch.float32)
    else:
        # No full model needed for standalone export.  PEFT's regex pattern
        # resolution and the rank read directly from each A matrix suffice.
        cfg = LoraConfig.from_pretrained(str(adapter_dir))
        for key, down in list(weights.items()):
            if not key.endswith(".lora_down.weight"):
                continue
            module = key[:-len(".lora_down.weight")].removeprefix(prefix + ".")
            matches = [pattern for pattern in cfg.alpha_pattern if re.match(rf"(.*\.)?({pattern})$", module)]
            alpha = cfg.alpha_pattern[matches[0]] if matches else cfg.lora_alpha
            rank = down.shape[0]
            effective_alpha = float(alpha) * (math.sqrt(rank) if cfg.use_rslora else 1.0)
            weights[key[:-len(".lora_down.weight")] + ".alpha"] = torch.tensor(effective_alpha)
    save_file(weights, path, metadata={"trainer": "JK-Step MR-FlowDPO",
                                      "scaling": "PEFT effective scaling preserved"})
    return result


def _evaluate(model, loader, config, device, distributed=None) -> dict:
    import torch
    saved = _rng_capture(device)
    rank = distributed.rank if distributed else 0
    eval_seed = config["seed"] + 1000003 + rank * 10000019
    random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(eval_seed)
    objective = config.get("objective", "flow_dpo")
    keys = ("validation_loss", "validation_chosen_fm") if objective == "sft" else (
        "validation_loss", "validation_dpo", "validation_preference_accuracy", "validation_chosen_fm")
    totals = {key:0.0 for key in keys}
    count, weight_mass = 0.0, 0.0
    try:
        with torch.no_grad():
            for index, batch in enumerate(loader):
                if config["eval_batches"] and index >= config["eval_batches"]:
                    break
                result = _training_step(model, batch, config, training=False)
                size = batch["target_latents" if objective == "sft" else "chosen_latents"].shape[0]
                mass = (float(batch["pair_weight"].sum())
                        if objective == "flow_dpo" and config["pair_weighting"] and "pair_weight" in batch else size)
                totals["validation_loss"] += float(result.loss) * mass
                if objective == "flow_dpo":
                    totals["validation_dpo"] += float(result.dpo_loss) * mass
                    totals["validation_preference_accuracy"] += float((result.preference_logits > 0).float().mean()) * size
                totals["validation_chosen_fm"] += float(result.chosen_fm.mean()) * size
                count += size
                weight_mass += mass
    finally:
        _rng_restore(saved, device)
        _policy_mode(model.decoder, True)
    if distributed:
        reduced = distributed.sum_dict({**totals,"count":count,"weight_mass":weight_mass})
        count = reduced.pop("count")
        weight_mass = reduced.pop("weight_mass")
        totals = reduced
    return {key: value / (weight_mass if key in ("validation_loss","validation_dpo") else count)
            for key, value in totals.items()} if count and weight_mass else {}


def resolve_training_model(config: dict, progress=None) -> dict:
    """Resolve/download the default XL before ranks wait at a collective.

    This imports no ML runtime and preserves an explicitly selected local base,
    checkpoint file, or architecture directory. Complete local HF weights are
    reused, including sharded checkpoints.
    """
    config = dict(config)
    if not config.get("auto_download",True) or config.get("checkpoint_file") or config.get("model_variant","xl-sft") != "xl-sft":
        return config
    from .models import ensure_models, hf_weights_ready
    root = Path(config.get("checkpoint_dir","checkpoints"))
    candidates = (root / "acestep-v15-xl-sft", root / "xl-sft")
    complete = [folder for folder in candidates if hf_weights_ready(folder)]
    single_file = next((folder / "acestep_v1.5_sft_xl.safetensors" for folder in complete
                        if (folder / "acestep_v1.5_sft_xl.safetensors").is_file()),None)
    if single_file:
        config["checkpoint_file"] = str(single_file.resolve())
        if not config.get("model_config_dir"):
            config["model_config_dir"] = str(single_file.parent.resolve())
    elif not complete:
        downloaded = ensure_models(checkpoint_dir=str(root),include_preprocess=False,progress=progress)
        for key in ("checkpoint_file","model_config_dir","checkpoint_dir","model_variant"):
            if key in downloaded and not (key=="model_config_dir" and config.get(key)):
                config[key] = downloaded[key]
    return config


def run_training(config: dict, *, progress: Callable[[dict], None] | None = None,
                 stop_event: Any = None) -> dict:
    """Public worker engine; launch_training selects devices/processes for UI/CLI."""
    import torch
    from .distributed import initialize_distributed
    config = validate_config(config,check_paths=False)
    requested_device = config["device"]
    world = int(os.environ.get("WORLD_SIZE","1"))
    if config.get("multi_gpu") and world <= 1:
        raise ValueError("Use launch_training for multi_gpu, or start rank workers explicitly")
    if world > 1 and str(requested_device).split(":")[0] in ("auto","cuda") and torch.cuda.is_available():
        config["device"] = f"cuda:{int(os.environ.get('LOCAL_RANK','0'))}"
    device, _, _ = _hardware(config)
    distributed = initialize_distributed(config,device)
    try:
        return _run_training_worker(config,progress=progress,stop_event=stop_event,
                                    distributed=distributed,requested_device=requested_device)
    finally:
        distributed.close()


def _run_training_worker(config: dict, *, progress=None, stop_event=None,
                         distributed, requested_device: str) -> dict:
    """Train and save a real LoRA; CLI and UI call exactly the same engine.

    ``progress`` receives JSON-compatible records.  Cancellation is honored at
    optimizer-update boundaries and saves a resumable ``stopped`` checkpoint.
    No audio dataset is generated with Turbo as part of this function.
    """
    import torch
    from torch.utils.data import DataLoader
    from jk_step.preprocess import collate_pairs
    config = validate_config(config, check_paths=False)
    objective = config["objective"]
    data_key = "dataset_manifest" if objective == "sft" else "pairs_manifest"
    data_path = Path(config[data_key])
    if not data_path.is_file() and not (objective == "sft" and data_path.is_dir()):
        raise FileNotFoundError(f"Training dataset not found: {data_path}")
    if distributed.is_main:
        config = resolve_training_model(config,progress)
    if distributed.active:
        config = distributed.broadcast_object(config if distributed.is_main else None)
        config["device"] = str(distributed.device)
    config = validate_config(config, check_paths=True)
    device, precision, dtype = _hardware(config)
    config["device"], config["precision"] = str(device), precision
    portable_config = {**config,"device":requested_device if distributed.active else str(device)}
    output = Path(config["output_dir"]).expanduser().resolve()
    exists = (output / "training_config.json").exists() if distributed.is_main else None
    exists = distributed.broadcast_object(exists)
    if exists and not config["resume_from"]:
        raise FileExistsError("This output_dir already contains a training run. Choose a new directory or resume_from.")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed(config["seed"])
    train_data, validation_data = _split_datasets(config)
    collate = collate_pairs
    if objective == "sft":
        from jk_step.supervised_data import collate_supervised
        collate = collate_supervised
    loader_options = dict(batch_size=config["batch_size"], collate_fn=collate,
                          num_workers=config["num_workers"],
                          pin_memory=config["pin_memory"] and device.type == "cuda")
    train_sampler = None
    if distributed.active:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_data,num_replicas=distributed.world_size,
            rank=distributed.rank,shuffle=True,seed=config["seed"],drop_last=False)
    local_samples = len(train_sampler) if train_sampler is not None else len(train_data)
    batches_per_epoch = math.ceil(local_samples / config["batch_size"])
    steps_per_epoch = math.ceil(batches_per_epoch / config["gradient_accumulation"])
    total_steps = config["max_steps"] or config["epochs"] * steps_per_epoch
    total_epochs = math.ceil(total_steps / steps_per_epoch) if config["max_steps"] else config["epochs"]
    if objective == "sft":
        from jk_step.supervised_data import supervised_manifest_fingerprint
        fingerprint = supervised_manifest_fingerprint(config["dataset_manifest"])
    else:
        fingerprint = hashlib.sha256(Path(config["pairs_manifest"]).read_bytes()).hexdigest()
    log_file = output / "metrics.jsonl"
    writer = None
    def emit(event: str, **values):
        if not distributed.is_main:
            return
        record = {"event": event, "time": time.time(), "objective": objective, **values}
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        if progress:
            progress(record)
        if writer and "step" in record:
            for key, value in record.items():
                if key not in ("step", "time") and isinstance(value, (float, int)):
                    writer.add_scalar(key, value, record["step"])
    emit("loading", message="Loading original ACE-Step checkpoint", device=str(device), precision=precision,
         world_size=distributed.world_size,distributed_backend=distributed.backend)
    if config["tensorboard"] and distributed.is_main:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError("Install tensorboard, or set tensorboard=false") from exc
        writer = SummaryWriter(str(output / "tensorboard"))
    model = None
    try:
        model = load_base_model(config)
        if getattr(model.config, "is_turbo", False):
            raise ValueError("Distilled Turbo is not supported by this SFT preference trainer")
        model, parameters = _inject_adapter(model, config)
        distributed.synchronize_parameters(parameters)
        if distributed.active:
            rank_seed = config["seed"] + distributed.rank * 10000019
            random.seed(rank_seed)
            torch.manual_seed(rank_seed)
            if device.type=="cuda":
                torch.cuda.manual_seed(rank_seed)
        optimizer = _optimizer(parameters, config, device)
        scheduler = _scheduler(optimizer, config, total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
        runtime = {"global_step": 0, "epoch": 0, "batch_cursor": 0,
                   "best_validation_loss": None, "manifest_sha256": fingerprint,
                   "base_fingerprint": _base_fingerprint(config),
                   "world_size":distributed.world_size,
                   "total_steps": total_steps, "objective": objective}
        if config["resume_from"]:
            state_path = Path(config["resume_from"]) / "training_state.pt"
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            if state.get("format") != "JK-Step MR-FlowDPO" or state.get("version") != 1:
                raise ValueError("resume_from is not a supported JK-Step MR-FlowDPO checkpoint")
            if state["config"].get("objective", "flow_dpo") != objective:
                raise ValueError("Resume objective changed; use init_adapter for a new training setup")
            if state["runtime"]["manifest_sha256"] != fingerprint:
                raise ValueError("The training dataset changed since the checkpoint; start a new run with init_adapter")
            if state["runtime"].get("base_fingerprint") != runtime["base_fingerprint"]:
                raise ValueError("Base checkpoint/configuration changed since the checkpoint; restore the original base or use init_adapter")
            if state['runtime'].get('world_size',1) != distributed.world_size:
                raise ValueError("Resume world_size changed; use the same GPU count or init_adapter for a new run")
            invariants = ("rank", "alpha", "dropout", "target_modules", "attention_type", "target_mlp", "layers",
                          "rank_pattern", "alpha_pattern", "use_rslora", "beta", "fm_regularization",
                          "reference_regularization", "label_smoothing", "pair_weighting", "timestep_sampling",
                          "timestep_mu", "timestep_sigma", "timestep_min", "timestep_max", "cfg_dropout",
                          "batch_size", "gradient_accumulation", "max_latent_length", "dataset_repeats",
                          "optimizer", "scheduler", "warmup_steps", "learning_rate", "min_lr_ratio",
                          "weight_decay", "adam_beta1", "adam_beta2", "adam_epsilon", "max_grad_norm",
                          "seed", "validation_fraction", "precision", "device",
                          "checkpoint_file", "model_config_dir", "checkpoint_dir", "model_variant",
                          "multi_gpu","gpu_ids","distributed_backend", "objective")
            if objective == "sft":
                ignored = {"beta", "fm_regularization", "reference_regularization", "label_smoothing", "pair_weighting"}
                invariants = tuple(key for key in invariants if key not in ignored) + ("dataset_manifest",)
            changed = [key for key in invariants if state["config"].get(key,portable_config.get(key)) != portable_config.get(key)]
            if changed:
                raise ValueError("Resume configuration changed: " + ", ".join(changed) + ". Use init_adapter for a new training setup.")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            if state.get("scaler"):
                scaler.load_state_dict(state["scaler"])
            runtime.update(state["runtime"])
            if runtime["global_step"] > total_steps:
                raise ValueError("max_steps/epochs would end before the resumed checkpoint's optimizer step")
            runtime["total_steps"] = total_steps
            states = state.get("rank_rng")
            if distributed.active and (not states or len(states)!=distributed.world_size):
                raise ValueError("Distributed checkpoint is missing per-rank RNG states")
            _rng_restore(states[distributed.rank] if states else state["rng"], device)
        if distributed.is_main:
            _json_write(output / "training_config.json", portable_config)
        counts = ({"train_samples": len(train_data), "validation_samples": len(validation_data) if validation_data else 0,
                   "local_train_samples": local_samples} if objective == "sft" else {
                   "train_pairs": len(train_data), "validation_pairs": len(validation_data) if validation_data else 0,
                   "local_train_pairs": local_samples})
        emit("started", step=runtime["global_step"], total_steps=total_steps, **counts,
             trainable_parameters=sum(p.numel() for p in parameters),
             message="SFT single-target flow matching" if objective == "sft" else "MR-FlowDPO preference training",
             world_size=distributed.world_size,
             effective_batch_size=config['batch_size']*config['gradient_accumulation']*distributed.world_size,
             distributed_backend=distributed.backend)
        if validation_data and hasattr(validation_data, "set_epoch"):
            validation_data.set_epoch(0)
        from .distributed import EvaluationShardSampler
        validation_sampler = EvaluationShardSampler(validation_data,distributed.rank,distributed.world_size) if validation_data and distributed.active else None
        validation_loader = DataLoader(validation_data, shuffle=False,sampler=validation_sampler,
                                       generator=torch.Generator().manual_seed(config["seed"] + 1000003),
                                       **loader_options) if validation_data else None
        commit_rng = _rng_capture(device)
        stopped = False
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        def save_checkpoint(destination):
            states = distributed.gather_object(commit_rng)
            path = None
            if distributed.is_main:
                path = _checkpoint_save(destination,model.decoder,portable_config,optimizer,scheduler,
                    scaler,dict(runtime),device,rng=states[0],rank_rng=states if distributed.active else None)
            return Path(distributed.broadcast_object(str(path) if path else None))
        try:
            for epoch in range(runtime["epoch"], total_epochs):
                if hasattr(train_data, "set_epoch"):
                    train_data.set_epoch(epoch)
                generator = torch.Generator().manual_seed(config["seed"] + epoch)
                # DataLoader bookkeeping consumes only this separate generator,
                # preserving restored diffusion RNG on exact mid-epoch resume.
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                loader = DataLoader(train_data, shuffle=train_sampler is None,sampler=train_sampler,
                                    generator=generator, **loader_options)
                cursor = runtime["batch_cursor"] if epoch == runtime["epoch"] else 0
                iterator = iter(loader)
                for _ in range(cursor):
                    next(iterator)
                batch_index = cursor
                while batch_index < batches_per_epoch and runtime["global_step"] < total_steps:
                    if distributed.any(stop_event is not None and stop_event.is_set()):
                        stopped = True
                        break
                    window = min(config["gradient_accumulation"], batches_per_epoch - batch_index)
                    microbatches = [next(iterator) for _ in range(window)]
                    masses = [float(batch["pair_weight"].sum())
                              if objective == "flow_dpo" and config["pair_weighting"] and "pair_weight" in batch
                              else float(batch["target_latents" if objective == "sft" else "chosen_latents"].shape[0])
                              for batch in microbatches]
                    if distributed.any(any(not math.isfinite(mass) or mass <= 0 for mass in masses)):
                        raise ValueError("Every microbatch must have positive finite example weight")
                    total_mass = distributed.sum(sum(masses))
                    sums = defaultdict(float)
                    for batch, mass in zip(microbatches, masses):
                        result = _training_step(model, batch, config, training=True)
                        if distributed.any(not bool(torch.isfinite(result.loss))):
                            raise FloatingPointError("Non-finite training loss; inspect cached tensors and learning rate")
                        scaler.scale(result.loss * (mass / total_mass)).backward()
                        for key, value in _step_metrics(result, objective).items():
                            sums[key] += float(value.detach()) * (mass / total_mass)
                        batch_index += 1
                    scaler.unscale_(optimizer)
                    distributed.synchronize_gradients(parameters)
                    sums = distributed.sum_dict(dict(sums))
                    if config["max_grad_norm"]:
                        norm = torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"], error_if_nonfinite=True)
                        sums["grad_norm"] = float(norm)
                    else:
                        if distributed.any(any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters)):
                            raise FloatingPointError("Non-finite adapter gradients")
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    runtime.update(global_step=runtime["global_step"] + 1, epoch=epoch,
                                   batch_cursor=batch_index)
                    if batch_index == batches_per_epoch:
                        runtime.update(epoch=epoch + 1, batch_cursor=0)
                    commit_rng = _rng_capture(device)
                    step = runtime["global_step"]
                    if step % config["log_every"] == 0 or step == total_steps:
                        emit("training", step=step, total_steps=total_steps, epoch=epoch + 1,
                             learning_rate=float(optimizer.param_groups[0]["lr"]),
                             elapsed_seconds=time.perf_counter() - started, **sums)
                    if validation_loader is not None and config["eval_every"] and step % config["eval_every"] == 0:
                        metrics = _evaluate(model, validation_loader, config, device,distributed)
                        emit("validation", step=step, **metrics)
                        score = metrics.get("validation_loss")
                        if score is not None and (runtime["best_validation_loss"] is None or score < runtime["best_validation_loss"]):
                            runtime["best_validation_loss"] = score
                            best = save_checkpoint(output / f"best-{step:08d}")
                            if distributed.is_main:
                                _json_write(output / "best.json", {"step": step, "adapter_dir": str(best), **metrics})
                    if config["save_every"] and step % config["save_every"] == 0:
                        path = save_checkpoint(output / f"checkpoint-{step:08d}")
                        emit("checkpoint", step=step, adapter_dir=str(path))
                        if distributed.is_main:
                            _prune_checkpoints(output, config["save_total_limit"])
                if stopped or runtime["global_step"] >= total_steps:
                    break
        except KeyboardInterrupt:
            if distributed.active:
                # A rank-local interrupt cannot enter checkpoint collectives
                # while peers are still inside an optimizer window. The parent
                # launcher terminates peers on rank failure; UI cancellation
                # uses the shared stop file and a synchronized boundary.
                raise
            stopped = True
            optimizer.zero_grad(set_to_none=True)
            _rng_restore(commit_rng, device)
        # Do not conflate the held-out velocity objective with an audible
        # improvement: these diagnostics do not measure sung-word correctness.
        if validation_loader is not None and not stopped:
            emit("validation", step=runtime["global_step"], **_evaluate(model, validation_loader, config, device,distributed))
        destination = output / ("stopped" if stopped else "final")
        saved = save_checkpoint(destination)
        result = {"status": "stopped" if stopped else "complete", "step": runtime["global_step"],
                  "total_steps": total_steps, "adapter_dir": str(saved),
                  "output_dir": str(output), "metrics_path": str(log_file),
                  "world_size":distributed.world_size,"distributed_backend":distributed.backend,
                  "objective": objective}
        if config["export_comfyui"] and distributed.is_main:
            export = export_adapter(saved, output / ("stopped_comfyui.safetensors" if stopped else "quality_comfyui.safetensors"),
                                    target=config["comfyui_target"], decoder=model.decoder)
            result["comfyui_path"] = export["output_path"]
        result = distributed.broadcast_object(result if distributed.is_main else None)
        if distributed.is_main:
            _json_write(output / "latest.json", result)
        emit(result["status"], **{key: value for key, value in result.items() if key != "status"})
        return result
    finally:
        if writer:
            writer.close()
        if model is not None:
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
