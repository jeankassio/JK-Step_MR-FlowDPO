"""Configuration shared by the JK-Step command line and local UI.

This module deliberately has no machine-learning imports.  Asking for help or
editing a configuration must work before PyTorch and the models are installed.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any


def _field(name, default, kind, group, help_text, **extra):
    return {"name": name, "label": name.replace("_", " ").capitalize(),
            "default": default, "type": kind, "group": group,
            "help": help_text, **extra}


_FIELDS = [
    _field("objective", "flow_dpo", "string", "Training", "flow_dpo learns preferences; sft learns directly from single audio targets with flow matching. SFT ignores all preference-only coefficients.", choices=["flow_dpo", "sft"]),
    _field("auto_download", True, "bool", "Model", "Download the default SFT XL checkpoint when no local model was selected; uses the trainer's own checkpoint_dir."),
    _field("checkpoint_dir", "checkpoints", "string", "Model", "Root of the Hugging Face ACE-Step checkpoint directories."),
    _field("checkpoint_file", "", "string", "Model", "Optional original/ComfyUI .safetensors file; needs model_config_dir with matching architecture."),
    _field("model_config_dir", "", "string", "Model", "Local config.json and Python architecture files for checkpoint_file. No model weights need downloading here."),
    _field("model_variant", "xl-sft", "string", "Model", "SFT/base architecture. Preference training of a distilled Turbo is not enabled.", choices=["xl-sft", "sft", "xl-base", "base"]),
    _field("pairs_manifest", "", "string", "Data", "Prepared preference manifest, with tensor_path for every selected pair."),
    _field("dataset_manifest", "", "string", "Data", "SFT only: prepared supervised JSON, a Side-Step manifest.json, or a directory of .pt tensors. No chosen/rejected pairs are needed."),
    _field("output_dir", "output/quality-lora", "string", "Data", "Run directory; checkpoints, logs, configuration and final adapter are written here."),
    _field("max_latent_length", 750, "int", "Data", "Flow-DPO crop in latent frames; 0 preserves prepared clips. SFT defaults to 0 and rejects cropping vocals without aligned clip lyrics.", min=0, step=50),
    _field("dataset_repeats", 1, "int", "Data", "Repeat the training data within each epoch.", min=1),
    _field("verify_checksums", False, "bool", "Data", "Verify cached tensor fingerprints when supported by the dataset."),
    _field("rank", 32, "int", "LoRA", "LoRA rank, for example 16, 32, 64 or 128.", min=1, max=1024),
    _field("alpha", 64.0, "float", "LoRA", "LoRA scaling alpha; inference exports preserve alpha/rank.", min=0.001, step=1),
    _field("dropout", 0.0, "float", "LoRA", "Adapter dropout; Flow-DPO shares masks across branches, SFT uses ordinary dropout. Base-model dropout stays disabled.", min=0, max=0.95, step=0.01),
    _field("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"], "list", "LoRA", "Linear decoder module suffixes; comma-separated or a JSON list."),
    _field("attention_type", "both", "string", "LoRA", "Attention projections to adapt.", choices=["both", "self", "cross"]),
    _field("target_mlp", False, "bool", "LoRA", "Also adapt gate_proj/up_proj/down_proj in the decoder MLPs."),
    _field("layers", "all", "string", "LoRA", "Decoder layer selection: all, 0,1,2, or 0-7,16-23."),
    _field("rank_pattern", {}, "json", "LoRA", "Optional PEFT per-module rank overrides (JSON object)."),
    _field("alpha_pattern", {}, "json", "LoRA", "Optional PEFT per-module alpha overrides (JSON object)."),
    _field("use_rslora", False, "bool", "LoRA", "Rank-stabilized LoRA; export embeds its effective scaling."),
    _field("beta", 2000.0, "float", "MR-FlowDPO", "Preference-loss scale. 2000 is the published MelodyFlow setup, not a calibrated ACE-Step optimum.", min=0.000001, step=10),
    _field("fm_regularization", 0.0, "float", "MR-FlowDPO", "Optional flow-matching error on chosen samples to anchor generation; 0 reproduces the core DPO objective.", min=0, step=0.01),
    _field("reference_regularization", 0.0, "float", "MR-FlowDPO", "Experimental velocity penalty versus the frozen SFT on both branches; does not guarantee diction preservation.", min=0, step=0.01),
    _field("label_smoothing", 0.0, "float", "MR-FlowDPO", "Preference-label smoothing for uncertain pairs (0 to 0.49).", min=0, max=0.49, step=0.01),
    _field("pair_weighting", True, "bool", "MR-FlowDPO", "Use pair_weight values in the prepared dataset instead of uniform weights."),
    _field("timestep_sampling", "logit_normal", "string", "MR-FlowDPO", "ACE's continuous max-of-two logit-normal sampling or the paper's uniform sampling.", choices=["logit_normal", "uniform"]),
    _field("timestep_mu", -0.4, "float", "MR-FlowDPO", "Location of the ACE logit-normal timestep distribution."),
    _field("timestep_sigma", 1.0, "float", "MR-FlowDPO", "Scale of the ACE logit-normal timestep distribution.", min=0.000001),
    _field("timestep_min", 0.0001, "float", "MR-FlowDPO", "Lower endpoint clamp for sampled timesteps.", min=0, max=0.999),
    _field("timestep_max", 0.9999, "float", "MR-FlowDPO", "Upper endpoint clamp for sampled timesteps.", min=0.001, max=1),
    _field("cfg_dropout", 0.0, "float", "MR-FlowDPO", "Shared condition dropout for both branches. 0 preserves lyrics conditioning on every pair.", min=0, max=1, step=0.01),
    _field("batch_size", 1, "int", "Training", "Examples per microbatch per GPU: one audio branch in SFT, two in Flow-DPO.", min=1),
    _field("gradient_accumulation", 4, "int", "Training", "Microbatches per optimizer update.", min=1),
    _field("epochs", 10, "int", "Training", "Epoch count when max_steps is 0.", min=1),
    _field("max_steps", 0, "int", "Training", "Positive optimizer-step limit overrides epochs; 0 uses epochs.", min=0),
    _field("learning_rate", 0.00001, "float", "Optimizer", "Adapter optimizer learning rate.", min=0.000000001, step=0.000001),
    _field("optimizer", "adamw", "string", "Optimizer", "AdamW, optional CUDA bitsandbytes AdamW8bit, or Adafactor.", choices=["adamw", "adamw8bit", "adafactor"]),
    _field("scheduler", "cosine", "string", "Optimizer", "Learning-rate schedule after warmup.", choices=["constant", "linear", "cosine"]),
    _field("warmup_steps", 100, "int", "Optimizer", "Optimizer updates used for learning-rate warmup.", min=0),
    _field("min_lr_ratio", 0.1, "float", "Optimizer", "Final learning rate relative to the initial value for linear/cosine.", min=0, max=1, step=0.01),
    _field("weight_decay", 0.01, "float", "Optimizer", "Decoupled weight decay.", min=0, step=0.001),
    _field("adam_beta1", 0.9, "float", "Optimizer", "First-moment decay for AdamW.", min=0, max=0.9999),
    _field("adam_beta2", 0.999, "float", "Optimizer", "Second-moment decay for AdamW.", min=0, max=0.99999),
    _field("adam_epsilon", 0.00000001, "float", "Optimizer", "AdamW denominator epsilon.", min=0.000000000001),
    _field("max_grad_norm", 1.0, "float", "Optimizer", "Gradient clipping; 0 disables clipping.", min=0, step=0.1),
    _field("precision", "auto", "string", "Hardware", "Base-model/forward precision; trainable adapter weights and loss reduction remain FP32.", choices=["auto", "bf16", "fp16", "fp32"]),
    _field("device", "auto", "string", "Hardware", "Single-device mode: auto, cpu, cuda:0, cuda:1, mps or xpu:0. Multi-GPU mode uses gpu_ids instead."),
    _field("multi_gpu", False, "bool", "Hardware", "Train synchronized LoRA replicas on multiple CUDA GPUs. Each GPU keeps its own frozen decoder; VRAM is not pooled."),
    _field("gpu_ids", ["0", "1"], "list", "Hardware", "CUDA GPU indices for multi-GPU training, e.g. 0,1. Batch size and gradient accumulation apply per GPU."),
    _field("distributed_backend", "auto", "string", "Hardware", "Communication backend: auto uses local CPU IPC on Windows and NCCL on Linux when available. local synchronizes LoRA gradients between processes without requiring Gloo/NCCL.", choices=["auto", "local", "gloo", "nccl"]),
    _field("offload_non_decoder", True, "bool", "Hardware", "Keep condition encoders/tokenizers on CPU after tensors have been prepared; decoder stays on the selected device."),
    _field("gradient_checkpointing", True, "bool", "Hardware", "Recompute decoder activations during backward to reduce VRAM."),
    _field("seed", 42, "int", "Hardware", "Training, split and shuffle seed.", min=0, max=2147483647),
    _field("num_workers", 0, "int", "Hardware", "DataLoader worker processes; 0 is reliable on Windows.", min=0, max=64),
    _field("pin_memory", True, "bool", "Hardware", "Pinned CPU data transfer when training on CUDA."),
    _field("validation_fraction", 0.1, "float", "Validation", "Fallback group-level holdout when no explicit validation split exists. Never splits variants of the same recording.", min=0, max=0.9, step=0.01),
    _field("eval_every", 100, "int", "Validation", "Optimizer updates between deterministic validation runs; 0 disables periodic evaluation.", min=0),
    _field("eval_batches", 20, "int", "Validation", "Maximum validation microbatches per evaluation; 0 evaluates the complete validation set.", min=0),
    _field("save_every", 100, "int", "Logging", "Optimizer updates between resumable checkpoints; 0 saves only at the end.", min=0),
    _field("save_total_limit", 3, "int", "Logging", "Periodic checkpoints retained; 0 keeps all. final/best/stopped are not deleted.", min=0),
    _field("log_every", 1, "int", "Logging", "Optimizer updates between JSONL/progress records.", min=1),
    _field("tensorboard", True, "bool", "Logging", "Write TensorBoard scalars; requires tensorboard when enabled."),
    _field("resume_from", "", "string", "Resume/export", "JK-Step checkpoint directory containing training_state.pt; restores optimizer, scheduler, RNG and batch cursor."),
    _field("init_adapter", "", "string", "Resume/export", "Optional PEFT adapter used to initialize policy. Reference remains the original SFT; mutually exclusive with resume_from."),
    _field("export_comfyui", True, "bool", "Resume/export", "Export final PEFT adapter as a ComfyUI .safetensors LoRA."),
    _field("comfyui_target", "native", "string", "Resume/export", "Side-Step ComfyUI key mapping preset.", choices=["native", "generic"]),
]


def schema_fields() -> list[dict[str, Any]]:
    return copy.deepcopy(_FIELDS)


def schema_defaults() -> dict[str, Any]:
    return {field["name"]: copy.deepcopy(field["default"]) for field in _FIELDS}


def parse_layers(value: str | list[int]) -> list[int] | None:
    if isinstance(value, list):
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in value):
            raise ValueError("layers must contain nonnegative integer indices")
        result = sorted(set(value))
    else:
        value = str(value).strip().lower()
        if value in ("", "all"):
            return None
        result = []
        for component in value.split(","):
            parts = component.strip().split("-")
            if len(parts) == 1 and parts[0].isdigit():
                result.append(int(parts[0]))
            elif len(parts) == 2 and all(p.isdigit() for p in parts):
                start, end = map(int, parts)
                if end < start or end > 10000:
                    raise ValueError("layers ranges must be ascending and below 10001")
                result.extend(range(start, end + 1))
            else:
                raise ValueError("layers must be 'all' or indices/ranges, e.g. 0-7,16-23")
        result = sorted(set(result))
    if not result:
        raise ValueError("layers selection cannot be empty")
    return result


def validate_config(config: dict[str, Any], *, check_paths: bool = False) -> dict[str, Any]:
    """Return normalized complete configuration, rejecting misspelled options."""
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a JSON object")
    defaults = schema_defaults()
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError("Unknown training options: " + ", ".join(sorted(unknown)))
    result = defaults | copy.deepcopy(config)
    if result["objective"] == "sft" and "max_latent_length" not in config:
        result["max_latent_length"] = 0
    for spec in _FIELDS:
        key, kind = spec["name"], spec["type"]
        value = result[key]
        try:
            if kind == "bool":
                if isinstance(value, str) and value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                if not isinstance(value, bool):
                    raise ValueError("must be true or false")
            elif kind in ("float", "int"):
                if isinstance(value, bool):
                    raise ValueError("must be a number, not a boolean")
                number = float(value)
                if not math.isfinite(number) or (kind == "int" and number != int(number)):
                    raise ValueError("must be finite" + (" and integral" if kind == "int" else ""))
                value = int(number) if kind == "int" else number
                if "min" in spec and value < spec["min"]:
                    raise ValueError(f"must be >= {spec['min']}")
                if "max" in spec and value > spec["max"]:
                    raise ValueError(f"must be <= {spec['max']}")
            elif kind == "json":
                value = json.loads(value) if isinstance(value, str) else value
                if not isinstance(value, dict):
                    raise ValueError("must be a JSON object")
                for name, override in value.items():
                    if not isinstance(name, str) or not name:
                        raise ValueError("override keys must be nonempty module names")
                    if isinstance(override, bool) or not math.isfinite(float(override)) or float(override) <= 0:
                        raise ValueError("overrides must be positive finite numbers")
                    if key == "rank_pattern" and (float(override) != int(float(override)) or int(override) > 1024):
                        raise ValueError("rank overrides must be integers between 1 and 1024")
                    value[name] = int(override) if key == "rank_pattern" else float(override)
            elif kind == "list":
                if isinstance(value, str):
                    value = json.loads(value) if value.strip().startswith("[") else value.replace(",", " ").split()
                if key == "gpu_ids" and isinstance(value, list):
                    value = [str(x) if isinstance(x, int) and not isinstance(x, bool) else x for x in value]
                if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x.strip() for x in value):
                    raise ValueError("must be a nonempty list of strings")
            elif key == "layers" and isinstance(value, list):
                value = ",".join(map(str, value))
            else:
                if value is None:
                    value = ""
                if not isinstance(value, (str, Path)):
                    raise ValueError("must be a string")
                value = str(value).strip()
            if "choices" in spec and value not in spec["choices"]:
                raise ValueError("must be one of: " + ", ".join(spec["choices"]))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{key}: {exc}") from exc
        result[key] = value
    parse_layers(result["layers"])
    gpu_ids = result["gpu_ids"]
    if any(not value.isdecimal() for value in gpu_ids):
        raise ValueError("gpu_ids must contain nonnegative CUDA indices, e.g. 0,1")
    result["gpu_ids"] = [str(int(value)) for value in gpu_ids]
    if len(set(result["gpu_ids"])) != len(gpu_ids):
        raise ValueError("gpu_ids must not contain duplicate GPUs")
    if result["multi_gpu"] and len(gpu_ids) < 2:
        raise ValueError("multi_gpu requires at least two gpu_ids")
    if result["multi_gpu"] and result["device"] not in ("auto", "cuda") and not result["device"].startswith("cuda:"):
        raise ValueError("multi_gpu requires CUDA; choose device auto or cuda")
    if result["timestep_min"] >= result["timestep_max"]:
        raise ValueError("timestep_min must be smaller than timestep_max")
    if result["resume_from"] and result["init_adapter"]:
        raise ValueError("resume_from and init_adapter are mutually exclusive")
    data_key = "dataset_manifest" if result["objective"] == "sft" else "pairs_manifest"
    if not result[data_key]:
        raise ValueError("dataset_manifest is required for SFT; preprocess audio first" if data_key == "dataset_manifest"
                         else "pairs_manifest is required; preprocess preference pairs first")
    if not result["output_dir"]:
        raise ValueError("output_dir cannot be empty")
    if check_paths:
        data_path = Path(result[data_key])
        if not data_path.is_file() and not (data_key == "dataset_manifest" and data_path.is_dir()):
            raise FileNotFoundError(f"Training dataset not found: {data_path}")
        for key in ("resume_from", "init_adapter"):
            if result[key] and not Path(result[key]).is_dir():
                raise FileNotFoundError(f"{key} directory not found: {result[key]}")
        if result["checkpoint_file"]:
            if not Path(result["checkpoint_file"]).is_file():
                raise FileNotFoundError(f"Checkpoint file not found: {result['checkpoint_file']}")
            if not result["checkpoint_file"].lower().endswith(".safetensors"):
                raise ValueError("checkpoint_file must be a .safetensors file")
            if not result["model_config_dir"]:
                raise ValueError("model_config_dir is required with checkpoint_file")
            if not (Path(result["model_config_dir"]) / "config.json").is_file():
                raise FileNotFoundError("model_config_dir must contain the checkpoint's matching config.json")
        elif not Path(result["checkpoint_dir"]).is_dir():
            raise FileNotFoundError(f"checkpoint_dir not found: {result['checkpoint_dir']}")
    return result
