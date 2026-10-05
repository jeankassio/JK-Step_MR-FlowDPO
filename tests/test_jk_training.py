"""Small real-PEFT tests: no large checkpoints and no mocked DPO objective."""
import json
import threading
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer

from jk_step.config import parse_layers, schema_defaults, schema_fields, validate_config
from jk_step.training import (_inject_adapter, _paired_dropout, _policy_mode,
                              _split_datasets, export_adapter, preference_step,
                              run_training)


class TinyConfig(PretrainedConfig):
    model_type = "jk_tiny_ace"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = 64
        self.is_turbo = False
        self.use_cache = False


class TinyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(64, 64, bias=False)
        self.v_proj = torch.nn.Linear(64, 64, bias=False)
        self.dropout = torch.nn.Dropout(0.8)
    def forward(self, inputs):
        return self.dropout(self.q_proj(inputs) + self.v_proj(inputs)) * 0.1


class TinyLayer(GradientCheckpointingLayer):
    def __init__(self):
        super().__init__()
        self.self_attn, self.cross_attn = TinyAttention(), TinyAttention()
        self.mlp = torch.nn.Module()
        self.mlp.up_proj = torch.nn.Linear(64, 64)
        self.mlp.down_proj = torch.nn.Linear(64, 64)
    def forward(self, inputs):
        return inputs + self.self_attn(inputs) + self.cross_attn(inputs)


class TinyDecoder(PreTrainedModel):
    config_class = TinyConfig
    supports_gradient_checkpointing = True
    def __init__(self):
        super().__init__(TinyConfig())
        self.layers = torch.nn.ModuleList([TinyLayer(), TinyLayer()])
        self.gradient_checkpointing = False
        self.calls = []
    def forward(self, hidden_states, timestep, timestep_r, attention_mask,
                encoder_hidden_states, encoder_attention_mask, context_latents,
                use_cache=False, **kwargs):
        self.calls.append({"hidden": hidden_states.detach().clone(),
                           "t": timestep.detach().clone(),
                           "cond": encoder_hidden_states.detach().clone()})
        hidden = hidden_states + 0.01 * encoder_hidden_states.mean(1, keepdim=True)
        hidden = hidden + context_latents[..., :64] * 0.01
        for layer in self.layers:
            hidden = layer(hidden)
        return (hidden,)


class TinyAce(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = TinyConfig()
        self.decoder = TinyDecoder()
        self.encoder = torch.nn.Linear(64, 64)
        self.null_condition_emb = torch.nn.Parameter(torch.zeros(1, 1, 64))


def configuration(**kwargs):
    return schema_defaults() | {"auto_download": False, "model_variant": "sft",
                                "rank": 2, "alpha": 4, "beta": 2,
                                "target_modules": ["q_proj", "v_proj"],
                                "precision": "fp32", "device": "cpu",
                                "tensorboard": False, "warmup_steps": 0,
                                "gradient_checkpointing": False,
                                "validation_fraction": 0, **kwargs}


def tensor_pair(frames=4):
    generator = torch.Generator().manual_seed(frames)
    return {"chosen_latents": torch.randn(frames, 64, generator=generator) * 0.2,
            "rejected_latents": torch.randn(frames, 64, generator=generator) * 0.3,
            "attention_mask": torch.ones(frames),
            "encoder_hidden_states": torch.randn(3, 64, generator=generator),
            "encoder_attention_mask": torch.ones(3),
            "context_latents": torch.zeros(frames, 128)}


def microbatch(frames=4):
    return {key: value.unsqueeze(0) for key, value in tensor_pair(frames).items()}


def prepare_manifest(directory: Path, count=4):
    rows = []
    for index in range(count):
        cached = directory / f"pair-{index}.pt"
        torch.save(tensor_pair(4 + index), cached)
        rows.append({"id": str(index), "group_id": f"song-{index}", "split": "train",
                     "tensor_path": str(cached), "chosen": "chosen.wav", "rejected": "rejected.wav",
                     "caption": "test", "lyrics": "same words"})
    manifest = directory / "pairs.json"
    manifest.write_text(json.dumps({"version": 1, "pairs": rows}), encoding="utf-8")
    return manifest


def test_configuration_is_complete_and_rejects_typographical_errors():
    defaults = schema_defaults()
    assert {row["name"] for row in schema_fields()} == set(defaults)
    parsed = validate_config({"pairs_manifest": "pairs.json", "rank": "64", "dropout": "0.05"})
    assert parsed["rank"] == 64 and parsed["dropout"] == 0.05
    with pytest.raises(ValueError, match="Unknown"):
        validate_config({"pairs_manifest": "pairs.json", "rnak": 64})
    with pytest.raises(ValueError):
        validate_config({"pairs_manifest": "pairs.json", "rank": 2.5})
    assert parse_layers("0-2,7") == [0, 1, 2, 7]


def test_real_peft_updates_only_lora_and_reference_is_original():
    torch.manual_seed(11)
    cfg = configuration()
    model = TinyAce()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    model, parameters = _inject_adapter(model, cfg)
    result = preference_step(model, microbatch(), cfg)
    assert result.loss.item() == pytest.approx(0.693147, abs=1e-5)
    result.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters)
    optimizer = torch.optim.AdamW(parameters, lr=0.01)
    optimizer.step()
    assert any("lora_B" in name and parameter.abs().sum() > 0 for name, parameter in model.named_parameters())
    for key, value in original.items():
        mapped = key.replace("decoder.", "decoder.base_model.model.", 1) if key.startswith("decoder.") else key
        if ".q_proj." in mapped or ".v_proj." in mapped:
            mapped = mapped.replace(".q_proj.", ".q_proj.base_layer.").replace(".v_proj.", ".v_proj.base_layer.")
        assert torch.equal(value, model.state_dict()[mapped])
    assert all("lora_" in key for key, param in model.named_parameters() if param.requires_grad)


def test_reference_and_policy_share_exact_noise_timestep_and_conditions():
    cfg = configuration(cfg_dropout=0.5)
    model, _ = _inject_adapter(TinyAce(), cfg)
    preference_step(model, microbatch(), cfg)
    reference, policy = model.decoder.get_base_model().calls[-2:]
    assert torch.equal(reference["hidden"], policy["hidden"])
    assert torch.equal(reference["t"], policy["t"])
    assert torch.equal(reference["cond"], policy["cond"])
    assert torch.equal(policy["t"][:1], policy["t"][1:])
    assert torch.equal(policy["cond"][:1], policy["cond"][1:])


def test_paired_adapter_dropout_reuses_mask():
    dropout = _paired_dropout(0.5)
    dropout.train()
    output = dropout(torch.ones(4, 8, 64))
    assert torch.equal(output[:2], output[2:])
    assert (output == 0).any() and (output == 2).any()


def test_gradient_checkpointing_keeps_base_dropout_disabled():
    cfg = configuration(gradient_checkpointing=True, dropout=0.2)
    model, parameters = _inject_adapter(TinyAce(), cfg)
    result = preference_step(model, microbatch(), cfg)
    result.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters)
    base = model.decoder.get_base_model()
    assert base.layers[0].training
    assert not base.layers[0].self_attn.dropout.training
    assert base.layers[0].self_attn.q_proj.lora_dropout["default"].training


def test_variable_length_padding_is_trimmed_before_decoder():
    from jk_step.preprocess import collate_pairs
    cfg = configuration()
    model, _ = _inject_adapter(TinyAce(), cfg)
    batch = collate_pairs([tensor_pair(4), tensor_pair(7)])
    result = preference_step(model, batch, cfg)
    assert torch.isfinite(result.loss)
    calls = model.decoder.get_base_model().calls
    assert {call["hidden"].shape[1] for call in calls} == {4, 7}
    assert all(call["hidden"].shape[0] == 2 for call in calls)


def test_decoder_layer_and_attention_selection():
    cfg = configuration(layers="1", attention_type="self")
    model, _ = _inject_adapter(TinyAce(), cfg)
    names = [name for name, param in model.named_parameters() if param.requires_grad]
    assert names and all("layers.1.self_attn." in name for name in names)


def test_group_holdout_has_no_recording_leakage(tmp_path):
    manifest = prepare_manifest(tmp_path, 6)
    cfg = configuration(pairs_manifest=str(manifest), validation_fraction=0.4)
    train, validation = _split_datasets(cfg)
    assert validation is not None
    assert {p["group_id"] for p in train.pairs}.isdisjoint({p["group_id"] for p in validation.pairs})


def test_resumable_end_to_end_training_matches_uninterrupted_run(tmp_path, monkeypatch):
    manifest = prepare_manifest(tmp_path)
    monkeypatch.setattr("jk_step.training.load_base_model", lambda config: TinyAce())
    common = configuration(checkpoint_dir=str(tmp_path), pairs_manifest=str(manifest),
                           max_steps=4, gradient_accumulation=2, learning_rate=0.001,
                           max_latent_length=3,
                           save_every=1, save_total_limit=2, export_comfyui=True)
    uninterrupted = run_training(common | {"output_dir": str(tmp_path / "whole")})
    event = threading.Event()
    def progress(record):
        if record["event"] == "training" and record["step"] == 2:
            event.set()
    stopped = run_training(common | {"output_dir": str(tmp_path / "split")},
                           progress=progress, stop_event=event)
    assert stopped["status"] == "stopped" and stopped["step"] == 2
    resumed = run_training(common | {"output_dir": str(tmp_path / "split"), "resume_from": stopped["adapter_dir"]})
    assert resumed["status"] == "complete" and resumed["step"] == 4
    a = load_file(str(Path(uninterrupted["adapter_dir"]) / "adapter_model.safetensors"))
    b = load_file(str(Path(resumed["adapter_dir"]) / "adapter_model.safetensors"))
    assert a.keys() == b.keys()
    assert all(torch.equal(a[key], b[key]) for key in a)
    assert Path(resumed["comfyui_path"]).is_file()
    assert len(list((tmp_path / "split").glob("checkpoint-*"))) <= 2


def test_comfy_export_preserves_rs_lora_scaling(tmp_path):
    cfg = configuration(use_rslora=True, rank=4, alpha=8)
    model, _ = _inject_adapter(TinyAce(), cfg)
    model.decoder.save_pretrained(str(tmp_path / "adapter"))
    result = export_adapter(tmp_path / "adapter", tmp_path / "comfy.safetensors")
    converted = load_file(result["output_path"])
    alphas = [value.item() for key, value in converted.items() if key.endswith(".alpha")]
    assert alphas and all(alpha == pytest.approx(16) for alpha in alphas)


def test_initial_adapter_cannot_silently_ignore_requested_rank(tmp_path):
    model, _ = _inject_adapter(TinyAce(), configuration(rank=4, alpha=8))
    model.decoder.save_pretrained(str(tmp_path / "adapter"))
    with pytest.raises(ValueError, match="existing adapter fixes"):
        _inject_adapter(TinyAce(), configuration(rank=2, alpha=4,
                                                 init_adapter=str(tmp_path / "adapter")))


def test_resume_rejects_silent_optimizer_override(tmp_path, monkeypatch):
    manifest = prepare_manifest(tmp_path)
    monkeypatch.setattr("jk_step.training.load_base_model", lambda config: TinyAce())
    config = configuration(checkpoint_dir=str(tmp_path), pairs_manifest=str(manifest),
                           output_dir=str(tmp_path / "run"), max_steps=1,
                           gradient_accumulation=1, export_comfyui=False)
    result = run_training(config)
    with pytest.raises(ValueError, match="Resume configuration changed: weight_decay"):
        run_training(config | {"resume_from": result["adapter_dir"], "max_steps":2,
                               "weight_decay": 0.33})
