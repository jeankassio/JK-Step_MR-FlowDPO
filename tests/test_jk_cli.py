import hashlib
import json
from pathlib import Path
import sys

import pytest

from jk_step.cli import _parser, main
from jk_step import models


def test_cli_overrides_all_config_types():
    args = _parser().parse_args(["train", "--rank", "64", "--alpha", "128",
                                "--no-auto-download", "--target-modules", "q_proj,v_proj",
                                "--rank-pattern", '{"layers.0.self_attn.q_proj":16}'])
    assert args.rank == 64 and args.alpha == 128 and not args.auto_download
    assert args.target_modules == ["q_proj", "v_proj"]
    assert args.rank_pattern["layers.0.self_attn.q_proj"] == 16


def test_folder_dataset_cli_inspects_without_models(tmp_path, capsys):
    (tmp_path / "song.wav").write_bytes(b"not decoded during inventory")
    (tmp_path / "ignored.txt").write_text("irrelevant")
    assert main(["dataset", "inspect", "--audio-dir", str(tmp_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["count"] == 1
    parsed = _parser().parse_args(["dataset", "prepare", "--audio-dir", str(tmp_path),
                                  "--options", '{"language":"pt"}'])
    assert parsed.audio_dir == str(tmp_path)


def test_sft_presets_and_cli_do_not_require_preference_pairs(tmp_path):
    from jk_step.cli import get_presets
    from jk_step.config import validate_config
    preset = get_presets()["sft_lora"]
    config = validate_config({**preset, "dataset_manifest": str(tmp_path / "supervised.json")})
    assert config["objective"] == "sft" and config["max_latent_length"] == 0
    assert not config["pairs_manifest"]
    parsed = _parser().parse_args(["train", "--objective", "sft", "--dataset-manifest", "data.json"])
    assert parsed.objective == "sft" and parsed.dataset_manifest == "data.json"


def test_multi_gpu_options_and_toolkit_compatibility():
    from jk_step.config import validate_config
    args = _parser().parse_args(["train", "--multi-gpu", "--gpu-ids", "0,1",
                                "--distributed-backend", "gloo"])
    assert args.multi_gpu and args.gpu_ids == ["0", "1"]
    config = validate_config({"pairs_manifest": "pairs.json", "multi_gpu": True,
                              "gpu_ids": [0, 1]})
    assert config["gpu_ids"] == ["0", "1"]
    for ids in (["0", "0"], ["0"], ["-1", "0"]):
        with pytest.raises(ValueError, match="gpu_ids"):
            validate_config({"pairs_manifest": "pairs.json", "multi_gpu": True,
                             "gpu_ids": ids})
    with pytest.raises(ValueError, match="CUDA"):
        validate_config({"pairs_manifest": "pairs.json", "multi_gpu": True,
                         "device": "cpu"})
    for command in ("toolkit", "sidestep"):
        parsed = _parser().parse_args([command, "dataset", "--help"])
        assert parsed.args == ["dataset", "--help"]


def test_config_generation_uses_public_relative_defaults(tmp_path, capsys):
    output = tmp_path / "config.json"
    assert main(["config", "--preset", "rank64", "--output", str(output)]) == 0
    data = json.loads(output.read_text())
    assert data["rank"] == 64 and data["auto_download"] is True
    assert data["checkpoint_dir"] == "checkpoints"
    assert data["checkpoint_file"] == ""


def test_model_ready_does_not_accept_configuration_only(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert not models.hf_weights_ready(tmp_path)


def test_single_safetensors_extent_and_shards(tmp_path):
    import torch
    from safetensors.torch import save_file
    (tmp_path / "config.json").write_text("{}")
    save_file({"weight": torch.zeros(4, 4)}, str(tmp_path / "model.safetensors"))
    assert models.hf_weights_ready(tmp_path)
    with (tmp_path / "model.safetensors").open("r+b") as handle:
        handle.truncate(64)
    assert not models.hf_weights_ready(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "missing.safetensors"}}))
    assert not models.hf_weights_ready(tmp_path)


def test_setup_does_not_accept_different_same_size_canonical(tmp_path, monkeypatch):
    import huggingface_hub
    target = tmp_path / "acestep-v15-xl-sft"
    target.mkdir()
    source = target / models.SFT_FILE
    source.write_bytes(b"real SFT fixture")
    canonical = target / "model.safetensors"
    canonical.write_bytes(b"wrong checkpoint")
    assert canonical.stat().st_size == source.stat().st_size
    monkeypatch.setattr(models, "prepare_metadata", lambda *args: {})
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args, **kwargs: str(source))
    monkeypatch.setattr(models, "SFT_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    with pytest.raises(RuntimeError, match="conflicts with the pure SFT"):
        models.ensure_models(str(tmp_path), include_preprocess=False)
    assert canonical.read_bytes() == b"wrong checkpoint"


def test_nonlink_canonical_copy_is_atomic(tmp_path, monkeypatch):
    import huggingface_hub
    target = tmp_path / "acestep-v15-xl-sft"; target.mkdir()
    source = target / models.SFT_FILE; source.write_bytes(b"complete checkpoint")
    monkeypatch.setattr(models, "prepare_metadata", lambda *args: {})
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args, **kwargs: str(source))
    def no_link(*args):
        raise OSError("Filesystem does not support hardlinks")
    monkeypatch.setattr(models.os, "link", no_link)
    result = models.ensure_models(str(tmp_path), include_preprocess=False)
    assert (target / "model.safetensors").read_bytes() == source.read_bytes()
    assert not (target / "model.safetensors.copying").exists()
    assert result["model_variant"] == "xl-sft"


def test_cache_key_includes_vae_tiling_and_architecture(tmp_path):
    from jk_step.preprocess import _fingerprint
    audio = tmp_path / "audio.wav"; audio.write_bytes(b"fixture")
    config = tmp_path / "model"; config.mkdir()
    architecture = config / "modeling.py"; architecture.write_text("v1")
    pair = {"chosen":str(audio), "rejected":str(audio), "caption":"tone", "lyrics":"[Instrumental]"}
    a = _fingerprint(pair, str(config), "xl-sft", {"vae_chunk_seconds":30})
    b = _fingerprint(pair, str(config), "xl-sft", {"vae_chunk_seconds":15})
    assert a != b
    architecture.write_text("v2 updated")
    assert a != _fingerprint(pair, str(config), "xl-sft", {"vae_chunk_seconds":30})
