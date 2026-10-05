"""CPU-only SFT objective, cache contracts, holdout and exact resume tests."""
import hashlib
import json
import threading
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from jk_step.config import validate_config
from jk_step.supervised_data import (SupervisedTensorDataset, collate_supervised,
    read_supervised_manifest, split_supervised_datasets, supervised_manifest_fingerprint)
from jk_step.training import _evaluate, _inject_adapter, run_training, supervised_step
from test_jk_training import TinyAce, configuration, tensor_pair


def tensor_sample(frames=4, *, lyrics="One two three"):
    pair = tensor_pair(frames)
    pair["target_latents"] = pair.pop("chosen_latents")
    pair.pop("rejected_latents")
    pair["metadata"] = {"lyrics": lyrics, "duration": frames / 25}
    return pair


def sft_config(**options):
    return configuration(objective="sft", max_latent_length=0, **options)


def prepare(directory, count=6, *, explicit=True, artists=True):
    rows = []
    for index in range(count):
        path = directory / f"audio-{index}.pt"
        torch.save(tensor_sample(4 + index % 3), path)
        row = {"id": str(index), "tensor_path": str(path),
               "tensor_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if artists:
            row["holdout_group"] = f"artist-{index // 2}"
        if explicit:
            row["split"] = "validation" if index >= count - 2 else "train"
        rows.append(row)
    path = directory / "supervised.json"
    path.write_text(json.dumps({"version": 1, "samples": rows}), encoding="utf-8")
    return path


def test_mode_specific_configuration_and_legacy_flow_defaults():
    cfg = validate_config({"objective": "sft", "dataset_manifest": "manifest.json"})
    assert cfg["max_latent_length"] == 0 and cfg["pairs_manifest"] == ""
    assert validate_config({"pairs_manifest": "pairs.json"})["objective"] == "flow_dpo"
    with pytest.raises(ValueError, match="dataset_manifest"):
        validate_config({"objective": "sft", "pairs_manifest": "pairs.json"})
    with pytest.raises(ValueError, match="pairs_manifest"):
        validate_config({"dataset_manifest": "manifest.json"})


def test_side_step_manifest_and_directory_exact_single_target_contract(tmp_path):
    cached = tmp_path / "recording.pt"
    torch.save(tensor_sample(7), cached)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": ["recording.pt"]}), encoding="utf-8")
    dataset = SupervisedTensorDataset(tmp_path)
    row = dataset[0]
    assert torch.equal(row["target_latents"], torch.load(cached, weights_only=True)["target_latents"])
    assert row["crop_start"] == 0 and row["target_latents"].shape == (7, 64)
    assert "rejected_latents" not in row and "chosen_latents" not in row
    assert row["metadata"]["lyrics"] == "One two three"
    # Bad manifest paths never fall back to a directory of unrelated caches.
    manifest.write_text('{"samples":["absent.pt"]}', encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        SupervisedTensorDataset(tmp_path)


def test_group_split_occurs_before_train_only_repeats(tmp_path):
    path = prepare(tmp_path, explicit=False)
    train, validation = split_supervised_datasets(sft_config(
        dataset_manifest=str(path), validation_fraction=0.4, dataset_repeats=7))
    assert validation is not None
    assert {row["holdout_group"] for row in train.records}.isdisjoint(
        {row["holdout_group"] for row in validation.records})
    assert len(train) == len(train.records) * 7
    assert len(validation) == len(validation.records)


def test_explicit_holdout_and_duplicate_recording_leakage_fail(tmp_path):
    path = prepare(tmp_path)
    document = json.loads(path.read_text())
    document["samples"][-1]["holdout_group"] = document["samples"][0]["holdout_group"]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="leakage"):
        split_supervised_datasets(sft_config(dataset_manifest=str(path)))
    document["samples"][-1]["holdout_group"] = "different-artist"
    document["samples"][-1]["tensor_path"] = document["samples"][0]["tensor_path"]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="leakage"):
        split_supervised_datasets(sft_config(dataset_manifest=str(path)))


def test_explicit_all_train_is_not_randomly_resplit(tmp_path):
    path = prepare(tmp_path)
    document = json.loads(path.read_text())
    for row in document["samples"]:
        row["split"] = "train"
    path.write_text(json.dumps(document))
    train, validation = split_supervised_datasets(sft_config(
        dataset_manifest=str(path), validation_fraction=0.5))
    assert len(train.records) == 6 and validation is None


def test_copied_audio_hash_and_normalized_artist_cannot_leak(tmp_path):
    path = prepare(tmp_path)
    document = json.loads(path.read_text())
    document["samples"][0]["audio_sha256"] = "abc123"
    document["samples"][-1]["audio_sha256"] = "abc123"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="leakage"):
        split_supervised_datasets(sft_config(dataset_manifest=str(path)))
    for row in document["samples"]:
        row.pop("audio_sha256", None)
        row.pop("holdout_group")
        row["metadata"] = {"artist": "Some unique artist " + row["id"]}
    document["samples"][0]["metadata"]["artist"] = "Túlio Borges"
    document["samples"][-1]["metadata"]["artist"] = "  tulio   borges "
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="leakage"):
        split_supervised_datasets(sft_config(dataset_manifest=str(path)))


def test_vocal_truncation_rejected_and_instrumental_crop_reproducible(tmp_path):
    cached = tmp_path / "clip.pt"
    torch.save(tensor_sample(12), cached)
    document = {"samples": [str(cached)]}
    with pytest.raises(ValueError, match="truncate vocals"):
        SupervisedTensorDataset(document, max_latent_length=4)
    data = tensor_sample(12, lyrics="[Instrumental]")
    torch.save(data, cached)
    a = SupervisedTensorDataset(document, max_latent_length=4, repeats=3, seed=83)
    b = SupervisedTensorDataset(document, max_latent_length=4, repeats=3, seed=83)
    a.set_epoch(4); b.set_epoch(4)
    assert all(torch.equal(a[i]["target_latents"], b[i]["target_latents"]) for i in range(3))


def test_corrupt_tensor_and_declared_checksum_rejected(tmp_path):
    path = prepare(tmp_path)
    row = json.loads(path.read_text())["samples"][0]
    cached = Path(row["tensor_path"])
    before = supervised_manifest_fingerprint(path)
    data = torch.load(cached, weights_only=True)
    data["target_latents"][0, 0] = float("nan")
    torch.save(data, cached)
    assert before != supervised_manifest_fingerprint(path)
    with pytest.raises(ValueError, match="checksum"):
        SupervisedTensorDataset(path, verify_checksums=True)
    with pytest.raises(ValueError, match="non-finite"):
        SupervisedTensorDataset(path)


def test_single_branch_has_gradients_no_reference_and_normal_dropout():
    torch.manual_seed(84)
    config = sft_config(dropout=0.3)
    model, parameters = _inject_adapter(TinyAce(), config)
    batch = collate_supervised([tensor_sample()])
    result = supervised_step(model, batch, config)
    assert result.loss.dtype == torch.float32 and not hasattr(result, "dpo_loss")
    assert len(model.decoder.get_base_model().calls) == 1
    assert model.decoder.get_base_model().calls[0]["hidden"].shape[0] == 1
    result.loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in parameters)
    dropout = model.decoder.get_base_model().layers[0].self_attn.q_proj.lora_dropout["default"]
    assert isinstance(dropout, torch.nn.Dropout) and dropout.p == 0.3


def test_sft_padding_does_not_change_loss_and_preference_coefficients_are_ignored():
    config = sft_config()
    model, _ = _inject_adapter(TinyAce(), config)
    batch = collate_supervised([tensor_sample(4), tensor_sample(7)])
    torch.manual_seed(85)
    first = supervised_step(model, batch, config, training=False)
    assert {call["hidden"].shape[1] for call in model.decoder.get_base_model().calls} == {4, 7}
    assert all(call["hidden"].shape[0] == 1 for call in model.decoder.get_base_model().calls)
    batch["target_latents"][0, 4:] = 500
    batch["encoder_hidden_states"][0, 3:] = 800
    torch.manual_seed(85)
    second = supervised_step(model, batch, config | {"beta": 1e9,
        "fm_regularization": 900, "reference_regularization": 100,
        "label_smoothing": 0.49}, training=False)
    assert torch.equal(first.loss, second.loss)


def test_sft_validation_returns_only_real_supervised_metrics():
    config = sft_config(eval_batches=0)
    model, _ = _inject_adapter(TinyAce(), config)
    metrics = _evaluate(model, [collate_supervised([tensor_sample()])], config, torch.device("cpu"))
    assert set(metrics) == {"validation_loss", "validation_chosen_fm"}
    assert metrics["validation_loss"] == metrics["validation_chosen_fm"]


def test_real_sft_resume_matches_uninterrupted_and_exports(tmp_path, monkeypatch):
    path = prepare(tmp_path)
    monkeypatch.setattr("jk_step.training.load_base_model", lambda config: TinyAce())
    common = sft_config(checkpoint_dir=str(tmp_path), dataset_manifest=str(path),
        max_steps=4, gradient_accumulation=2, learning_rate=0.001,
        dropout=0.2, eval_every=1, eval_batches=0, save_every=1,
        save_total_limit=2, export_comfyui=True)
    records = []
    whole = run_training(common | {"output_dir": str(tmp_path / "whole")}, progress=records.append)
    event = threading.Event()
    def progress(record):
        if record["event"] == "training" and record["step"] == 2:
            event.set()
    stopped = run_training(common | {"output_dir": str(tmp_path / "part")}, progress=progress, stop_event=event)
    resumed = run_training(common | {"output_dir": str(tmp_path / "part"), "resume_from": stopped["adapter_dir"]})
    assert whole["objective"] == "sft" and resumed["step"] == 4
    a = load_file(str(Path(whole["adapter_dir"]) / "adapter_model.safetensors"))
    b = load_file(str(Path(resumed["adapter_dir"]) / "adapter_model.safetensors"))
    assert all(torch.equal(a[key], b[key]) for key in a)
    assert any("lora_B" in key and a[key].abs().sum() > 0 for key in a)
    assert Path(resumed["comfyui_path"]).is_file()
    started = next(row for row in records if row["event"] == "started")
    assert started["train_samples"] == 4 and started["validation_samples"] == 2
    assert not any(any("dpo" in key or "preference" in key for key in row)
                   for row in records if row["event"] in ("training", "validation"))
    state = torch.load(Path(resumed["adapter_dir"]) / "training_state.pt", weights_only=True)
    assert state["runtime"]["objective"] == "sft"
    with pytest.raises(ValueError, match="dataset changed"):
        data = torch.load(tmp_path / "audio-0.pt", weights_only=True)
        data["target_latents"][0, 0] += 0.01
        torch.save(data, tmp_path / "audio-0.pt")
        run_training(common | {"output_dir": str(tmp_path / "part"), "resume_from": resumed["adapter_dir"]})
