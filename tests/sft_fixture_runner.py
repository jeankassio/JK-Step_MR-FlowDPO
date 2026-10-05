"""Real two-process CPU supervised training fixture; no CUDA visibility."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from safetensors.torch import load_file
from jk_step import training
from jk_step.distributed import initialize_distributed
from jk_step.supervised_data import collate_supervised
from test_jk_training import TinyAce
from test_jk_supervised import sft_config, tensor_sample


def adapter_hash(decoder):
    digest = hashlib.sha256()
    for name, value in sorted(decoder.state_dict().items()):
        if "lora_" in name:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    for name, value in sorted(load_file(str(path)).items()):
        digest.update(name.encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def gradient_oracle(context):
    config = sft_config(dropout=0)
    def policy():
        torch.manual_seed(87)
        return training._inject_adapter(TinyAce(), config)
    def batch(rank):
        # Unequal local batch sizes verify correct global sample weighting.
        return collate_supervised([tensor_sample(4) for _ in range(2 if rank == 0 else 1)])
    model, parameters = policy()
    torch.manual_seed(900 + context.rank)
    result = training.supervised_step(model, batch(context.rank), config)
    (result.loss * (2 if context.rank == 0 else 1) / 3).backward()
    context.synchronize_gradients(parameters, bucket_bytes=513)
    actual = torch.cat([parameter.grad.flatten() for parameter in parameters])
    oracle, oracle_parameters = policy()
    for rank in range(2):
        torch.manual_seed(900 + rank)
        result = training.supervised_step(oracle, batch(rank), config)
        (result.loss * (2 if rank == 0 else 1) / 3).backward()
    expected = torch.cat([parameter.grad.flatten() for parameter in oracle_parameters])
    difference = float((actual - expected).abs().max())
    assert torch.allclose(actual, expected, atol=2e-7, rtol=2e-5), difference
    return difference


def scenarios(directory, context):
    manifest = directory / "supervised.json"
    if context.is_main:
        rows = []
        for index in range(11):
            path = directory / f"clip-{index}.pt"
            torch.save(tensor_sample(4 + index % 3), path)
            rows.append({"id": str(index), "tensor_path": str(path),
                         "holdout_group": f"song-{index}",
                         "split": "train" if index < 8 else "validation"})
        manifest.write_text(json.dumps({"samples": rows}), encoding="utf-8")
    context.barrier()
    base = sft_config(checkpoint_dir=str(directory), dataset_manifest=str(manifest),
        batch_size=1, gradient_accumulation=2, max_steps=3, learning_rate=0.001,
        scheduler="constant", dropout=0.25, eval_every=1, eval_batches=0,
        save_every=1, save_total_limit=2, export_comfyui=True, distributed_backend="local")
    original_load, original_step, original_save = training.load_base_model, training.supervised_step, training._checkpoint_save
    current = None
    local_stop = None
    models, traces, saved, results = {}, {}, {}, {}
    def load(config):
        models[current] = TinyAce()
        return models[current]
    def step(model, batch, config, *, training=True):
        result = original_step(model, batch, config, training=training)
        trace = traces[current]
        if training:
            trace["ids"].extend(batch["id"])
            if current == "stopped" and context.rank == 1 and len(trace["ids"]) == 4:
                local_stop.set()
        else:
            trace["validation"].append({"ids": batch["id"], "loss": float(result.loss),
                                         "count": batch["target_latents"].shape[0]})
        return result
    def save(*args, **kwargs):
        saved[current].append(str(args[0]))
        return original_save(*args, **kwargs)
    training.load_base_model, training.supervised_step, training._checkpoint_save = load, step, save
    try:
        for current in ("whole", "stopped", "resumed"):
            local_stop = threading.Event()
            traces[current] = {"ids": [], "validation": [], "progress": []}
            saved[current] = []
            config = base | {"output_dir": str(directory / ("whole" if current == "whole" else "split"))}
            if current == "resumed":
                config["resume_from"] = results["stopped"]["adapter_dir"]
            results[current] = training.run_training(config, progress=traces[current]["progress"].append,
                                                      stop_event=local_stop)
            hashes = context.gather_object(adapter_hash(models[current].decoder))
            assert len(set(hashes)) == 1
            traces[current]["replica_hashes"] = hashes
    finally:
        training.load_base_model, training.supervised_step, training._checkpoint_save = original_load, original_step, original_save
    state = torch.load(Path(results["stopped"]["adapter_dir"]) / "training_state.pt", weights_only=True)
    return {"results": results, "traces": traces, "saves": saved,
            "rank_rng_count": len(state["rank_rng"]),
            "objective": state["runtime"]["objective"],
            "whole_hash": file_hash(Path(results["whole"]["adapter_dir"]) / "adapter_model.safetensors"),
            "resume_hash": file_hash(Path(results["resumed"]["adapter_dir"]) / "adapter_model.safetensors")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    directory = Path(parser.parse_args().directory)
    torch.set_num_threads(1)
    assert not torch.cuda.is_available(), "SFT fixture cannot run with visible GPUs"
    context = initialize_distributed({"distributed_backend": "local"}, torch.device("cpu"))
    try:
        report = {"rank": context.rank, "gradient_error": gradient_oracle(context),
                  "engine": scenarios(directory, context)}
        (directory / f"sft-rank-{context.rank}.json").write_text(json.dumps(report), encoding="utf-8")
        context.barrier()
    finally:
        context.close()


if __name__ == "__main__":
    main()
