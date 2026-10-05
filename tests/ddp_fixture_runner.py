"""Two real IPC workers for the distributed regression suite; CPU only."""
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
from test_jk_training import TinyAce, configuration, tensor_pair


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
        digest.update(name.encode())
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def new_policy(config):
    torch.manual_seed(811)
    return training._inject_adapter(TinyAce(), config)


def batch_for_rank(rank):
    result = {key: value.unsqueeze(0) for key, value in tensor_pair(4 + rank).items()}
    result["pair_weight"] = torch.tensor([1.0 + rank * 2])
    return result


def gradient_oracle(context):
    """Compare reduced real Flow-DPO gradients with one effective batch.

    Oracle executes the same two independently seeded examples serially and
    differentiates the weighted objective, without calling synchronization.
    Unequal weights make an accidental rank-average detectably incorrect.
    """
    config = configuration(dropout=0, fm_regularization=0.1)
    reports = {}
    for average in (False, True):
        policy, parameters = new_policy(config)
        torch.manual_seed(900 + context.rank)
        result = training.preference_step(policy, batch_for_rank(context.rank), config)
        scale = 1 if average else (1.0 + context.rank * 2) / 4
        (result.loss * scale).backward()
        context.synchronize_gradients(parameters, average=average, bucket_bytes=513)
        actual = torch.cat([parameter.grad.flatten() for parameter in parameters])
        oracle, oracle_parameters = new_policy(config)
        for rank in range(context.world_size):
            torch.manual_seed(900 + rank)
            expected_result = training.preference_step(oracle, batch_for_rank(rank), config)
            expected_scale = 0.5 if average else (1.0 + rank * 2) / 4
            (expected_result.loss * expected_scale).backward()
        expected = torch.cat([parameter.grad.flatten() for parameter in oracle_parameters])
        difference = float((actual - expected).abs().max())
        assert torch.allclose(actual, expected, atol=2e-7, rtol=2e-5), difference
        optimizer = torch.optim.AdamW(parameters, lr=0.01)
        optimizer.step()
        hashes = context.gather_object(adapter_hash(policy.decoder))
        assert len(set(hashes)) == 1
        reports["average" if average else "weighted_sum"] = {
            "maximum_gradient_error": difference,
            "gradient_norm": float(actual.norm()), "adapter_hashes": hashes}
    return reports


def prepare_data(directory, context):
    manifest = directory / "pairs.json"
    if context.is_main:
        rows = []
        for index in range(15):
            path = directory / f"pair-{index}.pt"
            torch.save(tensor_pair(4 + index % 3), path)
            rows.append({"id": str(index), "group_id": f"song-{index}",
                         "split": "train" if index < 10 else "validation",
                         "tensor_path": str(path), "pair_weight": 1.0 + index % 4,
                         "chosen": "chosen.wav", "rejected": "rejected.wav",
                         "caption": "Piano and voice", "lyrics": "One two three"})
        manifest.write_text(json.dumps({"version": 1, "pairs": rows}), "utf-8")
    context.barrier()
    return manifest


def training_scenarios(directory, context):
    manifest = prepare_data(directory, context)
    base = configuration(checkpoint_dir=str(directory), pairs_manifest=str(manifest),
                         batch_size=1, gradient_accumulation=2, max_steps=3,
                         learning_rate=0.001, scheduler="constant", max_latent_length=3,
                         dropout=0.25, eval_every=1, eval_batches=0,
                         save_every=1, save_total_limit=2, export_comfyui=True,
                         distributed_backend="local", multi_gpu=False)
    original_step = training.preference_step
    original_save = training._checkpoint_save
    original_load = training.load_base_model
    traces, saved_by_rank, models = {}, {}, {}
    current = None
    local_stop = None

    def load_tiny(config):
        model = TinyAce()
        models[current] = model
        return model

    def traced_step(model, batch, config, *, training=True):
        result = original_step(model, batch, config, training=training)
        trace = traces[current]
        ids = list(batch.get("id", []))
        if training:
            trace["training_ids"].extend(ids)
            # Only rank 1 requests cancellation; every worker must stop at the
            # same committed optimizer step and participate in checkpointing.
            if current == "stopped" and context.rank == 1 and len(trace["training_ids"]) == 4:
                local_stop.set()
        else:
            trace["validation"].append({"ids": ids, "loss": float(result.loss),
                "dpo": float(result.dpo_loss),
                "accuracy": float((result.preference_logits > 0).float().mean()),
                "chosen_fm": float(result.chosen_fm.mean()), "count": len(ids),
                "weight_mass": float(batch["pair_weight"].sum())})
        return result

    def traced_save(*args, **kwargs):
        saved_by_rank[current].append(str(args[0]))
        return original_save(*args, **kwargs)

    training.load_base_model = load_tiny
    training.preference_step = traced_step
    training._checkpoint_save = traced_save
    results = {}
    try:
        for scenario in ("whole", "stopped", "resumed"):
            current = scenario
            traces[current] = {"training_ids": [], "validation": [], "progress": []}
            saved_by_rank[current] = []
            local_stop = threading.Event()
            config = base | {"output_dir": str(directory / ("whole" if scenario == "whole" else "split"))}
            if scenario == "resumed":
                config["resume_from"] = results["stopped"]["adapter_dir"]
            result = training.run_training(config, progress=traces[current]["progress"].append,
                                           stop_event=local_stop)
            results[current] = result
            context.barrier()
            replica_hashes = context.gather_object(adapter_hash(models[current].decoder))
            assert len(set(replica_hashes)) == 1, (current, replica_hashes)
            traces[current]["replica_hashes"] = replica_hashes
    finally:
        training.load_base_model = original_load
        training.preference_step = original_step
        training._checkpoint_save = original_save
    checkpoint = Path(results["stopped"]["adapter_dir"])
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
    rank_rng = state.get("rank_rng", [])
    rng_different = len(rank_rng) == 2 and not torch.equal(rank_rng[0]["torch"], rank_rng[1]["torch"])
    return {"results": results, "traces": traces, "save_calls": saved_by_rank,
            "rank_rng_count": len(rank_rng), "rank_rng_different": rng_different,
            "checkpoint_world_size": state["runtime"].get("world_size"),
            "whole_adapter": file_hash(Path(results["whole"]["adapter_dir"]) / "adapter_model.safetensors"),
            "resumed_adapter": file_hash(Path(results["resumed"]["adapter_dir"]) / "adapter_model.safetensors")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    directory = Path(args.directory)
    torch.set_num_threads(1)
    assert not torch.cuda.is_available(), "Distributed fixture must run without GPU visibility"
    context = initialize_distributed({"distributed_backend": "local"}, torch.device("cpu"))
    try:
        report = {"rank": context.rank, "world_size": context.world_size, "backend": context.backend,
                  "gradients": gradient_oracle(context)}
        report["engine"] = training_scenarios(directory, context)
        (directory / f"rank-{context.rank}.json").write_text(json.dumps(report, indent=2), "utf-8")
        context.barrier()
    finally:
        context.close()


if __name__ == "__main__":
    main()
