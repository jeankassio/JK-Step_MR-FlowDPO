"""Exercise real ACE weights on two GPUs using synthetic preference tensors.

This checks execution and resumable state. The resulting adapter is a software
test fixture, not a musical-quality LoRA suitable for generation.
"""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-file", required=True)
    parser.add_argument("--model-config-dir", required=True)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Reload both workers and continue to a second optimizer update")
    parser.add_argument("--export-comfyui", action="store_true", help="Also check native export; the file remains a synthetic test fixture")
    parser.add_argument("--output", default=".jk_step/multigpu-smoke")
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file
    from jk_step.config import schema_defaults
    from jk_step.distributed import launch_training

    root = Path(args.output).resolve() / time.strftime("%Y%m%d-%H%M%S")
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for index in range(6):
        generator = torch.Generator().manual_seed(1234 + index)
        payload = {
            "chosen_latents": torch.randn(8, 64, generator=generator) * .2,
            "rejected_latents": torch.randn(8, 64, generator=generator) * .3,
            "attention_mask": torch.ones(8),
            "encoder_hidden_states": torch.randn(4, 2048, generator=generator),
            "encoder_attention_mask": torch.ones(4),
            "context_latents": torch.zeros(8, 128),
        }
        cached = root / f"pair-{index}.pt"
        torch.save(payload, cached)
        rows.append({"id": str(index), "group_id": f"fixture-{index}",
                     "split": "train" if index < 4 else "validation",
                     "tensor_path": str(cached), "chosen": "chosen.wav",
                     "rejected": "rejected.wav", "caption": "software test",
                     "lyrics": "[Instrumental]", "pair_weight": 1 + index * .1})
    manifest = root / "pairs.json"
    manifest.write_text(json.dumps({"version": 1, "pairs": rows}), encoding="utf-8")
    config = schema_defaults() | {
        "auto_download": False, "checkpoint_file": str(Path(args.checkpoint_file).resolve()),
        "model_config_dir": str(Path(args.model_config_dir).resolve()),
        "model_variant": "xl-sft", "pairs_manifest": str(manifest),
        "output_dir": str(root / "run"), "multi_gpu": True,
        "gpu_ids": args.gpu_ids.split(","), "distributed_backend": "auto",
        "device": "auto", "precision": "bf16", "rank": args.rank,
        "alpha": args.rank * 2, "beta": 2, "learning_rate": 1e-5,
        "batch_size": 1, "gradient_accumulation": 2,
        "max_steps": 1, "epochs": 1, "warmup_steps": 0,
        "save_every": 1, "eval_every": 1, "eval_batches": 0,
        "tensorboard": False, "export_comfyui": args.export_comfyui,
    }
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    start = time.monotonic()
    result = launch_training(config, progress=lambda value: print(json.dumps(value), flush=True))
    state = torch.load(Path(result["adapter_dir"]) / "training_state.pt", map_location="cpu", weights_only=False)
    assert result["world_size"] == len(config["gpu_ids"]), result
    assert len(state["rank_rng"]) == len(config["gpu_ids"]), "Missing worker RNG state"
    assert state["runtime"]["global_step"] == 1, "Optimizer update was not completed"
    if args.resume:
        resumed = json.loads((Path(result["adapter_dir"]) / "training_config.json").read_text("utf-8"))
        resumed.update(resume_from=result["adapter_dir"], max_steps=2)
        result = launch_training(resumed, progress=lambda value: print(json.dumps(value), flush=True))
        state = torch.load(Path(result["adapter_dir"]) / "training_state.pt", map_location="cpu", weights_only=False)
        assert state["runtime"]["global_step"] == 2, "Both workers did not resume"
        assert len(state["rank_rng"]) == len(config["gpu_ids"])
    adapter = load_file(str(Path(result["adapter_dir"]) / "adapter_model.safetensors"))
    assert all(torch.isfinite(value).all() for value in adapter.values()), "Nonfinite LoRA weights"
    assert any("lora_B" in key and value.abs().sum() > 0 for key, value in adapter.items()), "LoRA did not update"
    if args.export_comfyui:
        exported = load_file(result["comfyui_path"])
        assert exported and all(torch.isfinite(value).all() for value in exported.values())
    print(json.dumps({"event": "passed", "world_size": result["world_size"],
                      "distributed_backend": result["distributed_backend"],
                      "rank_rng_saved": len(state["rank_rng"]), "lora_updated": True,
                      "rank": args.rank, "resumed": args.resume, "exported": args.export_comfyui,
                      "seconds": round(time.monotonic() - start, 2), "fixture_dir": str(root)}), flush=True)


if __name__ == "__main__":
    main()
