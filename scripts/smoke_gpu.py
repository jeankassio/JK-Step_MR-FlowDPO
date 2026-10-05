"""Integration check: real ACE weights + LoRA backward, synthetic latents only.

This verifies the implementation, not perceptual improvement. It never exports
a trained quality LoRA from these synthetic tensors for use in music generation.
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
    args = parser.parse_args()
    import torch
    from jk_step.config import schema_defaults
    from jk_step.training import load_base_model, _inject_adapter, preference_step
    config = schema_defaults() | {"checkpoint_file": args.checkpoint_file,
        "model_config_dir": args.model_config_dir, "device": "cuda:0", "precision": "bf16",
        "rank": 2, "alpha": 4, "beta": 2, "gradient_checkpointing": True,
        "offload_non_decoder": True}
    torch.manual_seed(1234)
    start = time.time()
    print(json.dumps({"event": "loading", "gpu": torch.cuda.get_device_name(0)}), flush=True)
    model = load_base_model(config)
    model, trainable = _inject_adapter(model, config)
    base_parameter = next(p for p in model.decoder.parameters() if not p.requires_grad)
    base_probe = base_parameter.flatten()[:32].detach().clone()
    batch = {"chosen_latents": torch.randn(1, 8, 64),
             "rejected_latents": torch.randn(1, 8, 64),
             "attention_mask": torch.ones(1, 8),
             "encoder_hidden_states": torch.randn(1, 4, 2048),
             "encoder_attention_mask": torch.ones(1, 4),
             "context_latents": torch.zeros(1, 8, 128),
             "pair_weight": torch.ones(1)}
    before = [p.detach().clone() for p in trainable]
    print(json.dumps({"event": "backward", "trainable_parameters": sum(p.numel() for p in trainable)}), flush=True)
    result = preference_step(model, batch, config)
    result.loss.backward()
    assert torch.isfinite(result.loss), "Loss is nonfinite"
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable), "Gradient is nonfinite"
    grad = sum(float(p.grad.float().abs().sum()) for p in trainable if p.grad is not None)
    assert grad > 0, "No LoRA gradients"
    optimizer = torch.optim.AdamW(trainable, lr=1e-5)
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, trainable)), "LoRA did not update"
    assert torch.equal(base_probe, base_parameter.flatten()[:32]), "Frozen SFT changed"
    print(json.dumps({"event": "passed", "loss": float(result.loss.detach()),
                     "gradient_l1": grad, "seconds": round(time.time()-start, 2),
                     "peak_vram_gib": round(torch.cuda.max_memory_allocated()/2**30, 2),
                     "base_frozen": True, "lora_updated": True}), flush=True)


if __name__ == "__main__":
    main()
