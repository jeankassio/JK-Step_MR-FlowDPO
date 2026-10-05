"""Read checkpoint metadata without allocating tensors; fetch official config only."""
from __future__ import annotations
import json
from pathlib import Path
import struct
import os
import shutil
import hashlib

# Pin tested architecture/weights; never silently replace a checkpoint on resume.
SFT_REPO = "jeankassio/acestep_v1.5_sft_xl"
SFT_REVISION = "f906c560f7b3cfcaa4000d7939af9e3c418cc173"
SFT_FILE = "acestep_v1.5_sft_xl.safetensors"
SFT_SHA256 = "4d3416d564341d1c537956b06e97b290c97231d365d8bb2b5a7162b7b76ab0f6"
ARCH_REPO = "ACE-Step/acestep-v15-xl-sft"
ARCH_REVISION = "d06de46b4622f781cf07f4a013a67d591ca52819"


def ensure_models(checkpoint_dir: str = "checkpoints", *, include_preprocess: bool = True,
                  progress=None) -> dict:
    """Fetch the pure SFT XL from Jean's repo plus official encoder metadata.

    A Hugging Face local_dir download uses resumable cache metadata and verifies
    completion. Presence alone is not treated as a successful partial download.
    No Turbo, merge, or audio candidates are downloaded.
    """
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    root = Path(checkpoint_dir).expanduser().resolve()
    target = root / "acestep-v15-xl-sft"
    root.mkdir(parents=True, exist_ok=True)
    if progress:
        progress({"event": "model_download", "message": "SFT XL: 19.95 GB; existing complete downloads are reused", "repo": SFT_REPO})
    prepare_metadata(ARCH_REPO, str(target), ARCH_REVISION)
    weights = hf_hub_download(SFT_REPO, SFT_FILE, revision=SFT_REVISION, local_dir=str(target))
    # The original Side-Step tools expect the standard HF filename. A hardlink
    # gives both interfaces access to the same bytes without another 20 GB copy.
    canonical = target / "model.safetensors"
    expected_size = Path(weights).stat().st_size
    if canonical.exists() and canonical.stat().st_size != expected_size:
        raise RuntimeError(f"Incomplete/conflicting canonical checkpoint: {canonical}. Remove only this incomplete alias and rerun setup.")
    if canonical.exists() and not os.path.samefile(weights, canonical):
        if progress:
            progress({"event": "model_verify", "message": "Verifying existing separate canonical SFT copy"})
        with canonical.open("rb") as handle:
            actual_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual_hash != SFT_SHA256:
            raise RuntimeError(f"Canonical checkpoint conflicts with the pure SFT XL: {canonical}. Select another checkpoint directory; no file was overwritten.")
    if not canonical.exists():
        try:
            os.link(weights, canonical)
        except OSError:
            # Portable filesystems that lack links still get a valid HF layout.
            temporary = canonical.with_suffix(".safetensors.copying")
            shutil.copyfile(weights, temporary)
            if temporary.stat().st_size != expected_size:
                raise RuntimeError("Incomplete checkpoint copy")
            os.replace(temporary, canonical)
    record = {"checkpoint_dir": str(root), "checkpoint_file": str(Path(weights).resolve()),
              "model_config_dir": str(target), "model_variant": "xl-sft"}
    if include_preprocess:
        repo = "ACE-Step/Ace-Step1.5"
        source_file = root / "jk_preprocess_source.json"
        if source_file.is_file():
            revision = json.loads(source_file.read_text(encoding="utf-8"))["revision"]
        else:
            revision = HfApi().model_info(repo).sha
        if progress:
            progress({"event": "model_download", "message": "Official VAE and Qwen3-Embedding-0.6B", "repo": repo})
        snapshot_download(repo, revision=revision, local_dir=str(root),
                          allow_patterns=["vae/*", "Qwen3-Embedding-0.6B/*"])
        if not hf_weights_ready(root / "vae") or not hf_weights_ready(root / "Qwen3-Embedding-0.6B"):
            raise RuntimeError("Official preprocessing model download incomplete")
        source_file.write_text(json.dumps({"repo": repo, "revision": revision}, indent=2), encoding="utf-8")
    (root / "jk_models.json").write_text(json.dumps(record | {"repo": SFT_REPO, "revision": SFT_REVISION}, indent=2), encoding="utf-8")
    if progress:
        progress({"event": "models_ready", **record})
    return record


def hf_weights_ready(directory: str | Path) -> bool:
    """Check a local HF component has all weights, not just a downloaded config.

    Safetensors bounds are verified without importing torch. A shard index
    requires every referenced shard. This also accepts user-supplied complete
    local components that have no JK-Step download marker.
    """
    root = Path(directory)
    if not (root / "config.json").is_file():
        return False
    indexes = list(root.glob("*.index.json"))
    try:
        if indexes:
            required = set()
            for index in indexes:
                content = json.loads(index.read_text(encoding="utf-8"))
                required.update(content.get("weight_map", {}).values())
            files = [root / name for name in required]
        else:
            files = list(root.glob("*.safetensors")) or list(root.glob("*.bin"))
        if not files:
            return False
        for source in files:
            if not source.is_file() or source.stat().st_size < 64:
                return False
            if source.suffix == ".safetensors":
                with source.open("rb") as stream:
                    size = struct.unpack("<Q", stream.read(8))[0]
                    if size > 64 * 1024 * 1024:
                        return False
                    header = json.loads(stream.read(size))
                extent = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
                if 8 + size + extent != source.stat().st_size:
                    return False
        return True
    except (OSError, ValueError, TypeError, KeyError, struct.error):
        return False


def inspect_checkpoint(path: str) -> dict:
    source = Path(path).resolve()
    with source.open("rb") as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError("Not a safetensors file")
        size = struct.unpack("<Q", size_bytes)[0]
        if size > 64 * 1024 * 1024:
            raise ValueError("Invalid or unexpectedly large safetensors header")
        header = json.loads(stream.read(size))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    groups = {}
    for key, tensor in tensors.items():
        group = key.split(".")[0]
        groups[group] = groups.get(group, 0) + 1
    probes = {k: v for k, v in tensors.items()
              if any(part in k for part in ("in_proj.weight", "null_condition", "layers.0.self_attn.q_proj.weight"))}
    return {"file": str(source), "bytes": source.stat().st_size,
            "tensor_count": len(tensors), "groups": groups,
            "metadata": header.get("__metadata__", {}), "shape_probes": probes}


def prepare_metadata(repo: str, output: str, revision: str = "main") -> dict:
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().model_info(repo, revision=revision)
    pinned = info.sha
    available = {f.rfilename for f in info.siblings}
    target = Path(output).resolve(); target.mkdir(parents=True, exist_ok=True)
    selected = [p for p in available if p == "config.json" or p == "silence_latent.pt"
                or p.endswith(".py")]
    if "config.json" not in selected:
        raise ValueError("Repository has no config.json")
    files = []
    for name in sorted(selected):
        hf_hub_download(repo, name, revision=pinned, local_dir=str(target))
        files.append(name)
    record = {"repo": repo, "revision": pinned, "directory": str(target), "files": files,
              "weights_downloaded": False}
    (target / "jk_source.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
