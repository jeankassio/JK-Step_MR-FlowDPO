"""ACE-Step preference preprocessing and aligned tensor dataloading.

Each pair has ONE caption/lyric encoder output and ONE context. The rejected
audio contributes only its VAE target. This avoids conditioning the two losses
on different semantic plans and mistaking a conditioning change for preference.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import struct
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .pairs import read_manifest, validate_pairs, write_manifest, _resolve, is_cancelled


def _weight_file_complete(path: Path) -> bool:
    """Check a weight container without allocating tensors or reading its payload.

    Hugging Face normally publishes a final filename only after completion, but
    a manually copied/truncated final file must not masquerade as a ready model.
    Safetensors provides its exact expected payload length in the header.
    """
    if not path.is_file():
        return False
    try:
        size = path.stat().st_size
        if path.suffix == ".safetensors":
            with path.open("rb") as stream:
                prefix = stream.read(8)
                if len(prefix) != 8:
                    return False
                header_size = struct.unpack("<Q", prefix)[0]
                if not 2 <= header_size <= min(64*1024*1024,size-8):
                    return False
                header = json.loads(stream.read(header_size))
            ranges = []
            for name,metadata in header.items():
                if name == "__metadata__":
                    continue
                start,end = metadata["data_offsets"]
                if not isinstance(start,int) or not isinstance(end,int) or start < 0 or end < start:
                    return False
                ranges.append((start,end))
            if not ranges:
                return False
            ranges.sort()
            cursor = 0
            for start,end in ranges:
                if start != cursor:
                    return False
                cursor = end
            return size == 8 + header_size + cursor
        if path.suffix == ".bin":
            # Modern torch.save archives include a terminal central directory;
            # reading it detects interrupted downloads without unpickling data.
            if not zipfile.is_zipfile(path):
                return False
            with zipfile.ZipFile(path) as archive:
                files = archive.infolist()
                return bool(files) and all(info.header_offset + info.compress_size < size for info in files)
    except (OSError,ValueError,TypeError,KeyError,AttributeError,zipfile.BadZipFile):
        return False
    return False


def model_weights_complete(directory: str | Path, component: str = "transformer") -> bool:
    """Accept standard local single-file/sharded HF model layouts.

    Configuration alone is insufficient. Every shard named by the index must
    be complete, and an index may not reference files outside its model folder.
    """
    root = Path(directory).expanduser().resolve()
    if not (root / "config.json").is_file():
        return False
    stems = ("diffusion_pytorch_model",) if component == "vae" else ("model","pytorch_model")
    for stem in stems:
        for extension in ("safetensors","bin"):
            if _weight_file_complete(root / f"{stem}.{extension}"):
                return True
    for stem in stems:
        for extension in ("safetensors","bin"):
            index = root / f"{stem}.{extension}.index.json"
            if not index.is_file():
                continue
            try:
                mapping = json.loads(index.read_text(encoding="utf-8-sig"))["weight_map"]
                names = set(mapping.values())
                if not names:
                    return False
                shards = [(root / name).resolve() for name in names]
                return all(path.is_relative_to(root) and _weight_file_complete(path) for path in shards)
            except (OSError,ValueError,TypeError,KeyError,AttributeError):
                return False
    return False


def preprocessing_models_ready(checkpoint_dir: str | Path) -> bool:
    root = Path(checkpoint_dir)
    text = root / "Qwen3-Embedding-0.6B"
    tokenizer_present = (text / "tokenizer_config.json").is_file() and (
        (text / "tokenizer.json").is_file() or ((text / "vocab.json").is_file() and (text / "merges.txt").is_file()))
    return model_weights_complete(root / "vae", "vae") and model_weights_complete(text) and tokenizer_present


def validate_tensor_pair(data: dict) -> None:
    import torch
    required = {"chosen_latents": 2, "rejected_latents": 2, "attention_mask": 1,
                "encoder_hidden_states": 2, "encoder_attention_mask": 1, "context_latents": 2}
    for key, ndim in required.items():
        value = data.get(key)
        if not torch.is_tensor(value) or value.ndim != ndim:
            raise ValueError(f"{key} must be a {ndim}D tensor.")
        if not torch.isfinite(value).all():
            raise ValueError(f"{key} contains nonfinite values.")
    chosen, rejected = data["chosen_latents"], data["rejected_latents"]
    if chosen.shape != rejected.shape or chosen.shape[-1] != 64 or chosen.shape[0] == 0:
        raise ValueError("Chosen/rejected latents must have identical nonempty [T,64] shapes.")
    if data["attention_mask"].shape[0] != chosen.shape[0] or data["context_latents"].shape[0] != chosen.shape[0]:
        raise ValueError("Audio mask and context must align with both branches.")
    if data["context_latents"].shape[-1] != 128:
        raise ValueError("ACE-Step context must contain 64 source channels plus 64 generation-mask channels.")
    if data["encoder_attention_mask"].shape[0] != data["encoder_hidden_states"].shape[0]:
        raise ValueError("Encoder mask length does not match encoder hidden states.")
    for key in ("attention_mask", "encoder_attention_mask"):
        mask = data[key]
        if not ((mask == 0) | (mask == 1)).all() or not mask.any():
            raise ValueError(f"{key} must be a nonempty binary mask.")
        # Remote ACE XL forward does not respect arbitrary internal padding.
        if (mask[1:] > mask[:-1]).any():
            raise ValueError(f"{key} must have contiguous valid prefix followed by padding.")


class PreferenceTensorDataset:
    """Lazy pair dataset. Both branches and context receive exactly the same crop."""
    def __init__(self, manifest: str | dict, split: str | None = "train",
                 max_latent_length: int | None = None, chunk_duration: float | None = None,
                 seed: int = 42, repeats: int = 1, verify_checksums: bool = False, **kwargs):
        self.manifest, self.base = read_manifest(manifest)
        self.pairs = [p for p in self.manifest.get("pairs", [])
                      if split is None or p.get("split", "train") == split]
        if not self.pairs:
            raise ValueError(f"No pairs found for split {split!r}.")
        if repeats < 1:
            raise ValueError("repeats must be positive.")
        self.max_latent_length = max_latent_length
        self.chunk_duration = chunk_duration
        if max_latent_length is not None and max_latent_length < 1:
            raise ValueError("max_latent_length must be positive.")
        if chunk_duration is not None and chunk_duration <= 0:
            raise ValueError("chunk_duration must be positive.")
        self.seed, self.repeats, self.epoch = seed, repeats, 0
        for pair in self.pairs:
            if not pair.get("tensor_path"):
                raise ValueError(f"Pair {pair.get('id')} has no tensor_path; run preprocess first.")
            path = Path(_resolve(pair["tensor_path"], self.base))
            if not path.is_file():
                raise FileNotFoundError(path)
            if verify_checksums and pair.get("tensor_sha256"):
                if _file_sha256(path) != pair["tensor_sha256"]:
                    raise ValueError(f"Tensor checksum mismatch: {path}")

    def __len__(self):
        return len(self.pairs) * self.repeats

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict:
        import torch
        pair = self.pairs[index % len(self.pairs)]
        data = torch.load(_resolve(pair["tensor_path"], self.base), map_location="cpu", weights_only=True)
        validate_tensor_pair(data)
        length = int(data["attention_mask"].sum().item())
        limit = self.max_latent_length
        if limit is None and self.chunk_duration:
            duration = float(data.get("metadata", {}).get("duration", length / 25))
            limit = max(1, int(round(self.chunk_duration * length / max(duration, 1e-8))))
        size = min(length, limit or length)
        # Respect model patch alignment where cached preprocessing knows it.
        patch = int(data.get("metadata", {}).get("patch_size", 1))
        if patch > 1:
            size = max(patch, size - size % patch)
            size = min(size, length)
        max_start = length - size
        # Stable across worker count/order and exact mid-epoch resume. Repeated
        # samples have different indexes, and every epoch changes the window.
        crop_rng = random.Random(self.seed + self.epoch * 1_000_003 + index * 97)
        start = crop_rng.randrange(max_start // patch + 1) * patch if max_start else 0
        result = {key: value for key, value in data.items() if torch.is_tensor(value)}
        for key in ("chosen_latents", "rejected_latents", "attention_mask", "context_latents"):
            result[key] = result[key][start:start + size].contiguous()
        result.update({"pair_weight": float(pair.get("pair_weight", 1.0)),
                       "id": pair.get("id", str(index)), "group_id": pair.get("group_id", ""),
                       "crop_start": start, "metadata": data.get("metadata", {})})
        return result


def collate_pairs(batch: list[dict], pad_to_multiple_of: int = 1) -> dict:
    import torch
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    if pad_to_multiple_of < 1:
        raise ValueError("pad_to_multiple_of must be positive.")
    max_audio = max(p["chosen_latents"].shape[0] for p in batch)
    max_audio = math.ceil(max_audio / pad_to_multiple_of) * pad_to_multiple_of
    max_text = max(p["encoder_hidden_states"].shape[0] for p in batch)
    keys = ("chosen_latents", "rejected_latents", "attention_mask", "encoder_hidden_states",
            "encoder_attention_mask", "context_latents")
    result = {}
    for key in keys:
        size = max_text if key.startswith("encoder_") else max_audio
        trailing = batch[0][key].shape[1:]
        values = []
        for sample in batch:
            value = sample[key]
            if value.shape[1:] != trailing:
                raise ValueError(f"Incompatible feature dimensions in batch: {key}")
            padded = value.new_zeros((size, *trailing))
            padded[:value.shape[0]] = value
            values.append(padded)
        result[key] = torch.stack(values)
    result["pair_weight"] = torch.tensor([p.get("pair_weight", 1) for p in batch], dtype=torch.float32)
    for key in ("id", "group_id", "metadata", "crop_start"):
        result[key] = [p.get(key) for p in batch]
    return result


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pair_filename(identifier: Any) -> str:
    # Imported IDs are data, never filesystem paths. Hash them to prevent an
    # ID such as '../other/file' or a Windows device name escaping output_dir.
    return hashlib.sha256(str(identifier).encode()).hexdigest()[:24]


def _fingerprint(pair: dict, checkpoint_dir: str, variant: str, options: dict) -> str:
    source_files = []
    for key in ("chosen", "rejected"):
        path = Path(pair[key])
        stat = path.stat()
        source_files.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    config_files = []
    paths = [Path(checkpoint_dir), Path(options.get("model_config_dir") or checkpoint_dir)]
    if options.get("checkpoint_file"):
        paths.append(Path(options["checkpoint_file"]))
    for path in paths:
        if path.is_file():
            stat = path.stat()
            config_files.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
        elif path.is_dir():
            for file in sorted([*path.rglob("config.json"), *path.rglob("*.py")]):
                stat = file.stat()
                config_files.append((str(file.resolve()), stat.st_size, stat.st_mtime_ns))
            for file in sorted(path.rglob("*.safetensors")):
                stat = file.stat()
                config_files.append((str(file.resolve()), stat.st_size, stat.st_mtime_ns))
    relevant = {k:options.get(k) for k in ("max_duration", "normalize", "target_db", "target_lufs",
                 "context_mode", "storage_dtype", "vae_posterior", "precision", "patch_size", "vae_chunk_seconds")}
    payload = {"pipeline": 2, "sources": source_files, "model": config_files, "variant": variant,
               "caption": pair.get("caption"), "lyrics": pair.get("lyrics"),
               "metadata": pair.get("metadata", {}), "options": relevant}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class _DeterministicVAE:
    """Reuse Side-Step overlap/discard tiling with posterior mean, not samples."""
    def __init__(self, vae):
        self.vae = vae
        self.dtype = vae.dtype

    def parameters(self):
        return self.vae.parameters()

    def encode(self, audio):
        encoded = self.vae.encode(audio)
        dist = encoded.latent_dist
        mean = dist.mode() if hasattr(dist, "mode") else dist.mean
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: mean))


def _atomic_torch_save(data: dict, path: Path) -> None:
    import torch
    staging = path.with_suffix(path.suffix + ".writing")
    torch.save(data, staging)
    staging.replace(path)


def preprocess_pairs(manifest: str | dict, checkpoint_dir: str, model_variant: str,
                     output_dir: str, **options) -> dict:
    """Real two-pass VAE/text then ACE DiT conditioning pipeline.

    context_mode='chosen_semantic' extracts discrete semantic codes from chosen
    audio and detokenizes them ONCE, sharing that plan with rejected. These are
    teacher-forced audio tokens, not LM-generated music. 'silence' is standard
    text/lyrics-only training. No Turbo checkpoint or audio is involved.
    """
    import torch
    from jk_engine.models.loader import (load_vae, load_text_encoder, load_silence_latent,
        load_decoder_for_training, unload_models, _resolve_dtype)
    from jk_engine.models.gpu_utils import detect_gpu
    from jk_engine.vendor.preprocess_audio import load_audio_stereo
    from jk_engine.vendor.preprocess_text import encode_text
    from jk_engine.vendor.preprocess_lyrics import encode_lyrics
    from jk_engine.vendor.preprocess_encoder import run_encoder
    from jk_engine.vendor.preprocess_context import build_context_latents
    from jk_engine.data.preprocess_vae import tiled_vae_encode
    from jk_engine.data.preprocess_prompt import build_simple_prompt
    from jk_engine.data.audio_normalize import normalize_audio

    raw, base = read_manifest(manifest)
    records = [dict(pair) for pair in raw.get("pairs", [])]
    for pair in records:
        for key in ("chosen", "rejected"):
            pair[key] = _resolve(pair[key], base)
    report = validate_pairs({"pairs": records}, check_audio=True)
    if not report["valid"]:
        raise ValueError("Cannot preprocess invalid pairs: " + "; ".join(report["errors"][:5]))
    if options.get("auto_download", True):
        root = Path(checkpoint_dir)
        variant_dir = {"xl-sft":"acestep-v15-xl-sft", "sft":"acestep-v15-sft",
                       "xl-base":"acestep-v15-xl-base", "base":"acestep-v15-base"}.get(model_variant,model_variant)
        expected_config = Path(options.get("model_config_dir") or root / variant_dir)
        expected_weight = options.get("checkpoint_file") or str(expected_config / "acestep_v1.5_sft_xl.safetensors")
        missing_components = not preprocessing_models_ready(root)
        model_available = (expected_config / "config.json").is_file() and (
            _weight_file_complete(Path(expected_weight)) or model_weights_complete(expected_config))
        if missing_components or not model_available:
            if model_variant != "xl-sft":
                raise ValueError("Automatic setup currently supplies SFT XL. Install the requested model, VAE and text encoder manually and set auto_download=false.")
            from .models import ensure_models
            callback = options.get("progress_callback")
            def download_progress(event):
                if callback:
                    callback(event.get("current",0),event.get("total",0),event.get("message",event.get("event","Downloading model")))
            ensured = ensure_models(checkpoint_dir, include_preprocess=True, progress=download_progress)
            for key in ("checkpoint_file", "model_config_dir"):
                if not options.get(key):
                    options[key] = ensured[key]
            if model_variant != ensured.get("model_variant",model_variant):
                raise ValueError("Automatic setup currently supplies SFT XL. Provide the requested variant manually and set auto_download=false.")
        if not options.get("checkpoint_file") and Path(expected_weight).is_file():
            options["checkpoint_file"] = expected_weight
            options.setdefault("model_config_dir",str(expected_config))
    mode = options.get("context_mode", "chosen_semantic")
    if mode not in ("silence", "chosen_semantic"):
        raise ValueError("context_mode must be silence or chosen_semantic.")
    if options.get("vae_posterior", "mean") not in ("mean", "sample"):
        raise ValueError("vae_posterior must be mean or sample.")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    gpu = detect_gpu(options.get("device", "auto"), options.get("precision", "auto"))
    device, precision = gpu.device, gpu.precision
    dtype = _resolve_dtype(precision)
    storage = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(
        options.get("storage_dtype", "bfloat16"))
    if storage is None:
        raise ValueError("storage_dtype must be float32, float16 or bfloat16.")
    callback, cancelled = options.get("progress_callback"), lambda: is_cancelled(options)
    max_duration = float(options.get("max_duration", 0))
    vae_chunk_seconds = float(options.get("vae_chunk_seconds", 15 if (gpu.vram_total_mb or 99999) <= 8192 else 30))
    if vae_chunk_seconds <= 4:
        raise ValueError("vae_chunk_seconds must exceed 4 seconds for the 2-second overlap on both sides.")
    fingerprints = {p["id"]: _fingerprint(p, checkpoint_dir, model_variant,
                    {**options, "context_mode":mode, "precision":precision,
                     "vae_chunk_seconds":vae_chunk_seconds}) for p in records}
    pending, ready, errors = [], [], []
    for pair in records:
        key = fingerprints[pair["id"]]
        final = output / f"{_pair_filename(pair['id'])}.pt"
        if final.is_file() and not options.get("force", False):
            try:
                data = torch.load(final, weights_only=True, map_location="cpu")
                validate_tensor_pair(data)
                if data.get("cache_key") == key:
                    pair["tensor_path"] = str(final)
                    ready.append(pair)
                    continue
            except Exception:
                pass
        pending.append(pair)
    staged = []
    if pending and not cancelled():
        vae = text_enc = tokenizer = silence = None
        try:
            vae = load_vae(checkpoint_dir, device, precision)
            tokenizer, text_enc = load_text_encoder(checkpoint_dir, device, precision)
            silence = load_silence_latent(checkpoint_dir, device, precision, variant=model_variant)
            proxy = _DeterministicVAE(vae) if options.get("vae_posterior", "mean") == "mean" else vae
            for index, pair in enumerate(pending):
                if cancelled():
                    break
                try:
                    key = fingerprints[pair["id"]]
                    stage = output / f"{_pair_filename(pair['id'])}.intermediate.pt"
                    if stage.is_file() and not options.get("force", False):
                        cached = torch.load(stage, weights_only=True, map_location="cpu")
                        if cached.get("cache_key") == key:
                            staged.append((pair, stage))
                            continue
                    latents, duration = [], 0.0
                    for branch in ("chosen", "rejected"):
                        audio, sr = load_audio_stereo(pair[branch], 48000, max_duration or 1e9)
                        duration = audio.shape[-1] / sr
                        normalization = options.get("normalize", "none")
                        if normalization != "none":
                            audio = normalize_audio(audio, sr, method=normalization,
                                target_db=float(options.get("target_db", -1)),
                                target_lufs=float(options.get("target_lufs", -14)))
                        with torch.no_grad():
                            encoded = tiled_vae_encode(proxy, audio.unsqueeze(0).to(device=device, dtype=vae.dtype), dtype,
                                                       chunk_size=int(vae_chunk_seconds * 48000))
                        latents.append(encoded.squeeze(0).cpu())
                        del audio, encoded
                    if latents[0].shape != latents[1].shape:
                        raise ValueError("VAE pair lengths differ; no silent independent crops are allowed.")
                    length = latents[0].shape[0]
                    metadata = {**pair.get("metadata", {}), "caption":pair.get("caption", ""),
                                "lyrics":pair.get("lyrics", "[Instrumental]"), "duration": duration}
                    prompt = build_simple_prompt(metadata, tag_position=options.get("tag_position", "prepend"))
                    with torch.no_grad():
                        text_hs, text_mask = encode_text(text_enc, tokenizer, prompt, device, dtype)
                        lyric_hs, lyric_mask = encode_lyrics(text_enc, tokenizer, metadata["lyrics"], device, dtype)
                    data = {"chosen_latents":latents[0], "rejected_latents":latents[1],
                            "text_hidden_states":text_hs.cpu(), "text_attention_mask":text_mask.cpu(),
                            "lyric_hidden_states":lyric_hs.cpu(), "lyric_attention_mask":lyric_mask.cpu(),
                            "attention_mask":torch.ones(length, dtype=dtype), "silence_latent":silence.cpu(),
                            "metadata":metadata, "cache_key":key}
                    if any(not torch.isfinite(v).all() for v in data.values() if torch.is_tensor(v)):
                        raise ValueError("VAE/text outputs contain NaN or infinity.")
                    _atomic_torch_save(data, stage)
                    staged.append((pair, stage))
                    if callback:
                        callback(index+1, len(pending)*2, f"VAE and shared text: {pair['id']}")
                except Exception as exc:
                    errors.append({"id":pair["id"], "phase":"vae_text", "error":str(exc)})
                    if options.get("fail_fast", True):
                        raise
        finally:
            unload_models(vae, text_enc, tokenizer, silence)
            del vae, text_enc, tokenizer, silence
    if staged and not cancelled():
        model = None
        try:
            if options.get("checkpoint_file") or options.get("model_config_dir"):
                from .training import load_base_model
                config = {"checkpoint_dir":checkpoint_dir, "model_variant":model_variant,
                    "checkpoint_file":options.get("checkpoint_file", ""),
                    "model_config_dir":options.get("model_config_dir", ""), "device":device,
                    "precision":precision, "offload_non_decoder":False}
                model = load_base_model(config)
            else:
                model = load_decoder_for_training(checkpoint_dir, model_variant, device, precision)
            model_device, model_dtype = next(model.parameters()).device, next(model.parameters()).dtype
            patch_size = int(getattr(model.config, "patch_size", 1))
            for index, (pair, stage) in enumerate(staged):
                if cancelled():
                    break
                try:
                    data = torch.load(stage, map_location="cpu", weights_only=True)
                    length = data["chosen_latents"].shape[0]
                    # Equal aligned crop only for the model's convolution patch constraint.
                    length -= length % patch_size
                    if not length:
                        raise ValueError("Audio is shorter than one model patch.")
                    silence = data["silence_latent"].to(model_device, model_dtype)
                    if silence.ndim == 2:
                        silence = silence.unsqueeze(0)
                    with torch.no_grad():
                        encoder_hs, encoder_mask = run_encoder(model,
                            text_hidden_states=data["text_hidden_states"].to(model_device,model_dtype),
                            text_attention_mask=data["text_attention_mask"].to(model_device,model_dtype),
                            lyric_hidden_states=data["lyric_hidden_states"].to(model_device,model_dtype),
                            lyric_attention_mask=data["lyric_attention_mask"].to(model_device,model_dtype),
                            device=str(model_device), dtype=model_dtype)
                        src = None
                        if mode == "chosen_semantic":
                            chosen = data["chosen_latents"][:length].unsqueeze(0).to(model_device,model_dtype)
                            mask = data["attention_mask"][:length].unsqueeze(0).to(model_device,model_dtype)
                            if not callable(getattr(model,"tokenize", None)) or not callable(getattr(model,"detokenize", None)):
                                raise ValueError("This model does not expose ACE tokenize/detokenize; select context_mode=silence.")
                            quantized, _, _ = model.tokenize(chosen, silence, mask)
                            src = model.detokenize(quantized)[:, :length]
                            if src.shape[-1] != 64:
                                raise ValueError("Decoded semantic context must have 64 channels.")
                        context = build_context_latents(silence,length,str(model_device),model_dtype,src_latents=src)
                    encoded = {"chosen_latents":data["chosen_latents"][:length].to(storage),
                               "rejected_latents":data["rejected_latents"][:length].to(storage),
                               "attention_mask":data["attention_mask"][:length].to(torch.bool),
                               "encoder_hidden_states":encoder_hs.squeeze(0).cpu().to(storage),
                               "encoder_attention_mask":encoder_mask.squeeze(0).cpu().to(torch.bool),
                               "context_latents":context.squeeze(0).cpu().to(storage),
                               "cache_key":data["cache_key"],
                               "metadata":{**data["metadata"], "model_variant":model_variant,
                                   "context_mode":mode,"conditioning_source":"chosen_only",
                                   "patch_size":patch_size,"vae_posterior":options.get("vae_posterior","mean")}}
                    validate_tensor_pair(encoded)
                    final = output / f"{_pair_filename(pair['id'])}.pt"
                    _atomic_torch_save(encoded, final)
                    pair["tensor_path"] = str(final)
                    if options.get("checksums", False):
                        pair["tensor_sha256"] = _file_sha256(final)
                    ready.append(pair)
                    stage.unlink(missing_ok=True)
                    if callback:
                        callback(len(pending)+index+1,len(pending)*2,f"Shared semantic context: {pair['id']}")
                except Exception as exc:
                    errors.append({"id":pair["id"], "phase":"dit_condition", "error":str(exc)})
                    if options.get("fail_fast", True):
                        raise
        finally:
            unload_models(model)
            del model
    result = {"version":1, "metadata":{**raw.get("metadata", {}), "preprocessing":{
        "checkpoint_dir":str(Path(checkpoint_dir).resolve()),"model_variant":model_variant,
        "context_mode":mode,"shared_conditioning":True,"errors":errors}}, "pairs":ready}
    output_manifest = write_manifest(output / "pairs.preprocessed.json", result)
    return {"manifest":output_manifest,"processed":len(ready),"total":len(records),
            "failed":len(errors),"errors":errors,"cancelled":bool(cancelled()),"output_dir":str(output)}
