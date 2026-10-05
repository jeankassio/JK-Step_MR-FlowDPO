"""Single-target ACE-Step SFT data, compatible with Side-Step tensor caches.

Imports remain CPU/lightweight until tensor data is actually inspected. Split
unique recordings before repeating training examples. Prepared vocal clips
retain their exact audio and lyric conditioning; random vocal truncation is
deliberately rejected rather than silently training on the full song's lyrics.
"""
from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


TENSOR_KEYS = ("target_latents", "attention_mask", "encoder_hidden_states",
               "encoder_attention_mask", "context_latents")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_supervised_manifest(manifest: str | Path | dict | list) -> tuple[dict, Path]:
    """Read prepared rows or the original Side-Step list of cached .pt paths.

    Every returned tensor_path is absolute. An explicit JSON with broken paths
    is an error; it never silently falls back to unrelated files in its folder.
    """
    source = None
    if isinstance(manifest, (str, Path)):
        source = Path(manifest).expanduser().resolve()
        if source.is_dir():
            if (source / "manifest.json").is_file():
                source = source / "manifest.json"
            else:
                document = {"samples": [str(path) for path in sorted(source.glob("*.pt"))
                                         if not path.name.endswith(".tmp.pt")]}
        if source.is_file():
            document = json.loads(source.read_text(encoding="utf-8-sig"))
        elif not source.is_dir():
            raise FileNotFoundError(source)
        base = source.parent if source.is_file() else source
    else:
        document, base = manifest, Path.cwd()
    if isinstance(document, list):
        document = {"samples": document}
    if not isinstance(document, dict) or not isinstance(document.get("samples"), list):
        raise ValueError("Supervised manifest must contain a samples list")
    records, identifiers = [], set()
    for index, entry in enumerate(document["samples"]):
        row = {"tensor_path": entry} if isinstance(entry, str) else dict(entry) if isinstance(entry, dict) else None
        if row is None:
            raise ValueError(f"Invalid supervised sample at index {index}")
        value = row.get("tensor_path") or row.get("path")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Sample {row.get('id', index)} has no tensor_path; preprocess audio first")
        path = Path(value).expanduser()
        path = (path if path.is_absolute() else base / path).resolve()
        if path.suffix.lower() != ".pt" or not path.is_file():
            raise FileNotFoundError(f"Prepared SFT .pt file not found: {path}")
        row["tensor_path"] = str(path)
        row["id"] = str(row.get("id") or hashlib.sha256(str(path).encode()).hexdigest()[:24])
        if row["id"] in identifiers:
            raise ValueError(f"Duplicate supervised sample id: {row['id']}")
        identifiers.add(row["id"])
        # Rich prepared manifests already carry group metadata. Old Side-Step
        # manifests only list paths, so recover metadata from the actual cache.
        if not any(row.get(key) for key in ("holdout_group", "artist_id", "group_id", "song_id")):
            import torch
            data = torch.load(path, map_location="cpu", weights_only=True)
            metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
            row["metadata"] = {**metadata, **(row.get("metadata") or {})}
        records.append(row)
    if not records:
        raise ValueError("Supervised manifest contains no prepared samples")
    return {**document, "samples": records}, base


def supervised_group_key(row: dict) -> str:
    metadata = row.get("metadata") or {}
    for key in ("holdout_group", "artist_id", "artist", "group_id", "song_id"):
        value = row.get(key) or metadata.get(key)
        if value:
            if key == "artist":
                text = unicodedata.normalize("NFKD", str(value))
                value = " ".join("".join(c for c in text if not unicodedata.combining(c)).casefold().split())
            return f"{key}:{value}"
    return "recording:" + str(row.get("audio_sha256") or metadata.get("audio_sha256")
                               or row.get("audio_path") or metadata.get("audio_path")
                               or row["tensor_path"])


def supervised_manifest_fingerprint(manifest: str | Path | dict | list) -> str:
    """Detect edits to rows, resolved paths and caches across a resumed run.

    Declared content hashes participate in the fingerprint; filesystem size and
    nanosecond mtime additionally detect replacement of caches without rehashing
    every large tensor on each launch. verify_checksums enables content checking.
    """
    document, _ = read_supervised_manifest(manifest)
    stamps = []
    for row in document["samples"]:
        path = Path(row["tensor_path"])
        stat = path.stat()
        stamps.append((str(path), stat.st_size, stat.st_mtime_ns))
    payload = {"document": document, "files": stamps, "fingerprint_version": 1}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), default=str).encode()).hexdigest()


def validate_supervised_tensor(data: dict) -> None:
    import torch
    if not isinstance(data, dict):
        raise ValueError("SFT cache must be a dictionary of tensors")
    dimensions = {"target_latents": 2, "attention_mask": 1, "encoder_hidden_states": 2,
                  "encoder_attention_mask": 1, "context_latents": 2}
    for key, ndim in dimensions.items():
        tensor = data.get(key)
        if not torch.is_tensor(tensor) or tensor.ndim != ndim:
            raise ValueError(f"{key} must be a {ndim}D tensor")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{key} contains non-finite values; regenerate this cache")
    target = data["target_latents"]
    if target.shape[0] < 1 or target.shape[1] != 64:
        raise ValueError("target_latents must have nonempty [T,64] shape")
    if data["attention_mask"].shape[0] != target.shape[0] or data["context_latents"].shape != (target.shape[0], 128):
        raise ValueError("SFT audio mask and [T,128] context must match target length")
    encoder = data["encoder_hidden_states"]
    if not encoder.shape[1] or data["encoder_attention_mask"].shape[0] != encoder.shape[0]:
        raise ValueError("Encoder states and mask must have matching nonempty lengths")
    for key in ("attention_mask", "encoder_attention_mask"):
        mask = data[key]
        if not ((mask == 0) | (mask == 1)).all() or not mask.any() or (mask[1:] > mask[:-1]).any():
            raise ValueError(f"{key} must contain a nonempty binary prefix followed by padding")


def _is_instrumental(row: dict, metadata: dict) -> bool:
    for source in (row, metadata):
        value = source.get("is_instrumental")
        if isinstance(value, bool):
            return value
    lyrics = str(row.get("lyrics", metadata.get("lyrics", ""))).strip().casefold()
    # Missing lyrics are unknown vocals, not evidence of an instrumental.
    return lyrics in ("[instrumental]", "instrumental")


class SupervisedTensorDataset:
    """Single-target dataset; no rejected branch or frozen reference required."""
    def __init__(self, manifest: str | Path | dict | list, split: str | None = None,
                 max_latent_length: int | None = None, seed: int = 42,
                 repeats: int = 1, verify_checksums: bool = False):
        self.manifest, self.base = read_supervised_manifest(manifest)
        self.records = [row for row in self.manifest["samples"]
                        if split is None or row.get("split", "train") == split]
        self.samples = self.records
        if not self.records:
            raise ValueError(f"No supervised samples found for split {split!r}")
        if repeats < 1 or (max_latent_length is not None and max_latent_length < 0):
            raise ValueError("repeats must be positive and max_latent_length nonnegative")
        self.max_latent_length = max_latent_length or None
        self.seed, self.repeats, self.epoch = int(seed), int(repeats), 0
        for row in self.records:
            path = Path(row["tensor_path"])
            if verify_checksums:
                expected = row.get("tensor_sha256") or row.get("sha256")
                if expected and _sha256(path).lower() != str(expected).lower():
                    raise ValueError(f"Tensor checksum mismatch: {path}")
            # Reject invalid caches and vocal truncation before loading a model.
            self._read(row, index=0)

    def __len__(self):
        return len(self.records) * self.repeats

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _read(self, row: dict, index: int) -> dict:
        import torch
        data = torch.load(row["tensor_path"], map_location="cpu", weights_only=True)
        validate_supervised_tensor(data)
        metadata = {**(data.get("metadata") or {}), **(row.get("metadata") or {})}
        length = int(data["attention_mask"].sum())
        encoder_length = int(data["encoder_attention_mask"].sum())
        size = min(length, self.max_latent_length or length)
        if size < length and not _is_instrumental(row, metadata):
            raise ValueError(f"SFT sample {row['id']} would truncate vocals while retaining full lyrics. "
                             "Prepare shorter audio clips with aligned clip lyrics, or use max_latent_length=0.")
        start = 0
        if size < length:
            rng = random.Random(self.seed + self.epoch * 1000003 + index * 97)
            start = rng.randrange(length - size + 1)
        result = {key: data[key] for key in TENSOR_KEYS}
        for key in ("target_latents", "attention_mask", "context_latents"):
            result[key] = result[key][start:start + size].contiguous()
        for key in ("encoder_hidden_states", "encoder_attention_mask"):
            result[key] = result[key][:encoder_length].contiguous()
        result.update(id=row["id"], group_id=supervised_group_key(row), crop_start=start,
                      metadata=metadata)
        return result

    def __getitem__(self, index: int) -> dict:
        return self._read(self.records[index % len(self.records)], index)


def split_supervised_datasets(config: dict) -> tuple[SupervisedTensorDataset, SupervisedTensorDataset | None]:
    document, _ = read_supervised_manifest(config["dataset_manifest"])
    rows = document["samples"]
    explicit = any("split" in row for row in rows)
    train, validation = [], []
    grouped_splits, file_splits = defaultdict(set), defaultdict(set)
    if explicit:
        for row in rows:
            split = row.get("split", "train")
            if split not in ("train", "validation", "val", "test"):
                raise ValueError(f"Unknown supervised split: {split!r}")
            canonical = "validation" if split in ("validation", "val") else split
            grouped_splits[supervised_group_key(row)].add(canonical)
            file_splits["tensor_path:" + row["tensor_path"]].add(canonical)
            # Equivalent copied caches/audio may have different filenames;
            # declared hashes still identify them across explicit splits.
            metadata = row.get("metadata") or {}
            for key in ("tensor_sha256", "audio_sha256", "source_audio_sha256"):
                value = row.get(key) or metadata.get(key)
                if value:
                    file_splits[key + ":" + str(value).lower()].add(canonical)
            if split != "test":
                (validation if canonical == "validation" else train).append(row)
        if any(len(splits) > 1 for splits in [*grouped_splits.values(), *file_splits.values()]):
            raise ValueError("A supervised recording/artist group occurs in multiple splits; fix holdout leakage")
    else:
        groups = sorted({supervised_group_key(row) for row in rows})
        random.Random(config["seed"]).shuffle(groups)
        count = (min(len(groups) - 1, max(1, round(len(groups) * config["validation_fraction"])))
                 if config["validation_fraction"] > 0 and len(groups) > 1 else 0)
        holdout = set(groups[:count])
        for row in rows:
            (validation if supervised_group_key(row) in holdout else train).append(row)
    if not train:
        raise ValueError("Supervised training split is empty")
    def build(selected, split, repeats):
        selected = [{**row, "split": split} for row in selected]
        return SupervisedTensorDataset({**document, "samples": selected}, split=split,
            max_latent_length=config["max_latent_length"], seed=config["seed"], repeats=repeats,
            verify_checksums=config["verify_checksums"])
    return build(train, "train", config["dataset_repeats"]), build(validation, "validation", 1) if validation else None


def collate_supervised(batch: list[dict]) -> dict:
    import torch
    if not batch:
        raise ValueError("Cannot collate an empty supervised batch")
    max_audio = max(sample["target_latents"].shape[0] for sample in batch)
    max_encoder = max(sample["encoder_hidden_states"].shape[0] for sample in batch)
    result = {}
    for key in TENSOR_KEYS:
        length = max_encoder if key.startswith("encoder_") else max_audio
        feature_shape = batch[0][key].shape[1:]
        tensors = []
        for sample in batch:
            value = sample[key]
            if value.shape[1:] != feature_shape:
                raise ValueError(f"Incompatible supervised feature dimensions: {key}")
            padded = value.new_zeros((length, *feature_shape))
            padded[:value.shape[0]] = value
            tensors.append(padded)
        result[key] = torch.stack(tensors)
    for key in ("id", "group_id", "metadata", "crop_start"):
        result[key] = [sample.get(key) for sample in batch]
    return result
