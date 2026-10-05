"""Preference data creation for JK-Step MR-FlowDPO.

No Turbo generation is needed: import existing comparisons, build MRSD pairs
from scored candidates, or make explicitly labelled synthetic acoustic pairs.
Synthetic degradations are a practical adaptation, not the paper's rewards.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif", ".opus"}
DEFAULT_AXES = ("text_alignment", "production_quality", "semantic_consistency")


def read_manifest(manifest: str | Path | dict) -> tuple[dict, Path]:
    if isinstance(manifest, dict):
        return manifest, Path.cwd()
    path = Path(manifest).expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(raw, list):
        raw = {"samples": raw}
    if not isinstance(raw, dict):
        raise ValueError("Manifest must contain a JSON object or sample list.")
    return raw, path.parent


def _resolve(value: str, base: Path) -> str:
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())


def write_manifest(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".writing")
    staging.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    staging.replace(path)
    return str(path.resolve())


def _identity(*parts: Any) -> str:
    return hashlib.sha256("\0".join(str(p) for p in parts).encode()).hexdigest()[:20]


def is_cancelled(options: dict) -> bool:
    check = options.get("cancel_check")
    event = options.get("stop_event")
    return bool((check and check()) or (event and event.is_set()))


def _split_pairs(pairs: list[dict], fraction: float, seed: int) -> None:
    """Split at song/artist level, never at individual pair level."""
    if not 0 <= fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")
    groups = sorted({str(p.get("holdout_group") or p["group_id"]) for p in pairs})
    random.Random(seed).shuffle(groups)
    count = round(len(groups) * fraction) if len(groups) > 1 else 0
    if fraction and len(groups) > 1:
        count = max(1, min(count, len(groups) - 1))
    heldout = set(groups[:count])
    for pair in pairs:
        group = str(pair.get("holdout_group") or pair["group_id"])
        pair["split"] = "validation" if group in heldout else "train"


def _read_samples(source: str | Path) -> tuple[list[dict], dict]:
    path = Path(source).expanduser().resolve()
    if path.is_dir():
        from jk_engine.data.dataset_builder import load_sidecar_metadata
        samples = []
        for audio in sorted(path.rglob("*")):
            if audio.suffix.lower() not in AUDIO_EXTENSIONS or not audio.is_file():
                continue
            meta = load_sidecar_metadata(audio)
            json_sidecar = audio.with_suffix(".json")
            if json_sidecar.is_file():
                extra = json.loads(json_sidecar.read_text(encoding="utf-8-sig"))
                if isinstance(extra, dict):
                    meta.update(extra)
            for key in ("vocal_path", "semantic_features"):
                if meta.get(key):
                    meta[key] = _resolve(meta[key], audio.parent)
            samples.append({**meta, "audio_path": str(audio), "caption": meta.get("caption", audio.stem),
                            "lyrics": meta.get("lyrics", "[Instrumental]")})
        return samples, {}
    raw, base = read_manifest(path)
    samples = raw.get("samples", raw.get("candidates", []))
    if not isinstance(samples, list):
        raise ValueError("Manifest samples/candidates must be a list.")
    normalized = []
    for row in samples:
        row = dict(row)
        audio = row.get("audio_path") or row.get("path") or row.get("filename")
        if not audio:
            raise ValueError("Every sample needs audio_path/path/filename.")
        row["audio_path"] = _resolve(audio, base)
        for key in ("vocal_path", "semantic_features"):
            if row.get(key):
                row[key] = _resolve(row[key], base)
        row.setdefault("caption", Path(audio).stem)
        row.setdefault("lyrics", "[Instrumental]")
        normalized.append(row)
    return normalized, raw.get("metadata", {})


def load_score_table(path: str | Path) -> list[dict]:
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = raw if isinstance(raw, list) else raw.get("samples", raw.get("scores", []))
    if not isinstance(rows, list):
        raise ValueError("Score table must contain rows.")
    return rows


def _merge_scores(samples: list[dict], score_file: str | None) -> None:
    if not score_file:
        return
    rows = load_score_table(score_file)
    indexed = {}
    for row in rows:
        key = row.get("audio_path") or row.get("path") or row.get("id") or row.get("filename")
        if not key:
            raise ValueError("Score rows need audio_path, path, id or filename.")
        if key in indexed:
            raise ValueError(f"Duplicate score key: {key}")
        indexed[key] = row
    for sample in samples:
        row = next((indexed[key] for key in (sample["audio_path"], sample.get("id"),
                     Path(sample["audio_path"]).name) if key in indexed), None)
        if row:
            sample.update({k: v for k, v in row.items() if k not in ("audio_path", "path", "filename")})


def _pair_record(chosen: dict, rejected: dict, primary: str | None = None) -> dict:
    caption, lyrics = str(chosen.get("caption", "")), str(chosen.get("lyrics", "[Instrumental]"))
    group = str(chosen.get("group_id") or chosen.get("prompt_id") or chosen.get("song_id")
                or _identity(caption, lyrics))
    holdout = str(chosen.get("holdout_group") or chosen.get("artist_id")
                  or chosen.get("artist") or chosen.get("song_id") or group)
    return {"id": _identity(group, chosen["audio_path"], rejected["audio_path"], primary),
            "group_id": group, "holdout_group": holdout,
            "chosen": chosen["audio_path"], "rejected": rejected["audio_path"],
            "caption": caption, "lyrics": lyrics, "primary_axis": primary,
            "chosen_scores": chosen.get("scores", {}), "rejected_scores": rejected.get("scores", {}),
            "pair_weight": 1.0,
            "metadata": {k: v for k, v in chosen.items()
                         if k not in ("audio_path", "scores", "caption", "lyrics")}}


def select_mrsd_pairs(samples: list[dict], *, axes: list[str] | tuple[str, ...] = DEFAULT_AXES,
                      primary_quantile: float = .95, secondary_quantile: float = .5,
                      minimum_quantile: float = .05, maximum_quantile: float = 1.0,
                      semantic_max_quantile: float = .99, balance_axes: bool = True,
                      max_pairs: int = 0, seed: int = 42,
                      primary_margins: dict | None = None, secondary_margins: dict | None = None,
                      min_scores: dict | None = None, max_scores: dict | None = None,
                      lower_is_better: list[str] | tuple[str, ...] = (),
                      protected_axes: dict | list | tuple = (),
                      minimum_protected_scores: dict | None = None) -> tuple[list[dict], dict]:
    """MRSD: a positive dominates every axis, strongly in one primary axis.

    Margins are computed only over candidates sharing exact conditioning.
    Percentile defaults follow §3.4; all thresholds are persisted for audit.
    A speech-fidelity score may be supplied as an additional reward axis.
    """
    import numpy as np
    axes = tuple(a.strip() for a in axes.split(",") if a.strip()) if isinstance(axes,str) else tuple(axes)
    if len(axes) < 2 or len(set(axes)) != len(axes):
        raise ValueError("MRSD needs at least two distinct reward axes; supply manual/precomputed scores.")
    for q in (primary_quantile, secondary_quantile, minimum_quantile, maximum_quantile, semantic_max_quantile):
        if not 0 <= q <= 1:
            raise ValueError("Quantiles must be between 0 and 1.")
    signs = {axis: -1 if axis in lower_is_better else 1 for axis in axes}
    protection = dict(protected_axes) if isinstance(protected_axes,dict) else {axis:0.0 for axis in protected_axes}
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in protection.values()):
        raise ValueError("Protected-axis tolerances must be finite and nonnegative.")
    protected_signs = {axis:-1 if axis in lower_is_better else 1 for axis in protection}
    if set(minimum_protected_scores or {}) - set(protection):
        raise ValueError("minimum_protected_scores keys must also be listed in protected_axes.")
    if any(not math.isfinite(float(v)) for v in (minimum_protected_scores or {}).values()):
        raise ValueError("Protected minimum scores must be finite.")
    groups: dict[str, list[dict]] = defaultdict(list)
    values = {axis: [] for axis in axes}
    for source in samples:
        sample = dict(source)
        scores = dict(sample.get("scores", {}))
        for axis in axes:
            value = scores.get(axis, sample.get(axis))
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"Missing/nonfinite reward {axis!r} for {sample.get('audio_path')}")
            scores[axis] = float(value)
            values[axis].append(signs[axis] * float(value))
        for axis in protection:
            value = scores.get(axis,sample.get(axis))
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"Missing/nonfinite protected score {axis!r} for {sample.get('audio_path')}")
            scores[axis] = float(value)
        sample["scores"] = scores
        caption, lyrics = str(sample.get("caption", "")), str(sample.get("lyrics", "[Instrumental]"))
        group = str(sample.get("group_id") or sample.get("prompt_id") or sample.get("song_id")
                    or _identity(caption, lyrics))
        sample["group_id"] = group
        groups[group].append(sample)
    if not samples:
        raise ValueError("No scored samples were found.")
    diffs = {axis: [] for axis in axes}
    for group, candidates in groups.items():
        conditions = {(str(c.get("caption", "")), str(c.get("lyrics", "[Instrumental]"))) for c in candidates}
        if len(conditions) != 1:
            raise ValueError(f"Group {group!r} mixes captions/lyrics; pair conditions must match.")
        for left, right in itertools.combinations(candidates, 2):
            for axis in axes:
                diffs[axis].append(abs(left["scores"][axis] - right["scores"][axis]))
    if not any(diffs.values()):
        raise ValueError("Each comparison group needs at least two scored candidates.")
    primary = {a: float((primary_margins or {}).get(a, np.quantile(diffs[a], primary_quantile))) for a in axes}
    secondary = {a: float((secondary_margins or {}).get(a, np.quantile(diffs[a], secondary_quantile))) for a in axes}
    if any(not math.isfinite(v) or v < 0 for v in (*primary.values(), *secondary.values())):
        raise ValueError("Primary and secondary margins must be finite and nonnegative.")
    minima = {a: float((min_scores or {}).get(a, np.quantile(values[a], minimum_quantile))) for a in axes}
    maxima = {a: float((max_scores or {}).get(a, np.quantile(values[a],
              semantic_max_quantile if a == "semantic_consistency" else maximum_quantile))) for a in axes}
    buckets: dict[str, list[dict]] = {a: [] for a in axes}
    excluded = 0
    for candidates in groups.values():
        valid = [c for c in candidates if all(minima[a] <= signs[a]*c["scores"][a] <= maxima[a] for a in axes)]
        excluded += len(candidates) - len(valid)
        for winner, loser in itertools.permutations(valid, 2):
            if any(protected_signs[a]*(winner['scores'][a]-loser['scores'][a]) < -float(tolerance)
                   for a,tolerance in protection.items()):
                continue
            if any(protected_signs[a]*winner['scores'][a] < float(minimum)
                   for a,minimum in (minimum_protected_scores or {}).items()):
                continue
            margins = {a: signs[a]*(winner["scores"][a] - loser["scores"][a]) for a in axes}
            for axis in axes:
                if margins[axis] > primary[axis] and all(margins[a] > secondary[a] for a in axes if a != axis):
                    buckets[axis].append(_pair_record(winner, loser, axis))
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    if balance_axes:
        size = min(map(len, buckets.values()))
        pairs = [p for axis in axes for p in buckets[axis][:size]]
    else:
        pairs = [p for axis in axes for p in buckets[axis]]
    rng.shuffle(pairs)
    if max_pairs > 0:
        if balance_axes:
            quota = max_pairs // len(axes)
            pairs = [p for axis in axes for p in buckets[axis][:min(size, quota)]]
            rng.shuffle(pairs)
        else:
            pairs = pairs[:max_pairs]
    return pairs, {"algorithm": "MRSD", "axes": list(axes), "primary_margins": primary,
                   "secondary_margins": secondary, "minimum_scores_oriented": minima,
                   "maximum_scores_oriented": maxima, "candidates_per_axis": {a:len(v) for a,v in buckets.items()},
                   "excluded_outliers": excluded, "balanced": balance_axes, "lower_is_better": list(lower_is_better),
                   "protected_axes":protection,"minimum_protected_scores":minimum_protected_scores or {}}


def _degrade(audio, sample_rate: int, *, kind: str, rng, cutoff_hz: float,
             noise_snr_db: float, clip_threshold: float, lowpass_order: int = 4):
    import numpy as np
    from scipy.signal import butter, sosfiltfilt
    if kind in ("lowpass", "vocal_lowpass"):
        if not 0 < cutoff_hz < sample_rate / 2:
            raise ValueError("cutoff_hz must lie below the audio Nyquist frequency.")
        return sosfiltfilt(butter(lowpass_order, cutoff_hz, fs=sample_rate, output="sos"), audio, axis=0).astype("float32")
    if kind == "noise":
        noise = rng.standard_normal(audio.shape).astype("float32")
        signal_rms = max(float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))), 1e-8)
        noise *= signal_rms / (10 ** (noise_snr_db / 20))
        return (audio + noise).astype("float32")
    if kind == "clipping":
        if not 0 < clip_threshold <= 1:
            raise ValueError("clip_threshold must be in (0, 1].")
        return np.clip(audio, -clip_threshold, clip_threshold).astype("float32")
    raise ValueError(f"Unknown degradation {kind!r}; use lowpass, vocal_lowpass, noise or clipping.")


def _synthetic_pairs(samples: list[dict], output: Path, options: dict) -> list[dict]:
    import numpy as np
    import soundfile as sf
    seed = int(options.get("seed", 42))
    rng = np.random.default_rng(seed)
    kinds = options.get("degradation", "lowpass")
    kinds = [kinds] if isinstance(kinds, str) else list(kinds)
    variations = int(options.get("variations_per_audio", 1))
    if variations < 1:
        raise ValueError("variations_per_audio must be positive.")
    if not kinds:
        raise ValueError("Choose at least one degradation.")
    rejected_dir = output / "rejected_audio"
    rejected_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for i, sample in enumerate(samples):
        if is_cancelled(options):
            break
        audio, sr = sf.read(sample["audio_path"], dtype="float32", always_2d=True)
        if not np.isfinite(audio).all() or len(audio) == 0:
            raise ValueError(f"Invalid audio: {sample['audio_path']}")
        sample = dict(sample)
        sample["song_id"] = sample.get("song_id") or _identity(sample["audio_path"])
        for v in range(variations):
            kind = kinds[v % len(kinds)]
            source = audio
            if kind == "vocal_lowpass":
                vocal_path = sample.get("vocal_path")
                if not vocal_path and options.get("vocal_stems_dir"):
                    vocal_path = str(Path(options["vocal_stems_dir"]) / (Path(sample["audio_path"]).stem + ".wav"))
                if not vocal_path:
                    raise ValueError("vocal_lowpass needs vocal_path per sample or vocal_stems_dir; no automatic separation is implied.")
                source, stem_sr = sf.read(vocal_path, dtype="float32", always_2d=True)
                if stem_sr != sr or source.shape != audio.shape:
                    raise ValueError("Vocal stem and original must have identical sample rate, channel count and length.")
            degraded = _degrade(source, sr, kind=kind, rng=rng,
                cutoff_hz=float(options.get("cutoff_hz", 6000)),
                noise_snr_db=float(options.get("noise_snr_db", 24)),
                clip_threshold=float(options.get("clip_threshold", .15)),
                lowpass_order=int(options.get("lowpass_order", 4)))
            if kind == "vocal_lowpass":
                degraded = audio - source + degraded
            path = rejected_dir / f"{_identity(sample['audio_path'], seed, kind, v)}.wav"
            sf.write(path, degraded, sr, subtype="FLOAT")
            loser = {**sample, "audio_path": str(path.resolve())}
            pair = _pair_record(sample, loser, "synthetic_acoustic_quality")
            pair["metadata"].update({"pair_source": "synthetic_acoustic_degradation", "degradation": kind,
                                     "seed": seed, "original_quality_unverified": True})
            pairs.append(pair)
        callback = options.get("progress_callback")
        if callback:
            callback(i + 1, len(samples), f"Created acoustic pairs: {Path(sample['audio_path']).name}")
    return pairs


def build_preference_pairs(input_manifest_or_audio_dir: str, output_dir: str, **options) -> dict:
    """Return a report including ``manifest`` and ``pairs`` count.

    mode: auto/import/scored/degraded. No model/audio generation occurs here.
    Every scored group must share caption and lyrics; pairs from different songs
    must be deliberately grouped in the candidate manifest, not guessed.
    """
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(input_manifest_or_audio_dir).expanduser().resolve()
    mode = options.get("mode", "auto")
    raw, base = ({}, source) if source.is_dir() else read_manifest(source)
    if mode == "auto":
        scored = options.get("score_file") or options.get("scores_file") or any(
            isinstance(row,dict) and bool(row.get("scores")) for row in raw.get("samples",raw.get("candidates",[])))
        mode = "import" if "pairs" in raw else ("scored" if scored else "degraded")
    metadata = {"trainer": "JK-Step MR-FlowDPO", "version": 1, "mode": mode,
                "source": str(source), "seed": int(options.get("seed", 42))}
    if mode == "import":
        if not isinstance(raw.get("pairs"), list):
            raise ValueError("Import mode requires a JSON manifest containing pairs.")
        pairs = []
        for index, record in enumerate(raw["pairs"]):
            pair = dict(record)
            for key in ("chosen", "rejected", "tensor_path"):
                if pair.get(key):
                    pair[key] = _resolve(pair[key], base)
            pair.setdefault("caption", "")
            pair.setdefault("lyrics", "[Instrumental]")
            pair.setdefault("group_id", _identity(pair["caption"], pair["lyrics"]))
            pair.setdefault("id", _identity(pair.get("chosen"), pair.get("rejected"), index))
            pair.setdefault("pair_weight", 1.0)
            pair.setdefault("metadata", {})
            pairs.append(pair)
        metadata["source_metadata"] = raw.get("metadata", {})
    else:
        samples, ds_meta = _read_samples(source)
        if not samples:
            raise ValueError("No audio samples found.")
        for sample in samples:
            if not Path(sample["audio_path"]).is_file():
                raise FileNotFoundError(sample["audio_path"])
        metadata["source_metadata"] = ds_meta
        if mode == "degraded":
            pairs = _synthetic_pairs(samples, output, options)
            metadata["scientific_scope"] = "Synthetic acoustic preference adaptation; not automatic MRSD reward evaluation."
        elif mode == "scored":
            _merge_scores(samples, options.get("score_file") or options.get("scores_file"))
            keys = {"axes", "primary_quantile", "secondary_quantile", "minimum_quantile", "maximum_quantile",
                    "semantic_max_quantile", "balance_axes", "max_pairs", "seed", "primary_margins",
                    "secondary_margins", "min_scores", "max_scores", "lower_is_better"}
            keys.update(("protected_axes","minimum_protected_scores"))
            pairs, report = select_mrsd_pairs(samples, **{k:v for k,v in options.items() if k in keys})
            metadata["selection"] = report
            if not pairs:
                raise ValueError("No MRSD pairs satisfy all reward margins/outlier filters. Inspect scores, groups or relax thresholds; no pairs are invented.")
        else:
            raise ValueError(f"Unknown pair mode: {mode}")
    if not pairs:
        raise ValueError("No preference pairs were produced.")
    if mode != "import" or options.get("resplit", False) or any("split" not in p for p in pairs):
        _split_pairs(pairs, float(options.get("validation_fraction", .1)), int(options.get("seed", 42)))
    payload = {"version": 1, "metadata": metadata, "pairs": pairs}
    validation = validate_pairs(payload, check_audio=bool(options.get("check_audio", True)))
    if not validation["valid"]:
        raise ValueError("Invalid pairs: " + "; ".join(validation["errors"][:5]))
    path = write_manifest(output / "pairs.json", payload)
    return {"manifest": path, "pairs": len(pairs), "train": sum(p["split"] == "train" for p in pairs),
            "validation": sum(p["split"] == "validation" for p in pairs), "metadata": metadata,
            "warnings": validation["warnings"]}


def validate_pairs(manifest: str | Path | dict, *, check_audio: bool = True, check_tensors: bool = False) -> dict:
    raw, base = read_manifest(manifest)
    pairs = raw.get("pairs", [])
    errors, warnings, ids, groups, paths = [], [], set(), defaultdict(set), defaultdict(set)
    if not isinstance(pairs, list) or not pairs:
        return {"valid": False, "pairs": 0, "errors": ["Manifest pairs must be a nonempty list."], "warnings": []}
    infos = {}
    for i, pair in enumerate(pairs):
        label = str(pair.get("id", i))
        if label in ids:
            errors.append(f"Duplicate pair id: {label}")
        ids.add(label)
        group = str(pair.get("holdout_group") or pair.get("group_id", ""))
        split = pair.get("split", "train")
        if split not in ("train", "validation", "test"):
            errors.append(f"{label}: split must be train, validation or test.")
        if not group:
            errors.append(f"{label}: missing group_id.")
        groups[group].add(split)
        try:
            weight = float(pair.get("pair_weight", 1))
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError
        except (ValueError, TypeError):
            errors.append(f"{label}: pair_weight must be finite and positive.")
        branch_paths = []
        for branch in ("chosen", "rejected"):
            value = pair.get(branch)
            if not isinstance(value, str) or not value:
                errors.append(f"{label}: missing {branch} audio.")
                continue
            path = _resolve(value, base)
            branch_paths.append(path)
            paths[path].add(split)
            if not Path(path).is_file():
                errors.append(f"{label}: missing file {path}")
            elif check_audio:
                try:
                    import soundfile as sf
                    if path not in infos:
                        infos[path] = sf.info(path)
                    if infos[path].frames <= 0:
                        errors.append(f"{label}: empty audio {path}")
                except Exception as exc:
                    errors.append(f"{label}: unreadable audio {path}: {exc}")
        if len(branch_paths) == 2:
            if branch_paths[0] == branch_paths[1]:
                errors.append(f"{label}: chosen and rejected are the same file.")
            if all(p in infos for p in branch_paths):
                left, right = (infos[p] for p in branch_paths)
                if abs(left.duration - right.duration) > max(1/left.samplerate, 1/right.samplerate) * 2:
                    errors.append(f"{label}: pair durations differ; align/crop the same time window before preprocessing.")
        if pair.get("tensor_path") and not Path(_resolve(pair["tensor_path"], base)).is_file():
            errors.append(f"{label}: tensor_path does not exist.")
        elif check_tensors and pair.get("tensor_path"):
            try:
                from .preprocess import validate_tensor_pair
                import torch
                data = torch.load(_resolve(pair["tensor_path"], base), map_location="cpu", weights_only=True)
                validate_tensor_pair(data)
            except Exception as exc:
                errors.append(f"{label}: invalid tensor pair: {exc}")
        if not str(pair.get("caption", "")).strip():
            warnings.append(f"{label}: empty caption.")
        if pair.get("metadata", {}).get("original_quality_unverified"):
            warnings.append(f"{label}: synthetic preference; original quality/diction require curation.")
    for group, splits in groups.items():
        if len(splits) > 1:
            errors.append(f"Holdout leakage: group {group!r} appears in {sorted(splits)}.")
    for path, splits in paths.items():
        if len(splits) > 1:
            errors.append(f"Audio leakage across splits: {path}")
    return {"valid": not errors, "pairs": len(pairs), "groups": len(groups),
            "splits": {s: sum(p.get('split', 'train') == s for p in pairs) for s in ('train','validation','test')},
            "errors": errors, "warnings": warnings}
