"""Prepare a music folder for supervised or explicit preference LoRA.

Generated audio/metadata stays outside the source directory. Automatic captions,
transcripts and times are provenance-labelled estimates, not verified labels.
Preference mode labels its controlled synthetic comparisons explicitly.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import unicodedata

from .audio_annotation import (ANNOTATION_VERSION, LocalAudioAnnotator,
    MODEL_REVISIONS, CAPTION_PROMPT, CAPTION_SYSTEM_PROMPT, PreparationCancelled, check_stop, read_audio, validate_phrases)

DEFAULTS = {"output_dir": "", "dataset_name": "", "checkpoint_dir": "checkpoints",
    "model_variant": "xl-sft", "device": "auto", "precision": "bf16",
    "caption_model": "Qwen/Qwen2.5-Omni-7B", "caption_tier": "auto",
    "lyrics_model": "openai/whisper-large-v3", "language": "auto", "content_mode": "auto",
    "clip_seconds": 30.0, "min_clip_seconds": 8.0, "custom_tag": "", "normalize": "none", "allow_download": True,
    "reuse": True, "preprocess": True, "validation_fraction": .1, "seed": 42,
    "objective": "sft", "pair_options": {}}
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".wma", ".aif", ".aiff"}


def inspect_folder(audio_dir):
    """Lightweight folder inventory; no torch imports, models or network."""
    folder = Path(audio_dir).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError("Audio folder does not exist")
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)
    metadata = Counter()
    examples = []
    for p in files:
        try:
            fields = _read_source_metadata(p)
            metadata["caption"] += bool(fields.get("caption"))
            metadata["lyrics"] += bool(fields.get("lyrics"))
            metadata["any"] += bool(fields)
            if len(examples) < 5:
                examples.append({"path": str(p), "caption": str(fields.get("caption", ""))[:180],
                                 "has_lyrics": bool(fields.get("lyrics"))})
        except (OSError, ValueError) as error:
            metadata["unreadable"] += 1
            if len(examples) < 5:
                examples.append({"path": str(p), "error": str(error)})
    return {"audio_dir": str(folder), "count": len(files), "supported_files": [str(p) for p in files],
            "existing_metadata": dict(metadata), "examples": examples}


def _read_source_metadata(audio_path):
    """Pure text reader; an empty structured sidecar is never mistaken for lyrics."""
    audio_path = Path(audio_path)
    plain = audio_path.with_suffix(".txt")
    text = plain.read_text(encoding="utf-8-sig") if plain.is_file() else ""
    keys = {"caption", "genre", "bpm", "key", "keyscale", "signature", "timesignature",
            "lyrics", "is_instrumental", "artist", "language", "custom_tag", "repeat", "prompt_override"}
    fields, current, structured = {}, None, False
    for line in text.splitlines():
        if current == "lyrics":
            fields[current] += "\n" + line
            continue
        match = re.match(r"^\s*([a-z_]+)\s*:\s*(.*)$", line, re.I)
        if match and match[1].lower() in keys:
            current = match[1].lower();fields[current] = match[2];structured = True
        elif current:
            fields[current] += "\n" + line
    if text.strip() and not structured:
        fields["lyrics"] = text.strip()
    for key in ("caption", "lyrics"):
        separate = audio_path.with_suffix("." + key + ".txt")
        if not str(fields.get(key, "")).strip() and separate.is_file():
            fields[key] = separate.read_text(encoding="utf-8-sig").strip()
    if str(fields.get("caption", "")).strip().lower() in ("n/a", "none", "null", "unknown"):
        fields["caption"] = ""
    if str(fields.get("lyrics", "")).strip().lower() in ("n/a", "none", "null"):
        fields["lyrics"] = ""
    return {k: str(v).strip() for k,v in fields.items()}


def _hash_file(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        staged = Path(f.name)
        try:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        except BaseException:
            f.close(); staged.unlink(missing_ok=True); raise
    try:
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def validate_options(config):
    options = DEFAULTS | dict(config)
    if options['objective'] not in ('sft', 'flow_dpo'):
        raise ValueError('objective must be sft or flow_dpo')
    if not isinstance(options['pair_options'], dict):
        raise ValueError('pair_options must be an object')
    if options['objective'] == 'flow_dpo':
        from .folder_preferences import preference_options
        options['pair_options'] = preference_options(options['pair_options'])
    if not str(options.get("audio_dir", "")).strip():
        raise ValueError("audio_dir is required")
    source = Path(options["audio_dir"]).expanduser().resolve()
    if not source.is_dir():
        raise ValueError("Audio folder does not exist")
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", source.name).strip("_")[:64] or "music"
    if name.upper() in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"(?:COM|LPT)[1-9]", name.upper()):
        name = "music_" + name
    source_key = hashlib.sha256(os.path.normcase(str(source)).encode("utf-8")).hexdigest()[:12]
    destination = Path(options["output_dir"] or Path("datasets/generated") / f"{name}_{source_key}").expanduser().resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError("Output must be outside the source audio folder; source files are never overwritten")
    options.update(audio_dir=str(source), output_dir=str(destination), dataset_name=options["dataset_name"] or source.name)
    options["clip_seconds"] = float(options["clip_seconds"])
    options["min_clip_seconds"] = float(options["min_clip_seconds"])
    options["validation_fraction"] = float(options["validation_fraction"])
    if not math.isfinite(options["clip_seconds"]) or not 10 <= options["clip_seconds"] <= 120:
        raise ValueError("clip_seconds must be between 10 and 120")
    if not math.isfinite(options["min_clip_seconds"]) or not 1 <= options["min_clip_seconds"] <= options["clip_seconds"]:
        raise ValueError("min_clip_seconds must be finite, at least 1 and no greater than clip_seconds")
    if not 0 <= options["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if options["content_mode"] not in ("auto", "vocal", "instrumental"):
        raise ValueError("content_mode must be auto, vocal or instrumental")
    if options["caption_tier"] not in ("auto", "8gb", "16gb", "8-10gb"):
        raise ValueError("Invalid caption_tier")
    if options["normalize"] != "none":
        raise ValueError("Folder preparation preserves amplitude; normalize must be none")
    if options["model_variant"] not in ("xl-sft", "sft", "xl-base", "base"):
        raise ValueError("Folder supervision requires an SFT/base model variant")
    for key in ("allow_download", "reuse", "preprocess"):
        if not isinstance(options[key], bool):
            raise ValueError(f"{key} must be a boolean")
    return options


def _words(text):
    # Alignment-only normalization. Returned training text stays original.
    spans = list(re.finditer(r"[^\W_]+(?:['’][^\W_]+)*", text, re.UNICODE))
    def normalized(word):
        return "".join(c for c in unicodedata.normalize("NFKD", word.casefold().replace("’", "'")) if not unicodedata.combining(c))
    return [(normalized(m.group()), m.start(), m.end()) for m in spans]


def preserve_human_lyrics(phrases, lyrics):
    """Use human text only when ASR supplies an exact monotonic word alignment.

    Disagreement is quarantined, rather than replacing human words with guesses.
    Section labels are not sung words and are excluded from the alignment copy.
    """
    text = re.sub(r"(?m)^\s*\[[^\]\n]+\]\s*$", "", lyrics)
    words = _words(text)
    automatic = [_words(p["text"]) for p in phrases]
    if [w[0] for w in words] != [w[0] for group in automatic for w in group]:
        raise ValueError("Human lyrics disagree with automatic transcript; retain full source lyrics and quarantine until alignment is corrected")
    result, cursor = [], 0
    for phrase, group in zip(phrases, automatic):
        if not group:
            continue
        first = words[cursor][1]
        cursor += len(group)
        last = words[cursor][1] if cursor < len(words) else len(text)
        result.append({**phrase, "text": text[first:last].strip(), "text_source": "user sidecar; exact word agreement with automatic ASR timing"})
    return result


def phrase_groups(phrases, duration, limit, min_seconds=8):
    groups, rejected, current = [], [], []
    def finish():
        nonlocal current
        if current:
            span = current[-1]["end"] - current[0]["start"]
            if span + .5 >= min_seconds:
                groups.append(current)
            else:
                rejected.append({"reason": f"Complete phrase group shorter than {min_seconds:g} seconds", "phrases": current})
            current = []
    for phrase in phrases:
        if phrase["end"] - phrase["start"] > limit:
            finish(); rejected.append({"reason": "Complete ASR phrase exceeds clip_seconds; no word boundary invented", "phrases": [phrase]}); continue
        if current and (phrase["start"] - current[-1]["end"] > 5 or phrase["end"] - current[0]["start"] > limit):
            finish()
        current.append(phrase)
    finish()
    clips = []
    for group in groups:
        first, last = group[0]["start"], group[-1]["end"]
        pad = min(.25, max(0, (limit - (last - first)) / 2))
        before = max((p["end"] for p in phrases if p["end"] <= first), default=0)
        after = min((p["start"] for p in phrases if p["start"] >= last), default=duration)
        clips.append({"start": max(0, before, first - pad), "end": min(duration, after, last + pad),
                      "lyrics": "\n".join(p["text"] for p in group), "phrases": group})
    return clips, rejected


def _instrumental_clips(duration, limit, min_seconds=8):
    clips, start = [], 0.0
    while start < duration:
        end = min(duration, start + limit)
        if end - start >= min_seconds:
            clips.append({"start": start, "end": end, "lyrics": "[Instrumental]", "phrases": []})
        start = end
    return clips


def _write_clip(path, audio, sample_rate):
    import numpy as np
    import soundfile as sf
    floating = float(np.max(np.abs(audio))) > 1
    path = Path(path).with_suffix(".wav" if floating else ".flac")
    format_, subtype = ("WAV", "FLOAT") if floating else ("FLAC", "PCM_24")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        actual, sr = sf.read(path, dtype="float64", always_2d=True)
        if sr != sample_rate or actual.shape != audio.shape:
            raise ValueError("Cached crop shape does not match its source")
        good = np.array_equal(actual, audio.astype(np.float32).astype(np.float64)) if floating else np.max(np.abs(actual - audio)) <= 2 ** -23 + 1e-10
        if not good or sf.info(path).subtype != subtype:
            raise ValueError("Cached crop samples changed; refusing overwrite")
    else:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            staging = Path(f.name)
        try:
            sf.write(staging, audio, sample_rate, format=format_, subtype=subtype)
            os.replace(staging, path)
        finally:
            staging.unlink(missing_ok=True)
    return path, {"format": format_, "subtype": subtype, "sha256": _hash_file(path),
                  "sample_rate": sample_rate, "channels": audio.shape[1], "frames": len(audio),
                  "duration_seconds": len(audio) / sample_rate, "gain_applied": False, "clipped_samples": 0}


def assign_splits(samples, fraction=.1, seed=42):
    # Container tags/timestamps can differ while decoded audio is identical.
    # Join the corresponding artist groups before splitting source aliases.
    parent = {s["holdout_group"]: s["holdout_group"] for s in samples}
    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]];value = parent[value]
        return value
    songs = {}
    for sample in samples:
        group = sample["holdout_group"]
        song = sample.get("source_pcm_sha256") or sample.get("song_id", group)
        if song in songs:
            first, second = find(group), find(songs[song])
            parent[max(first, second)] = min(first, second)
        else:
            songs[song] = group
    for sample in samples:
        sample["holdout_group"] = find(sample["holdout_group"])
    groups = sorted({s["holdout_group"] for s in samples})
    random.Random(seed).shuffle(groups)
    count = min(len(groups) - 1, max(1, round(len(groups) * fraction))) if fraction and len(groups) > 1 else 0
    held = set(groups[:count])
    for sample in samples:
        sample["split"] = "validation" if sample["holdout_group"] in held else "train"


def validate_supervised_tensor(path):
    import torch
    data = torch.load(path, map_location="cpu", weights_only=True)
    required = ("target_latents", "attention_mask", "encoder_hidden_states", "encoder_attention_mask", "context_latents")
    if any(not torch.is_tensor(data.get(k)) for k in required):
        raise ValueError("Supervised tensor is missing required ACE conditioning/target fields")
    for key in required:
        if not torch.isfinite(data[key]).all():
            raise ValueError(f"Nonfinite tensor field {key}")
    latent, mask, hidden, hmask, context = (data[k] for k in required)
    if latent.ndim != 2 or latent.shape[0] < 1 or latent.shape[1] != 64 or mask.shape != (len(latent),):
        raise ValueError("Invalid latent target/mask shape")
    if hidden.ndim != 2 or hidden.shape[0] < 1 or hmask.shape != (len(hidden),):
        raise ValueError("Invalid encoded conditioning/mask shape")
    if context.ndim != 2 or context.shape != (len(latent), 128):
        raise ValueError("Invalid ACE context shape")
    if not (mask > 0).any() or not (hmask > 0).any():
        raise ValueError("Empty attention masks")
    return {"latent_frames": len(latent), "conditioning_tokens": len(hidden)}


def _model_fingerprint(options):
    root = Path(options["checkpoint_dir"]).resolve()
    entries = []
    for folder in (root / ("acestep-v15-" + options["model_variant"]), root / "vae", root / "Qwen3-Embedding-0.6B"):
        if folder.is_dir():
            for p in sorted(folder.iterdir()):
                if p.is_file() and (p.suffix in (".json", ".py", ".safetensors", ".bin", ".pt")):
                    entries.append([str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns,
                                    _hash_file(p) if p.suffix in (".json", ".py") else None])
    return _fingerprint([str(root), options["model_variant"], entries])


def _selected_inputs(destination, samples, metadata, key, stop_event=None):
    """Isolate accepted clips from quarantined/stale managed audio before encoding.

    Hardlinks are only between generated copies; source recordings are never
    linked or edited. A cancelled staging directory is not a published input.
    """
    selected = Path(destination) / "selected_inputs" / key
    selected_samples = [{**s, "audio_path": str(selected / s["filename"])} for s in samples]
    selected_dataset = {"version": 1, "metadata": metadata, "samples": selected_samples}
    expected = {s["filename"] for s in samples} | {Path(s["filename"]).with_suffix(".txt").name for s in samples} | {"dataset.json"}
    selected.parent.mkdir(parents=True, exist_ok=True)
    if not selected.exists():
        staging = Path(tempfile.mkdtemp(prefix=".staging_", dir=selected.parent))
        try:
            for sample in samples:
                check_stop(stop_event)
                source = Path(sample["audio_path"])
                target = staging / source.name
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copyfile(source, target)
                shutil.copyfile(source.with_suffix(".txt"), target.with_suffix(".txt"))
            _write_json(staging / "dataset.json", selected_dataset)
            check_stop(stop_event)
            os.replace(staging, selected)
        finally:
            if staging.exists():
                # This exact, newly-created directory is inside selected_inputs.
                if not staging.resolve().is_relative_to(selected.parent.resolve()):
                    raise ValueError("Refusing cleanup outside the managed selected-input directory")
                shutil.rmtree(staging)
    if not selected.is_dir() or {p.name for p in selected.iterdir()} != expected:
        raise ValueError("Selected preprocessing inputs contain unexpected or missing files")
    if json.loads((selected / "dataset.json").read_text(encoding="utf-8")) != selected_dataset:
        raise ValueError("Selected preprocessing manifest differs from the approved samples")
    for sample in samples:
        check_stop(stop_event)
        target = selected / sample["filename"]
        if _hash_file(target) != sample["sha256"]:
            raise ValueError("Selected preprocessing audio differs from the prepared clip")
        if _hash_file(target.with_suffix(".txt")) != _hash_file(Path(sample["audio_path"]).with_suffix(".txt")):
            raise ValueError("Selected preprocessing sidecar differs from the prepared metadata")
    return selected, selected / "dataset.json"


def prepare_folder(config, progress=None, stop_event=None):
    """Local one-folder pipeline. GPU/download validation is separate from CPU fixtures."""
    from jk_engine.data.sidecar_io import write_sidecar
    options = validate_options(config)
    source, destination = Path(options["audio_dir"]), Path(options["output_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    annotator = LocalAudioAnnotator(options, progress, stop_event)
    candidates, samples, rows, quarantine, sources = [], [], [], [], []
    preferences = {}
    tensor_dir, selected_dir = None, None
    def emit(stage, message, current=None, total=None):
        check_stop(stop_event)
        if progress:
            progress({"event": "progress", "stage": stage, "message": message,
                      "current": current, "total": total})
    cancelled = False
    try:
        files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)
        if not files:
            raise ValueError("No supported audio files found")
        for number, path in enumerate(files, 1):
            emit("transcription", f"Reading {path.name}", number - 1, len(files))
            identity = _hash_file(path)
            meta = _read_source_metadata(path)
            source_record = {"audio_path": str(path), "sha256": identity, "user_sidecar_metadata": meta}
            sources.append(source_record)
            try:
                audio, sr = read_audio(path)
                import numpy as np
                pcm = hashlib.sha256(json.dumps([int(sr), list(audio.shape)]).encode())
                pcm.update(memoryview(np.ascontiguousarray(audio, dtype='<f4')).cast('B'))
                pcm_identity = pcm.hexdigest()
                source_record['pcm_sha256'] = pcm_identity
                duration = len(audio) / sr
                if duration < options["min_clip_seconds"]:
                    raise ValueError(f"Source audio shorter than {options['min_clip_seconds']:g} seconds")
                lyrics = str(meta.get("lyrics", "")).strip()
                declared_instrumental = str(meta.get("is_instrumental", "")).lower() in ("true", "yes", "1") or lyrics.lower() == "[instrumental]"
                instrumental = options["content_mode"] == "instrumental" or (options["content_mode"] == "auto" and declared_instrumental)
                if instrumental:
                    if lyrics and lyrics.lower() != "[instrumental]":
                        raise ValueError("Instrumental selection conflicts with existing sung lyrics")
                    clips = _instrumental_clips(duration, options["clip_seconds"], options["min_clip_seconds"])
                    label_source = "user instrumental selection/declaration"
                elif lyrics and duration <= options["clip_seconds"]:
                    clips = [{"start": 0, "end": duration, "lyrics": lyrics, "phrases": []}]
                    label_source = "user full-clip lyrics; no automatic timing required"
                else:
                    asr_key = _fingerprint([ANNOTATION_VERSION, identity, options["lyrics_model"], MODEL_REVISIONS.get(options["lyrics_model"]), options["language"],
                                            options["device"], options["precision"], options["clip_seconds"], options["min_clip_seconds"]])
                    cache = destination / "annotation_cache" / f"asr_{asr_key}.json"
                    if options["reuse"] and cache.is_file():
                        saved = json.loads(cache.read_text(encoding="utf-8"))
                        if saved["key"] != asr_key:
                            raise ValueError("ASR cache identity mismatch")
                        phrases = validate_phrases(saved["phrases"], duration)
                    else:
                        phrases = annotator.transcribe(audio, sr)
                        _write_json(cache, {"key": asr_key, "phrases": phrases, "origin": annotator.origin})
                    if lyrics:
                        phrases = preserve_human_lyrics(phrases, lyrics)
                    label_source = "user text with exact ASR word agreement" if lyrics else "automatic Whisper transcription"
                    if phrases:
                        clips, rejected = phrase_groups(phrases, duration, options["clip_seconds"], options["min_clip_seconds"])
                        quarantine.extend({"source_audio_path": str(path), **r} for r in rejected)
                    elif options["content_mode"] == "auto":
                        # No transcript is not evidence of instrumental music.
                        # The caption phase must explicitly confirm no vocals.
                        clips = _instrumental_clips(duration, options["clip_seconds"], options["min_clip_seconds"])
                        label_source = "no ASR words; pending independent audio vocal classification"
                    else:
                        raise ValueError("No intelligible transcript for a vocal recording")
                if _hash_file(path) != identity:
                    raise ValueError("Source file changed while being decoded")
                try:
                    from jk_engine.data.audio_metadata import resolve_metadata
                    artist = str(meta.get("artist") or resolve_metadata(path).artist or "")
                except Exception:
                    artist = str(meta.get("artist", ""))
                canonical = " ".join("".join(c for c in unicodedata.normalize("NFKD", artist.casefold()) if not unicodedata.combining(c)).split())
                holdout = "artist:" + hashlib.sha256(canonical.encode()).hexdigest()[:24] if canonical else "song:" + identity[:24]
                for clip in clips:
                    check_stop(stop_event)
                    first, last = max(0, math.floor(clip["start"] * sr)), min(len(audio), math.ceil(clip["end"] * sr))
                    if last - first < options["min_clip_seconds"] * sr or last - first > math.ceil(options["clip_seconds"] * sr):
                        quarantine.append({"source_audio_path": str(path), "reason": "Neighbor-bounded complete phrase crop outside duration limit", "clip": clip}); continue
                    uid = "clip_" + _fingerprint([identity, first, last, meta])[:24]
                    target, info = _write_clip(destination / "audio" / (uid + ".flac"), audio[first:last], sr)
                    candidates.append({"id": uid, "audio_path": str(target), "filename": target.name,
                        "source_audio_path": str(path), "source_audio_sha256": identity,
                        "source_pcm_sha256": pcm_identity, "source_sidecar_metadata": meta,
                        "audio_info": info, "sha256": info["sha256"], "duration": (last - first) / sr,
                        "crop_start_frame": first, "crop_end_frame": last, "crop_start": first / sr, "crop_end": last / sr,
                        "lyrics": clip["lyrics"], "lyrics_source": label_source, "phrases": clip["phrases"],
                        "timestamps_human_verified": False, "transcription_human_verified": False,
                        "lyrics_user_provided": bool(lyrics),
                        "artist": artist, "holdout_group": holdout, "song_id": "song:" + identity[:24],
                        "is_instrumental": instrumental, "vocal_classification_required": options["content_mode"] == "auto" and not declared_instrumental and not lyrics,
                        "language": options["language"] if options["language"] != "auto" else "unknown"})
                emit("transcription", f"Read/transcribed {path.name}", number, len(files))
            except PreparationCancelled:
                raise
            except Exception as error:
                quarantine.append({"source_audio_path": str(path), "reason": str(error), "entire_song": True})
        annotator.unload_asr()
        unique = {c["id"]: c for c in candidates}
        for number, sample in enumerate(unique.values(), 1):
            emit("caption", f"Annotating {sample['filename']}", number - 1, len(unique))
            try:
                meta = sample["source_sidecar_metadata"]
                cap_key = _fingerprint([ANNOTATION_VERSION, sample["sha256"], options["caption_model"], MODEL_REVISIONS.get(options["caption_model"]),
                                        CAPTION_PROMPT, CAPTION_SYSTEM_PROMPT, options["caption_tier"], options["device"], options["precision"], meta,
                                        sample["lyrics"], sample["vocal_classification_required"], options["clip_seconds"], options["min_clip_seconds"]])
                cache = destination / "annotation_cache" / f"caption_{cap_key}.json"
                if options["reuse"] and cache.is_file():
                    saved = json.loads(cache.read_text(encoding="utf-8"))
                    if saved["key"] != cap_key:
                        raise ValueError("Caption cache identity mismatch")
                    fields = saved["fields"]
                elif meta.get("caption") and not sample["vocal_classification_required"]:
                    fields = {"caption": str(meta["caption"]), "genre": meta.get("genre", ""),
                              "vocal_status": "no_vocals" if sample["is_instrumental"] else "singing"}
                else:
                    fields = annotator.caption(sample["audio_path"])
                    _write_json(cache, {"key": cap_key, "fields": fields, "origin": annotator.origin})
                if not str(fields.get("caption", "")).strip():
                    raise ValueError("No usable caption; filename fallback is disabled")
                status = fields.get("vocal_status", "uncertain")
                if sample["vocal_classification_required"]:
                    if status == "uncertain":
                        raise ValueError("Voice detection uncertain; no fabricated instrumental label")
                    has_words = sample["lyrics"] != "[Instrumental]"
                    if (status == "no_vocals" and has_words) or (status != "no_vocals" and not has_words):
                        raise ValueError("Vocal classifier and transcript disagree; needs review")
                    sample["is_instrumental"] = status == "no_vocals"
                sample.update(caption=str(meta.get("caption") or fields["caption"]), genre=meta.get("genre") or fields.get("genre", ""),
                    bpm=meta.get("bpm") or None, keyscale=meta.get("key", meta.get("keyscale", "")),
                    timesignature=meta.get("signature", meta.get("timesignature", "")),
                    vocal_status=status, custom_tag=options["custom_tag"], labeled=True, repeat=1,
                    caption_source="user sidecar" if meta.get("caption") else "automatic local Qwen audio analysis")
                write_sidecar(Path(sample["audio_path"]).with_suffix(".txt"),
                    {"caption": sample["caption"], "genre": sample["genre"], "lyrics": sample["lyrics"],
                     "is_instrumental": str(sample["is_instrumental"]).lower(), "custom_tag": options["custom_tag"]})
                samples.append(sample)
                emit("caption", f"Annotated {sample['filename']}", number, len(unique))
            except PreparationCancelled:
                raise
            except Exception as error:
                quarantine.append({"id": sample["id"], "source_audio_path": sample["source_audio_path"], "reason": str(error)})
        annotator.close()
        assign_splits(samples, options["validation_fraction"], options["seed"])
        dataset = destination / "dataset.json"
        _write_json(dataset, {"version": 1, "metadata": {"name": options["dataset_name"], "custom_tag": options["custom_tag"],
                    "tag_position": "prepend", "genre_ratio": 0, "num_samples": len(samples)}, "samples": samples})
        _write_json(destination / "sources.json", {"sources": sources, "metadata": {"originals_modified": False}})
        if samples and options['objective'] == 'flow_dpo':
            from .folder_preferences import prepare_preferences
            try:
                preferences = prepare_preferences(str(dataset), options, progress, stop_event)
                rows = preferences['rows']
                tensor_dir = Path(preferences['tensor_dir']) if preferences['tensor_dir'] else None
                quarantine.extend(preferences['exclusions'])
            except PreparationCancelled:
                raise
            except Exception as error:
                check_stop(stop_event)
                quarantine.append({'phase': 'preferences', 'reason': str(error)})
        elif samples and options["preprocess"]:
            from .preprocess import preprocessing_models_ready, model_weights_complete
            model_dir = Path(options["checkpoint_dir"]) / ("acestep-v15-" + options["model_variant"])
            if not preprocessing_models_ready(options["checkpoint_dir"]) or not model_weights_complete(model_dir):
                if not options["allow_download"] or options["model_variant"] != "xl-sft":
                    raise ValueError("Preprocessing models incomplete; enable downloads for default SFT XL or provide complete local checkpoints")
                from .models import ensure_models
                check_stop(stop_event)
                ensure_models(options["checkpoint_dir"], progress=progress)
                check_stop(stop_event)
            key = _fingerprint([ANNOTATION_VERSION, _model_fingerprint(options), options["precision"], options["normalize"],
                                options["clip_seconds"], options["min_clip_seconds"],
                                [{k: s[k] for k in ("id", "sha256", "caption", "lyrics", "custom_tag", "split")} for s in samples]])
            tensor_dir = destination / "tensors" / key[:24]
            selected_dir, selected_json = _selected_inputs(destination, samples,
                {"name": options["dataset_name"], "custom_tag": options["custom_tag"], "tag_position": "prepend",
                 "genre_ratio": 0, "num_samples": len(samples)}, key, stop_event)
            from jk_engine.data.preprocess import preprocess_audio_files
            emit("preprocess", "Encoding audio and matching caption/lyrics")
            preprocess_audio_files(audio_dir=str(selected_dir), output_dir=str(tensor_dir),
                checkpoint_dir=options["checkpoint_dir"], variant=options["model_variant"], dataset_json=str(selected_json),
                max_duration=0, device=options["device"], precision=options["precision"], normalize="none",
                progress_callback=lambda current,total,msg: emit("preprocess", msg, current,total),
                cancel_check=lambda: stop_event is not None and stop_event.is_set(), custom_tag=options["custom_tag"])
            check_stop(stop_event)
            for sample in samples:
                path = tensor_dir / (Path(sample["audio_path"]).stem + ".pt")
                try:
                    shape = validate_supervised_tensor(path)
                    rows.append({"id": sample["id"], "tensor_path": str(path), "tensor_sha256": _hash_file(path),
                        "split": sample["split"], "holdout_group": sample["holdout_group"], "song_id": sample["song_id"],
                        "caption": sample["caption"], "lyrics": sample["lyrics"], "metadata": {**sample, **shape}})
                except Exception as error:
                    quarantine.append({"id": sample["id"], "reason": "Tensor not ready: " + str(error)})
    except PreparationCancelled:
        cancelled = True
    finally:
        annotator.close()
    cancelled = cancelled or (stop_event is not None and stop_event.is_set())
    ready = bool(rows) and any(row["split"] == "train" for row in rows) and not cancelled
    result = {"status": "cancelled" if cancelled else ("partial" if ready and quarantine else "ready" if ready else "annotated" if samples and not options["preprocess"] else "failed"),
              "ready": ready, "cancelled": cancelled, "objective": options['objective'], "dataset_json": str(destination / "dataset.json"),
              "dataset_manifest": str(destination / "supervised_manifest.json") if options['objective']=='sft' else '',
              "supervised_manifest": str(destination / "supervised_manifest.json") if options['objective']=='sft' else '',
              "pairs_manifest": preferences.get('pairs_manifest',''), "raw_pairs_manifest": preferences.get('raw_pairs_manifest',''),
              "pairs": preferences.get('pairs',0),
              "tensor_dir": str(tensor_dir) if tensor_dir else "", "report": str(destination / "preparation_report.json"),
              "selected_input_dir": str(selected_dir) if selected_dir else "",
              "samples": len(samples), "quarantined": len(quarantine), "preprocessed": len(rows),
              "model_variant": options["model_variant"], "checkpoint_dir": options["checkpoint_dir"],
              "dataset_name": options["dataset_name"], "output_dir": str(destination)}
    report = {**result, "config": options, "sources": sources, "quarantine": quarantine, "annotation_models": annotator.origin,
              "warnings": ["Automatic singing transcripts, times and captions can be wrong; they are not human verification.",
                           "Controlled synthetic preferences teach degradation avoidance, not verified global quality or MRSD rewards." if options['objective']=='flow_dpo' else "No preferences/reward scores were generated; this is a supervised dataset.",
                           "Source audio was preserved; the ACE preprocessing copy uses its required sample rate/channels."],
              "splits": dict(Counter(row["split"] for row in rows)), "originals_modified": False}
    if options['objective'] == 'sft':
        _write_json(destination / "supervised_manifest.json", {"version": 1, "samples": rows, "metadata": report})
    _write_json(destination / "preparation_report.json", report)
    if progress:
        progress({"event": "result", **result})
    return result
