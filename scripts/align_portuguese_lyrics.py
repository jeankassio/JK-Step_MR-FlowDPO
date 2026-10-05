"""Conservative MMS forced alignment of existing Portuguese human lyrics.

The text is never replaced by an ASR transcript. Automatic CTC timestamps and
uncalibrated alignment scores do not establish human-verified lyric accuracy.
Only bounded complete line groups passing configured checks enter dataset.json;
rejected candidates and source lines remain in quarantine/report files.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import unicodedata
import urllib.request

try:
    from scripts.import_muljam_pt import _root, _inside, _sha256, _write_json, _write_audio, normalize_artist
except ModuleNotFoundError:
    from import_muljam_pt import _root, _inside, _sha256, _write_json, _write_audio, normalize_artist


MODEL_URL = "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt"
MODEL_BYTES = 1262047414
MODEL_SHA256 = "20ef12963ab4924bef49ac4fc7f58ad5da2ee43b2c11bc8c853c9b90ecdbc680"
MODEL_LICENSE = "https://github.com/facebookresearch/fairseq/tree/100cd91db19bb27277a06a25eb4154c805b10189/examples/mms#license"
METHOD_VERSION = 1
ALIGN_RATE, STRIDE = 16000, 320


def normalized_line(text: str) -> str:
    """Romanize a Portuguese-only alignment copy; keep original text elsewhere."""
    text = text.replace("’", "'").replace("‘", "'")
    plain = "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))
    if re.search(r"[0-9]", plain):
        raise ValueError("Numeric lyrics require explicit Portuguese pronunciation; refusing to silently drop them")
    return " ".join(re.sub(r"[^a-z' ]", " ", plain).split())


def prepare_text(lyrics: str, dictionary: dict) -> tuple[list[dict], list[int]]:
    lines, tokens = [], []
    for number, original in enumerate(lyrics.splitlines(), 1):
        if not original.strip():
            continue
        if re.fullmatch(r"\s*\[[^]]+\]\s*", original):
            raise ValueError("Section labels must be separated from sung human lyrics before alignment")
        normalized = normalized_line(original)
        words = normalized.split()
        if not words:
            raise ValueError(f"No alignable characters in source lyric line {number}")
        start = len(tokens)
        sizes = []
        for word in words:
            word_tokens = [dictionary[c] for c in word]
            tokens.extend(word_tokens)
            sizes.append(len(word_tokens))
        lines.append({"source_line": number, "text": original, "normalized_alignment_text": normalized,
                      "words_normalized": words, "word_token_counts": sizes,
                      "token_start": start, "token_end": len(tokens)})
    if not tokens:
        raise ValueError("Human lyric text is empty")
    return lines, tokens


def viterbi_ctc(log_probs, tokens, *, blank=0, max_trellis_mb=512):
    """Exact CTC Viterbi fallback with a bounded uint8 traceback, on CPU.

    State indices, rather than just labels, preserve adjacent repeated letters.
    This implements the CTC transition rules without copied vendor source.
    """
    import numpy as np
    probs = np.asarray(log_probs, dtype=np.float32)
    targets = np.asarray(tokens, dtype=np.int64)
    if probs.ndim != 2 or len(targets) == 0 or not np.isfinite(probs).all():
        raise ValueError("Expected finite T-by-vocabulary emissions and nonempty targets")
    if (targets < 0).any() or (targets >= probs.shape[1]).any() or (targets == blank).any():
        raise ValueError("Invalid CTC target labels")
    frames, states = len(probs), 2 * len(targets) + 1
    repeated = int((targets[1:] == targets[:-1]).sum())
    if frames < len(targets) + repeated:
        raise ValueError("Audio has insufficient CTC frames for this exact text")
    if frames * states > max_trellis_mb * 1024 * 1024:
        raise ValueError("CTC traceback exceeds configured CPU memory limit")
    labels = np.full(states, blank, dtype=np.int64)
    labels[1::2] = targets
    skip = np.zeros(states, dtype=bool)
    skip[2:] = (labels[2:] != blank) & (labels[2:] != labels[:-2])
    trace = np.zeros((frames, states), dtype=np.uint8)
    previous = np.full(states, -np.inf, dtype=np.float32)
    previous[:2] = probs[0, labels[:2]]
    for frame in range(1, frames):
        advance = np.full(states, -np.inf, dtype=np.float32)
        jump = np.full(states, -np.inf, dtype=np.float32)
        advance[1:] = previous[:-1]
        jump[2:] = previous[:-2]
        jump[~skip] = -np.inf
        choices = np.stack((previous, advance, jump))
        selected = choices.argmax(axis=0)
        trace[frame] = selected
        previous = choices[selected, np.arange(states)] + probs[frame, labels]
    state = states - 1 if previous[-1] > previous[-2] else states - 2
    if not np.isfinite(previous[state]):
        raise ValueError("No finite CTC alignment path for exact text")
    path = np.empty(frames, dtype=np.int64)
    for frame in range(frames - 1, -1, -1):
        path[frame] = state
        if frame:
            state -= int(trace[frame, state])
    spans = []
    for index, label in enumerate(targets):
        positions = np.flatnonzero(path == 2 * index + 1)
        if not len(positions):
            raise ValueError("CTC path omitted a target character")
        spans.append({"token": int(label), "start_frame": int(positions[0]), "end_frame": int(positions[-1]) + 1,
                      "score": float(np.exp(probs[positions, label]).mean())})
    return spans


def align_tokens(log_probs, tokens, *, max_trellis_mb=512, backend="auto") -> tuple[list[dict], str]:
    import numpy as np
    frames, states = len(log_probs), 2 * len(tokens) + 1
    if frames * states > max_trellis_mb * 1024 * 1024:
        raise ValueError("CTC traceback exceeds configured CPU memory limit")
    if backend != "numpy":
        try:
            import torch
            import torchaudio
            emissions = torch.from_numpy(np.asarray(log_probs, dtype=np.float32)).contiguous().unsqueeze(0)
            targets = torch.tensor([tokens], dtype=torch.int32)
            alignment, scores = torchaudio.functional.forced_align(emissions, targets, blank=0)
            merged = torchaudio.functional.merge_tokens(alignment[0], scores[0].exp())
            if len(merged) != len(tokens) or any(span.token != target for span, target in zip(merged, tokens)):
                raise ValueError("Native CTC alignment does not preserve the exact target token sequence")
            return [{"token": span.token, "start_frame": span.start, "end_frame": span.end, "score": span.score}
                    for span in merged], "torchaudio_cpu"
        except (AttributeError, NotImplementedError, RuntimeError):
            if backend == "native":
                raise
    return viterbi_ctc(log_probs, tokens, max_trellis_mb=max_trellis_mb), "numpy_viterbi"


def line_timings(lines: list[dict], spans: list[dict], duration: float) -> list[dict]:
    aligned = []
    for line in lines:
        chars = spans[line["token_start"]:line["token_end"]]
        count = sum(c["end_frame"] - c["start_frame"] for c in chars)
        score = sum(c["score"] * (c["end_frame"] - c["start_frame"]) for c in chars) / count
        words, cursor = [], 0
        for word, size in zip(line["words_normalized"], line["word_token_counts"]):
            word_chars = chars[cursor:cursor + size]
            mass = sum(c["end_frame"] - c["start_frame"] for c in word_chars)
            words.append({"normalized_text": word,
                          "start": word_chars[0]["start_frame"] * STRIDE / ALIGN_RATE,
                          "end": min(duration, word_chars[-1]["end_frame"] * STRIDE / ALIGN_RATE),
                          "ctc_score": sum(c["score"] * (c["end_frame"] - c["start_frame"]) for c in word_chars) / mass})
            cursor += size
        aligned.append({**line, "start": words[0]["start"], "end": words[-1]["end"],
                        "ctc_score": score, "low_character_fraction": sum(c["score"] < .15 for c in chars) / len(chars),
                        "words": words, "timestamps_human_verified": False})
    return aligned


def group_lines(lines: list[dict], *, min_seconds=8, max_seconds=30, max_gap=5, padding=.25,
                min_line_score=.35, max_low_character_fraction=.25, max_word_seconds=8) -> tuple[list[list[dict]], list[dict]]:
    accepted, quarantine, current = [], [], []
    def finish():
        nonlocal current
        if current:
            if current[-1]["end"] - current[0]["start"] + 2 * padding >= min_seconds:
                accepted.append(current)
            else:
                quarantine.append({"reason": "complete line group is shorter than minimum duration", "lines": current})
            current = []
    for line in lines:
        reasons = []
        if line["ctc_score"] < min_line_score:
            reasons.append("low uncalibrated CTC line score")
        if line["low_character_fraction"] > max_low_character_fraction:
            reasons.append("too many weakly supported characters")
        if any(word["end"] - word["start"] > max_word_seconds for word in line["words"]):
            reasons.append("implausibly long forced word span")
        span = line["end"] - line["start"]
        if span <= 0 or span + 2 * padding > max_seconds:
            reasons.append("complete line cannot fit crop duration")
        if reasons:
            finish()
            quarantine.append({"reason": "; ".join(reasons), "lines": [line]})
            continue
        if current and (line["start"] - current[-1]["end"] > max_gap
                        or line["end"] - current[0]["start"] + 2 * padding > max_seconds):
            finish()
        current.append(line)
    finish()
    return accepted, quarantine


def ensure_model(cache, progress=None) -> tuple[Path, dict]:
    root = _root(cache)
    target = _inside(root, "model.pt")
    if not target.is_file():
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", dir=root, suffix=".downloading", delete=False) as handle:
            staging = Path(handle.name)
            try:
                with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
                    received, previous = 0, 0
                    while block := response.read(8 * 1024 * 1024):
                        handle.write(block)
                        received += len(block)
                        if progress and received - previous >= 128 * 1024 * 1024:
                            progress({"event": "download_model", "bytes": received, "total": MODEL_BYTES})
                            previous = received
            except BaseException:
                handle.close()
                staging.unlink(missing_ok=True)
                raise
        try:
            if staging.stat().st_size != MODEL_BYTES or _sha256(staging) != MODEL_SHA256:
                raise ValueError("Official MMS_FA model download failed byte-size/SHA256 verification")
            staging.replace(target)
        finally:
            staging.unlink(missing_ok=True)
    if target.stat().st_size != MODEL_BYTES or _sha256(target) != MODEL_SHA256:
        raise ValueError("Cached MMS_FA model does not match the pinned official model hash")
    provenance = {"name": "torchaudio.pipelines.MMS_FA", "url": MODEL_URL, "sha256": MODEL_SHA256,
                  "size_bytes": MODEL_BYTES, "license": "CC-BY-NC 4.0", "license_url": MODEL_LICENSE,
                  "domain": "Multilingual speech forced alignment; singing accuracy is not established by this model card"}
    _write_json(root / "model.source.json", provenance)
    return target, provenance


def emissions_for_audio(model, audio, sample_rate, device, *, chunk_seconds=20, context_seconds=2, dtype="float16", progress=None):
    import numpy as np
    import torch
    from scipy.signal import resample_poly
    # Downmix/resample an inference-only copy. Training crops use the original
    # channels/sample rate/amplitude through the importer helper below.
    mono = np.asarray(audio, dtype=np.float32).mean(axis=1)
    divisor = math.gcd(int(sample_rate), ALIGN_RATE)
    if sample_rate != ALIGN_RATE:
        mono = resample_poly(mono, ALIGN_RATE // divisor, sample_rate // divisor).astype(np.float32)
    core = round(chunk_seconds * ALIGN_RATE / STRIDE) * STRIDE
    context = round(context_seconds * ALIGN_RATE / STRIDE) * STRIDE
    if core <= STRIDE or context < STRIDE:
        raise ValueError("Emission chunk/context must be positive multiples of encoder stride")
    parts, start_frame = [], 0
    inference_dtype = torch.float16 if dtype == "float16" and str(device).startswith("cuda") else torch.float32
    for core_start in range(0, len(mono), core):
        core_end = min(core_start + core, len(mono))
        start, end = max(0, core_start - context), min(len(mono), core_end + context)
        wave = torch.from_numpy(mono[start:end].copy()).unsqueeze(0).to(device=device, dtype=inference_dtype)
        with torch.inference_mode():
            emission, _ = model(wave)
        # Start offsets and core windows use the exact 320-sample feature grid,
        # avoiding accumulated time stretching across overlapping windows.
        first = (core_start - start) // STRIDE
        last = min((core_end - start + STRIDE - 1) // STRIDE, emission.shape[1])
        if first >= last:
            raise ValueError("Emission chunk contains no valid CTC feature frames")
        expected_start = core_start // STRIDE
        if expected_start != start_frame:
            raise ValueError("Emission stitching would introduce a timeline gap or duplicate frames")
        part = emission[0, first:last].float().cpu().numpy()
        if not np.isfinite(part).all():
            raise ValueError("Nonfinite MMS_FA emissions")
        parts.append(part)
        start_frame += len(part)
        del wave, emission
        if progress:
            progress({"event": "emissions", "seconds": core_end / ALIGN_RATE, "total_seconds": len(mono) / ALIGN_RATE})
    return np.concatenate(parts, axis=0)


def align_dataset(manifest, output, *, device="cpu", model_cache=".cache/mms_fa", max_songs=None,
                  song_ids=(), chunk_seconds=20, context_seconds=2, dtype="float16", min_seconds=8,
                  max_seconds=30, max_gap=5, padding=.25, min_line_score=.35,
                  max_low_character_fraction=.25, max_word_seconds=8, max_trellis_mb=512,
                  backend="auto", progress=None) -> dict:
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    numeric = [chunk_seconds, context_seconds, min_seconds, max_seconds, max_gap, padding, min_line_score, max_low_character_fraction, max_word_seconds, max_trellis_mb]
    if not all(math.isfinite(float(x)) for x in numeric) or not 0 < min_seconds <= max_seconds or not 0 <= min_line_score <= 1 or not 0 <= max_low_character_fraction <= 1 or padding < 0 or 2 * padding >= max_seconds or max_gap < 0 or max_word_seconds <= 0 or max_trellis_mb <= 0:
        raise ValueError("Invalid duration, memory or confidence parameters")
    manifest_path, destination = Path(manifest).expanduser().resolve(), _root(output)
    raw = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    tracks = raw.get("samples", [])
    if song_ids:
        tracks = [s for s in tracks if str(s.get("song_id", s.get("id"))) in set(song_ids)]
    if max_songs is not None:
        if max_songs < 1:
            raise ValueError("max_songs must be positive")
        tracks = tracks[:max_songs]
    if not tracks:
        raise ValueError("No selected source songs")
    model_path, model_origin = ensure_model(model_cache, progress)
    bundle = torchaudio.pipelines.MMS_FA
    dictionary = bundle.get_dict(star=None)
    model = bundle.get_model(with_star=False, dl_kwargs={"model_dir": str(model_path.parent)}).to(device=device, dtype=torch.float16 if dtype == "float16" and str(device).startswith("cuda") else torch.float32).eval()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    parameters = {"method_version": METHOD_VERSION, "chunk_seconds": chunk_seconds, "context_seconds": context_seconds,
                  "dtype": dtype if str(device).startswith("cuda") else "float32", "min_seconds": min_seconds,
                  "max_seconds": max_seconds, "max_gap": max_gap, "padding": padding, "min_line_score": min_line_score,
                  "max_low_character_fraction": max_low_character_fraction, "max_word_seconds": max_word_seconds,
                  "max_trellis_mb": max_trellis_mb, "ctc_backend": backend}
    samples, quarantined, full_tracks, counts = [], [], [], Counter()
    for track_number, source in enumerate(tracks, 1):
        song_id = str(source.get("song_id", source.get("id", "")))
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", song_id)
        if not safe_id:
            raise ValueError("Missing source song id")
        try:
            if str(source.get("language", "pt")).casefold() not in ("pt", "portuguese", "pt-br", "português"):
                raise ValueError("Source song is not labeled Portuguese")
            audio_path = Path(source["audio_path"])
            audio_path = (audio_path if audio_path.is_absolute() else manifest_path.parent / audio_path).resolve()
            digest = _sha256(audio_path)
            expected = source.get("sha256") or source.get("audio_info", {}).get("sha256")
            if not expected or digest != expected:
                raise ValueError("Source audio SHA256 missing or mismatched")
            audio, sample_rate = sf.read(audio_path, dtype="float64", always_2d=True)
            if not len(audio) or not np.isfinite(audio).all():
                raise ValueError("Source audio is empty/nonfinite")
            duration = len(audio) / sample_rate
            lyrics = source.get("lyrics", "")
            lines, tokens = prepare_text(lyrics, dictionary)
            cache_key = hashlib.sha256(json.dumps([METHOD_VERSION, digest, lyrics, MODEL_SHA256, chunk_seconds, context_seconds, parameters["dtype"]], ensure_ascii=False).encode()).hexdigest()
            cache_path = _inside(destination, f"alignments/{safe_id}_{cache_key[:24]}.json")
            if cache_path.is_file():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("cache_key") != cache_key:
                    raise ValueError("Alignment cache provenance mismatch")
                aligned, actual_backend = cached["lines"], cached["ctc_backend"]
                counts["cached_alignments"] += 1
            else:
                if progress:
                    progress({"event": "align_start", "song_id": song_id, "song": track_number, "songs": len(tracks), "seconds": duration, "tokens": len(tokens)})
                emissions = emissions_for_audio(model, audio, sample_rate, device, chunk_seconds=chunk_seconds,
                                                context_seconds=context_seconds, dtype=dtype,
                                                progress=(lambda event: progress({"song_id": song_id, **event})) if progress else None)
                spans, actual_backend = align_tokens(emissions, tokens, max_trellis_mb=max_trellis_mb, backend=backend)
                aligned = line_timings(lines, spans, duration)
                _write_json(cache_path, {"cache_key": cache_key, "song_id": song_id, "audio_sha256": digest,
                                        "lyrics_sha256": hashlib.sha256(lyrics.encode()).hexdigest(), "model_sha256": MODEL_SHA256,
                                        "ctc_backend": actual_backend, "lines": aligned})
                del emissions, spans
            groups, rejected = group_lines(aligned, min_seconds=min_seconds, max_seconds=max_seconds, max_gap=max_gap,
                                           padding=padding, min_line_score=min_line_score,
                                           max_low_character_fraction=max_low_character_fraction, max_word_seconds=max_word_seconds)
            quarantined.extend({"song_id": song_id, **item} for item in rejected)
            artist_key = normalize_artist(source.get("artist", ""))
            if not artist_key:
                raise ValueError("Source artist is required for grouped holdout")
            artist_id = "artist:" + hashlib.sha256(artist_key.encode()).hexdigest()[:24]
            shared = {**source, "song_id": song_id, "artist_id": artist_id, "holdout_group": artist_id,
                      "source_audio_path": str(audio_path), "source_audio_sha256": digest,
                      "lyrics_source": "Human text from source manifest, unchanged",
                      "timestamp_source": "Automatic MMS_FA CTC forced alignment", "timestamps_human_verified": False,
                      "ctc_score_is_calibrated_accuracy": False, "ctc_backend": actual_backend,
                      "alignment_model_sha256": MODEL_SHA256, "alignment_audio_processing": "Inference-only mono 16 kHz copy; source training audio preserved"}
            full_tracks.append({**shared, "audio_path": str(audio_path), "duration": duration, "lines": aligned, "lyrics": lyrics})
            for segment_number, group in enumerate(groups, 1):
                first, last = group[0]["start"], group[-1]["end"]
                chosen_lines = {line["source_line"] for line in group}
                before = max((line["end"] for line in aligned if line["source_line"] not in chosen_lines and line["end"] <= first), default=0)
                after = min((line["start"] for line in aligned if line["source_line"] not in chosen_lines and line["start"] >= last), default=duration)
                start_frame = max(0, math.floor((first - padding) * sample_rate), math.ceil(before * sample_rate))
                end_frame = min(len(audio), math.ceil((last + padding) * sample_rate), math.floor(after * sample_rate))
                crop_duration = (end_frame - start_frame) / sample_rate
                if crop_duration < min_seconds or crop_duration > max_seconds:
                    quarantined.append({"song_id": song_id, "reason": "neighbor-bounded crop is outside duration limits", "lines": group})
                    continue
                identity = f"{safe_id}_fa_{segment_number:03d}"
                target = _inside(destination, f"audio/{identity}.flac")
                target, crop_hash, action, encoding = _write_audio(target, audio[start_frame:end_frame], sample_rate)
                segment_lyrics = "\n".join(line["text"] for line in group)
                condition_hash = hashlib.sha256(json.dumps([source.get("caption", ""), segment_lyrics], ensure_ascii=False).encode()).hexdigest()
                samples.append({**shared, **encoding, "id": identity, "audio_path": str(target), "sha256": crop_hash,
                                "sampling_rate": sample_rate, "channels": audio.shape[1], "duration": crop_duration,
                                "crop_start_frame": start_frame, "crop_end_frame": end_frame,
                                "crop_start": start_frame / sample_rate, "crop_end": end_frame / sample_rate,
                                "lyrics": segment_lyrics, "lines": [{**line, "segment_start": line["start"] - start_frame / sample_rate,
                                                                   "segment_end": line["end"] - start_frame / sample_rate} for line in group],
                                "words": [], "group_id": f"{identity}:{condition_hash[:24]}", "condition_sha256": condition_hash,
                                "ctc_score_mean": sum(line["ctc_score"] for line in group) / len(group)})
                counts[f"crops_{action}"] += 1
            counts["songs_aligned"] += 1
            if progress:
                progress({"event": "aligned_song", "song_id": song_id, "line_score_mean": sum(line["ctc_score"] for line in aligned) / len(aligned),
                          "candidate_groups": len(groups), "rejected_groups": len(rejected), "accepted_segments_total": len(samples)})
        except (ValueError, OSError, RuntimeError, KeyError) as error:
            quarantined.append({"song_id": song_id, "reason": str(error), "entire_song": True})
            counts["songs_failed"] += 1
            if progress:
                progress({"event": "song_quarantined", "song_id": song_id, "reason": str(error)})
    counts.update(source_songs=len(tracks), segments=len(samples), quarantine_entries=len(quarantined),
                  accepted_lines=sum(len(sample["lines"]) for sample in samples))
    origin = {"source_manifest": str(manifest_path), "source_manifest_sha256": _sha256(manifest_path),
              "source_metadata": raw.get("metadata", {}), "model": model_origin,
              "torch_version": torch.__version__, "torchaudio_version": torchaudio.__version__,
              "confidence": "Uncalibrated forced alignment posterior scores; not probability of correct lyrics or musical quality",
              "audio_processing": "Crops from original full-bandwidth channels/sample rate/amplitude; FLAC PCM24 or WAV FLOAT32 for out-of-range peaks",
              "preferences_provided": False, "postproduction_applied": False, "human_verified_timestamps": False}
    report = {"manifest": str(destination / "dataset.json"), "full_tracks_manifest": str(destination / "full_tracks.json"),
              "parameters": parameters, "origin": origin, "counts": dict(counts),
              "accepted_seconds": sum(sample["duration"] for sample in samples),
              "accepted_genres": dict(Counter(str(sample.get("genre", "unknown")) for sample in samples)),
              "device": str(device), "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else 0}
    _write_json(destination / "dataset.json", {"version": 1, "samples": samples, "metadata": report})
    _write_json(destination / "full_tracks.json", {"version": 1, "samples": full_tracks, "metadata": report})
    _write_json(destination / "quarantine.json", {"version": 1, "entries": quarantined, "metadata": report})
    _write_json(destination / "alignment_report.json", report)
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-cache", default=".cache/mms_fa")
    parser.add_argument("--max-songs", type=int)
    parser.add_argument("--song-id", action="append", default=[])
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    for name, default in (("chunk-seconds", 20), ("context-seconds", 2), ("min-seconds", 8), ("max-seconds", 30),
                          ("max-gap", 5), ("padding", .25), ("min-line-score", .35),
                          ("max-low-character-fraction", .25), ("max-word-seconds", 8), ("max-trellis-mb", 512)):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--ctc-backend", choices=("auto", "native", "numpy"), default="auto")
    args = vars(parser.parse_args(argv))
    args["song_ids"] = args.pop("song_id")
    args["backend"] = args.pop("ctc_backend")
    report = align_dataset(**args, progress=lambda event: print(json.dumps(event, ensure_ascii=True), flush=True))
    print(json.dumps({k: report[k] for k in ("manifest", "counts", "accepted_seconds", "accepted_genres", "cuda_peak_allocated_bytes")}, ensure_ascii=True))
    return 0 if report["counts"]["segments"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
