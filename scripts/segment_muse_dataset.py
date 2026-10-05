"""Crop complete, lyric-bearing Muse sections from an existing local manifest.

No downloads, scoring, stem separation, gain changes or resampling are performed.
Automatic section timestamps are preserved, not claimed to be verified word
boundaries. One source track per song is selected; song-level holdout also keeps
any subsequently created acoustic-degradation variants in the same split.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re

import numpy as np
import soundfile as sf

if __package__:
    from .import_muse import _json_write, _root_path, _sha256, _target_inside
    from .import_muljam_pt import _write_audio
else:
    from import_muse import _json_write, _root_path, _sha256, _target_inside
    from import_muljam_pt import _write_audio


def _source_path(sample: dict, manifest: Path) -> Path:
    value = sample.get("audio_path")
    if not isinstance(value, str) or not value:
        raise ValueError("Missing audio_path")
    requested = Path(value).expanduser()
    path = _root_path(requested) if requested.is_absolute() else _target_inside(manifest.parent, value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _lyrics(sections: list[dict]) -> str:
    # Match import_muse: keep every source word, newline and punctuation mark.
    return "\n\n".join((f"[{s['section']}]\n" if s["section"] else "") + s["text"] for s in sections)


def _caption(style: str, sections: list[dict]) -> str:
    pieces = [style]
    for section in sections:
        if section.get("desc"):
            label = f"{section['section']}: " if section["section"] else ""
            pieces.append(label + section["desc"])
    return "\n\n".join(piece for piece in pieces if piece)


def _family(label: str) -> str:
    normalized = label.casefold().replace("-", "").replace(" ", "")
    for family in ("prechorus", "chorus", "verse", "bridge", "refrain", "hook"):
        if family in normalized:
            return family
    return normalized.rstrip("0123456789")


def _has_words(text: str) -> bool:
    without_markers = re.sub(r"\[[^\]]*\]", "", text)
    return any(character.isalnum() for character in without_markers)


def _candidates(sample: dict, info, *, min_seconds: float, max_seconds: float,
                max_gap: float) -> tuple[list[dict], list[dict], Counter]:
    raw = sample.get("sections")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Missing nonempty sections list")
    sections, issues, counts = [], [], Counter()
    for index, section in enumerate(raw):
        if not isinstance(section, dict):
            raise ValueError(f"Section {index} is not an object")
        label, text, desc = section.get("section", ""), section.get("text", ""), section.get("desc", "")
        if not all(isinstance(value, str) for value in (label, text, desc)):
            raise ValueError(f"Section {index} labels, lyrics and desc must be strings")
        start, end = float(section["startS"]), float(section["endS"])
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
            raise ValueError(f"Section {index} needs finite 0 <= startS < endS")
        if sections and start < sections[-1]["startS"]:
            raise ValueError("Source sections are not in chronological order")
        sections.append({**section, "section": label, "text": text, "desc": desc,
                         "startS": start, "endS": end})
        if end > info.frames / info.samplerate:
            counts["sections_outside_audio"] += 1
            issues.append({"kind": "excluded_outside_audio_annotation", "section_index": index,
                           "section": label, "startS": start, "endS": end,
                           "audio_duration": info.frames / info.samplerate,
                           "reason": "Section extends beyond decoded audio; not clamped"})
        if not _has_words(text):
            counts["sections_without_known_lyrics"] += 1
        elif end - start > max_seconds:
            counts["sections_over_duration_limit"] += 1
            issues.append({"kind": "excluded_long_section", "section_index": index,
                           "section": label, "startS": start, "endS": end,
                           "duration": end - start, "reason": "Complete section exceeds max_seconds; not truncated"})
    candidates = []
    for index, section in enumerate(sections):
        if not _has_words(section["text"]):
            continue
        groups = [[index]]
        if index + 1 < len(sections) and _has_words(sections[index + 1]["text"]):
            groups.append([index, index + 1])
        for indices in groups:
            selected = [sections[i] for i in indices]
            first, last = selected[0]["startS"], max(s["endS"] for s in selected)
            if len(selected) == 2 and selected[1]["startS"] - selected[0]["endS"] > max_gap:
                counts["candidate_gap_over_limit"] += 1
                continue
            if any(s["endS"] - s["startS"] > max_seconds for s in selected):
                continue
            start_frame, end_frame = math.floor(first * info.samplerate), math.ceil(last * info.samplerate)
            frames = end_frame - start_frame
            if frames < math.ceil(min_seconds * info.samplerate):
                counts["candidate_too_short"] += 1
                continue
            if frames > math.floor(max_seconds * info.samplerate):
                counts["candidate_too_long"] += 1
                continue
            if not 0 <= start_frame < end_frame <= info.frames:
                counts["candidate_outside_audio"] += 1
                issues.append({"kind": "excluded_outside_audio", "section_indices": indices,
                               "startS": first, "endS": last, "audio_duration": info.frames / info.samplerate})
                continue
            # Neither boundary may cut another annotated section, including a
            # section whose vocals have no known transcription. Two overlapping
            # complete sections can be retained together; larger clusters cannot.
            if any(i not in indices and s["startS"] < last and s["endS"] > first
                   for i, s in enumerate(sections)):
                counts["candidate_overlaps_unselected_section"] += 1
                continue
            candidates.append({"indices": indices, "sections": selected, "start_frame": start_frame,
                               "end_frame": end_frame, "lyrics": _lyrics(selected),
                               "families": {_family(s["section"]) for s in selected}})
    counts["eligible_candidates"] += len(candidates)
    return candidates, issues, counts


def _select(candidates: list[dict], maximum: int, rng: random.Random) -> list[dict]:
    remaining = list(candidates)
    rng.shuffle(remaining)
    selected = []
    while remaining and len(selected) < maximum:
        families = set().union(*(item["families"] for item in selected))
        remaining.sort(key=lambda item: len(item["families"] - families), reverse=True)
        chosen = remaining.pop(0)
        selected.append(chosen)
        remaining = [item for item in remaining
                     if item["lyrics"] != chosen["lyrics"]
                     and (item["end_frame"] <= chosen["start_frame"]
                          or item["start_frame"] >= chosen["end_frame"])]
    return selected


def _assign_splits(samples: list[dict], fraction: float, seed: int) -> None:
    songs = sorted({sample["song_id"] for sample in samples})
    random.Random(seed).shuffle(songs)
    count = round(len(songs) * fraction)
    if fraction and len(songs) > 1:
        count = max(1, min(count, len(songs) - 1))
    heldout = set(songs[:count])
    for sample in samples:
        sample["split"] = "validation" if sample["song_id"] in heldout else "train"


def segment_muse_dataset(input_manifest, output, *, max_samples=1000, max_seconds=30.0,
                         min_seconds=8.0, max_per_song=2, max_gap=5.0, seed=42,
                         validation_fraction=0.1, progress=None) -> dict:
    options = (float(min_seconds), float(max_seconds), float(max_gap), float(validation_fraction))
    if (not all(math.isfinite(value) for value in options) or not 0 < min_seconds <= max_seconds
            or max_gap < 0 or not 0 <= validation_fraction < 1 or max_samples < 1
            or max_per_song not in (1, 2)):
        raise ValueError("Invalid segment limits: positive durations/count, max_per_song 1 or 2, validation_fraction in [0,1)")
    source = _root_path(input_manifest)
    destination = _root_path(output)
    manifest = _target_inside(destination, "dataset.json")
    if source == manifest:
        raise ValueError("Output dataset.json must differ from the input manifest")
    raw = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or not isinstance(raw.get("samples"), list):
        raise ValueError("Input must be a Muse imported manifest with a samples list")
    parameters = {"max_samples": max_samples, "max_seconds": max_seconds, "min_seconds": min_seconds,
                  "max_per_song": max_per_song, "max_sections_per_segment": 2, "max_gap": max_gap,
                  "seed": seed, "validation_fraction": validation_fraction}
    segmentation = {"input_manifest": str(source), "input_manifest_sha256": _sha256(source),
                    "parameters": parameters}
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8-sig"))
        if previous.get("metadata", {}).get("segmentation") != segmentation:
            raise FileExistsError("Existing output manifest has different source/options; choose a new output directory")
    grouped, counts, issues = defaultdict(list), Counter(), []
    counts["input_variants"] = len(raw["samples"])
    for row_number, sample in enumerate(raw["samples"]):
        if not isinstance(sample, dict) or not isinstance(sample.get("song_id"), str) or not sample["song_id"]:
            raise ValueError(f"Sample {row_number} has no valid song_id")
        if sample.get("language") != "en" or sample.get("error"):
            counts["excluded_language_or_source_error"] += 1
            continue
        if not isinstance(sample.get("style", sample.get("caption", "")), str):
            raise ValueError(f"Sample {row_number} has an invalid style/caption")
        grouped[sample["song_id"]].append(sample)
    rng = random.Random(seed)
    song_ids = sorted(grouped)
    rng.shuffle(song_ids)
    plans = []
    counts["input_songs"] = len(song_ids)
    for song_id in song_ids:
        variants = sorted(grouped[song_id], key=lambda s: (str(s.get("track_index", "")), str(s.get("audio_path", ""))))
        rng.shuffle(variants)
        for sample in variants:
            counts["inspected_variants"] += 1
            try:
                audio_path = _source_path(sample, source)
                info = sf.info(audio_path)
                if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
                    raise ValueError("Empty or invalid source audio")
                candidates, skipped, candidate_counts = _candidates(
                    sample, info, min_seconds=min_seconds, max_seconds=max_seconds, max_gap=max_gap)
                counts.update(candidate_counts)
                issues.extend({"song_id": song_id, "source_sample_id": sample.get("id"), **item} for item in skipped)
                selected = _select(candidates, max_per_song, rng)
                if selected:
                    plans.append((sample, audio_path, info, selected))
                    break
                counts["variants_without_eligible_sections"] += 1
            except (ValueError, KeyError, TypeError, RuntimeError, OSError) as error:
                counts["invalid_source_variants"] += 1
                issues.append({"song_id": song_id, "source_sample_id": sample.get("id"),
                               "kind": "excluded_source_variant", "reason": str(error)})
        if progress and len(plans) % 100 == 0:
            progress({"event": "plan", "eligible_songs": len(plans), "inspected_variants": counts["inspected_variants"]})
    samples, hashes = [], {}
    # Give each source song its first segment before taking second segments.
    for segment_round in range(max_per_song):
        for sample, audio_path, info, selected in plans:
            if len(samples) >= max_samples:
                break
            if segment_round >= len(selected):
                continue
            candidate = selected[segment_round]
            start, end = candidate["start_frame"], candidate["end_frame"]
            with sf.SoundFile(audio_path) as audio_file:
                audio_file.seek(start)
                audio = audio_file.read(end - start, dtype="float64", always_2d=True)
            if audio.shape != (end - start, info.channels) or not np.isfinite(audio).all():
                raise ValueError(f"Invalid decoded crop for {audio_path} sections {candidate['indices']}")
            if audio_path not in hashes:
                digest = _sha256(audio_path)
                expected = sample.get("sha256") or sample.get("audio_sha256")
                if expected and digest.casefold() != str(expected).casefold():
                    raise ValueError(f"Source audio SHA256 mismatch: {audio_path}")
                hashes[audio_path] = digest
            style = sample.get("style", sample.get("caption", ""))
            caption = _caption(style, candidate["sections"])
            condition = hashlib.sha256(json.dumps([caption, candidate["lyrics"]], ensure_ascii=False,
                                                 separators=(",", ":")).encode()).hexdigest()
            identity = hashlib.sha256(json.dumps([sample["song_id"], sample.get("track_index"), str(audio_path),
                                                  start, end, condition], ensure_ascii=False,
                                                 separators=(",", ":")).encode()).hexdigest()[:24]
            target = _target_inside(destination, f"audio/{identity}.flac")
            target, digest, action, encoding = _write_audio(target, audio, info.samplerate)
            counts[f"crops_{action}"] += 1
            start_seconds = start / info.samplerate
            samples.append({"id": identity, "audio_path": str(target), "sha256": digest,
                            "caption": caption, "style": style, "lyrics": candidate["lyrics"], "language": "en",
                            "song_id": sample["song_id"], "track_index": sample.get("track_index"),
                            "group_id": f"{sample['song_id']}:{identity}:{condition[:24]}",
                            "holdout_group": sample["song_id"], "artist_holdout": sample["song_id"],
                            "condition_sha256": condition, "duration": (end - start) / info.samplerate,
                            "sampling_rate": info.samplerate, "channels": info.channels, **encoding,
                            "source_sample_id": sample.get("id"), "source_group_id": sample.get("group_id"),
                            "source_audio_path": str(audio_path), "source_audio_sha256": hashes[audio_path],
                            "source_metadata_audio_path": sample.get("source_audio_path"),
                            "source_audio_encoding": info.format,
                            "source_audio_lossy": bool(sample.get("lossy_audio", audio_path.suffix.casefold() == ".mp3")),
                            "source_style_sim": sample.get("style_sim"), "style_sim_is_musical_quality": False,
                            "original_quality_unverified": True, "timestamps_human_verified": False,
                            "crop_start": start_seconds, "crop_end": end / info.samplerate,
                            "crop_start_frame": start, "crop_end_frame": end,
                            "source_section_indices": candidate["indices"],
                            "sections": [{**section, "segment_start": section["startS"] - start_seconds,
                                          "segment_end": section["endS"] - start_seconds}
                                         for section in candidate["sections"]]})
            if progress and (len(samples) % 25 == 0 or len(samples) == max_samples):
                progress({"event": "crop", "segments": len(samples), "max_samples": max_samples})
        if len(samples) >= max_samples:
            break
    _assign_splits(samples, validation_fraction, seed)
    counts.update(eligible_songs=len(plans), selected_songs=len({s["song_id"] for s in samples}),
                  segments=len(samples), selected_source_variants=len(hashes),
                  flac_pcm24_segments=sum(s["audio_format"] == "FLAC" for s in samples),
                  wav_float32_segments=sum(s["float_fallback"] for s in samples), clipped_samples=0)
    metadata = {"origin": raw.get("metadata", {}).get("origin", {}), "segmentation": segmentation,
                "counts": dict(counts), "split_policy": "source song_id; all segments and later acoustic variants stay together",
                "split_songs": dict(Counter({s["song_id"]: s["split"] for s in samples}.values())),
                "split_segments": dict(Counter(s["split"] for s in samples)),
                "audio_processing": "Original full mix, complete sections only. FLAC PCM24 or FLOAT32 WAV for overshoot; no gain, clipping, resampling, channel changes or source bandwidth restoration.",
                "annotations": "Muse automatic section timestamps and exact source section lyrics/descriptions; no word timings inferred or listening verification",
                "original_quality_unverified": True, "preferences_provided": False}
    _json_write(manifest, {"version": 1, "samples": samples, "metadata": metadata})
    report = {"manifest": str(manifest), **metadata, "issues": issues,
              "crops": [{key: sample[key] for key in ("id", "song_id", "track_index", "split", "audio_path",
                        "sha256", "source_audio_path", "source_audio_sha256", "source_section_indices",
                        "crop_start_frame", "crop_end_frame", "crop_start", "crop_end", "duration",
                        "sampling_rate", "channels", "audio_format", "audio_subtype", "peak", "float_fallback")}
                        for sample in samples]}
    _json_write(_target_inside(destination, "crop_report.json"), report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Local import_muse dataset.json")
    parser.add_argument("--output", required=True, help="Output directory for crops, dataset.json and crop_report.json")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--min-seconds", type=float, default=8.0)
    parser.add_argument("--max-per-song", type=int, choices=(1, 2), default=2)
    parser.add_argument("--max-gap", type=float, default=5.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    report = segment_muse_dataset(args.input, args.output, max_samples=args.max_samples,
                                 max_seconds=args.max_seconds, min_seconds=args.min_seconds,
                                 max_per_song=args.max_per_song, max_gap=args.max_gap, seed=args.seed,
                                 validation_fraction=args.validation_fraction,
                                 progress=lambda event: print(json.dumps(event, ensure_ascii=True), flush=True))
    print(json.dumps({key: report[key] for key in ("manifest", "counts", "split_songs", "split_segments")},
                     ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
