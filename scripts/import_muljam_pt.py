"""Import verified local Portuguese MulJam songs and automatic line annotations.

No network requests, scoring, preference pairs, gain changes or resampling are
performed. FLAC PCM24 stores ordinary crops; WAV FLOAT32 preserves MP3 decoded
peaks outside the integer PCM range. Neither restores lost MP3 bandwidth.
The source timestamps are automatic estimates, never human-verified boundaries.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import tempfile
import unicodedata


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def normalize_artist(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value))
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(plain.casefold().split())


def _inside(root: Path, name: str) -> Path:
    normalized = str(name).replace("\\", "/")
    relative = PurePosixPath(normalized)
    if not normalized or "\0" in normalized or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe relative audio path: {name!r}")
    for part in relative.parts:
        if ":" in part or part.endswith((" ", ".")) or re.match(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", part, re.I):
            raise ValueError(f"Unsafe path component: {part!r}")
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root):
        raise ValueError(f"Audio path escapes its root: {name!r}")
    for candidate in (target, *target.parents):
        if candidate == root.parent:
            break
        if candidate.is_symlink() or (hasattr(candidate, "is_junction") and candidate.is_junction()):
            raise ValueError(f"Refusing an audio/output path through a link: {candidate}")
    return target


def _root(value) -> Path:
    requested = Path(value).expanduser().absolute()
    for candidate in (requested, *requested.parents):
        if candidate.is_symlink() or (hasattr(candidate, "is_junction") and candidate.is_junction()):
            raise ValueError(f"Refusing a root through a link: {candidate}")
    return requested.resolve()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".writing", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _audio_path(track: dict, metadata_path: Path, audio_root) -> Path:
    name = track.get("audio_path") or track.get("file_name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Missing relative audio_path for {track.get('song_id')}")
    if audio_root is None:
        return _inside(_root(metadata_path.parent), name)
    root = _root(audio_root)
    # The published metadata convention is audio/XX/track.mp3; the explicit
    # audio root points at that audio folder, while nested names stay intact.
    parts = PurePosixPath(name.replace("\\", "/")).parts
    if parts and parts[0].casefold() == "audio":
        name = "/".join(parts[1:])
    return _inside(root, name)


def _load_tracks(metadata_path: Path) -> tuple[dict, list[dict]]:
    raw = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if isinstance(raw, list):
        source, tracks = {}, raw
    elif isinstance(raw, dict):
        source, tracks = raw.get("metadata", {}), raw.get("tracks", [])
    else:
        raise ValueError("Audio metadata must be an object containing tracks, or a track list")
    if not isinstance(source, dict) or not isinstance(tracks, list) or not tracks:
        raise ValueError("Expected nonempty tracks list and metadata object")
    seen = set()
    for track in tracks:
        if not isinstance(track, dict):
            raise ValueError("Every audio track must be a JSON object")
        song_id = str(track.get("song_id", ""))
        if not re.fullmatch(r"muljam_[0-9]+", song_id) or song_id in seen:
            raise ValueError(f"Invalid or duplicate MulJam song_id: {song_id!r}")
        seen.add(song_id)
        if track.get("language", "pt").casefold() not in ("pt", "pt-br", "portuguese", "português"):
            raise ValueError(f"Audio metadata is not Portuguese: {song_id}")
        if not normalize_artist(track.get("artist", "")):
            raise ValueError(f"Missing artist for grouped split: {song_id}")
        expected = str(track.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise ValueError(f"Missing valid expected audio sha256 for {song_id}")
        license_url = str(track.get("license_url", ""))
        if "/by-nd/" in license_url or "/by-nc-nd/" in license_url:
            raise ValueError(f"NoDerivatives track requires a separate authorization: {song_id}")
        if not track.get("license") or not license_url:
            raise ValueError(f"Missing per-track license and license_url for {song_id}")
    return source, tracks


def _read_annotations(path: Path, selected_ids: set[str]) -> tuple[dict[str, list[dict]], dict]:
    grouped = defaultdict(list)
    counts = Counter()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"song_id", "language", "subset", "start", "end", "text"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Annotation CSV is missing fields: {sorted(required - set(reader.fieldnames or []))}")
        for row_number, row in enumerate(reader, 2):
            counts["source_rows"] += 1
            if row["song_id"] not in selected_ids:
                continue
            if row["language"].strip().casefold() not in ("pt", "pt-br", "portuguese", "português"):
                raise ValueError(f"Selected annotation is not Portuguese at CSV row {row_number}")
            try:
                start, end = float(row["start"]), float(row["end"])
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid timestamp at CSV row {row_number}") from error
            if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
                raise ValueError(f"Expected finite 0 <= start < end at CSV row {row_number}")
            text = row["text"].strip()
            if not text:
                raise ValueError(f"Empty lyric line at CSV row {row_number}")
            grouped[row["song_id"]].append({"start": start, "end": end, "text": text,
                                          "source_row": row_number, "source_subset": row["subset"]})
            counts["selected_rows"] += 1
    for song_id in selected_ids:
        if not grouped[song_id]:
            raise ValueError(f"No line annotations found for selected audio {song_id}")
        grouped[song_id].sort(key=lambda line: (line["start"], line["end"], line["source_row"]))
    return dict(grouped), dict(counts)


def _groups(lines: list[dict], *, max_seconds: float, max_gap: float, padding: float) -> tuple[list[list[dict]], list[dict], list[dict]]:
    clusters, overlaps = [], []
    for line in lines:
        if clusters and line["start"] < max(item["end"] for item in clusters[-1]):
            previous_end = max(item["end"] for item in clusters[-1])
            overlaps.append({"previous_rows": [item["source_row"] for item in clusters[-1]],
                             "next_row": line["source_row"], "seconds": previous_end - line["start"]})
            clusters[-1].append(line)
        else:
            clusters.append([line])
    groups, skipped, current = [], [], []
    # Reserve both pads. This bound also covers neighboring annotations that
    # restrict a pad. Never split one line or an overlapping cluster in half.
    content_limit = max_seconds - 2 * padding
    for cluster in clusters:
        first, last = cluster[0]["start"], max(line["end"] for line in cluster)
        if last - first > content_limit:
            if current:
                groups.append(current)
                current = []
            skipped.append({"source_rows": [line["source_row"] for line in cluster],
                            "start": first, "end": last, "reason": "one line/overlapping cluster exceeds crop limit"})
            continue
        if current and (first - max(line["end"] for line in current) > max_gap
                        or last - current[0]["start"] > content_limit):
            groups.append(current)
            current = []
        current.extend(cluster)
    if current:
        groups.append(current)
    return groups, overlaps, skipped


def _caption(track: dict) -> str:
    if track.get("caption"):
        return str(track["caption"])
    categories = defaultdict(list)
    for tag in track.get("tags", []):
        if "---" in str(tag):
            category, label = str(tag).split("---", 1)
            categories[category].append(label)
    pieces = ["Music with Portuguese lyrics."]
    for key, label in (("genre", "Genres"), ("instrument", "Instruments"), ("mood/theme", "Mood and themes")):
        if categories[key]:
            pieces.append(f"{label}: {', '.join(categories[key])}.")
    return " ".join(pieces)


def _split_tracks(tracks: list[dict], fraction: float, seed: int) -> dict[str, str]:
    artists = sorted({normalize_artist(track["artist"]) for track in tracks})
    random.Random(seed).shuffle(artists)
    count = min(round(len(artists) * fraction), len(artists) - 1)
    if fraction and len(artists) > 1:
        count = max(1, min(count, len(artists) - 1))
    heldout = set(artists[:count])
    return {track["song_id"]: "validation" if normalize_artist(track["artist"]) in heldout else "train" for track in tracks}


def _write_audio(path: Path, audio, sample_rate: int) -> tuple[Path, str, str, dict]:
    import numpy as np
    import soundfile as sf
    if not np.isfinite(audio).all():
        raise ValueError(f"Nonfinite decoded audio for {path.name}")
    peak = float(np.max(np.abs(audio), initial=0))
    floating = peak > 1
    audio_format, subtype = ("WAV", "FLOAT") if floating else ("FLAC", "PCM_24")
    if floating:
        path = _inside(path.parent.resolve(), path.with_suffix(".wav").name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".writing", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        sf.write(temporary, audio, sample_rate, format=audio_format, subtype=subtype)
        info = sf.info(temporary)
        if info.frames != len(audio) or info.samplerate != sample_rate or info.channels != audio.shape[1] or info.subtype != subtype:
            raise OSError(f"Encoded audio metadata does not match the source crop: {path.name}")
        digest = _sha256(temporary)
        if path.exists():
            if not path.is_file():
                raise FileExistsError(f"Existing crop differs; refusing to overwrite: {path}")
            existing_digest = _sha256(path)
            if existing_digest != digest:
                # libsndfile's float WAV PEAK chunk includes a write timestamp.
                # Reuse an existing file only when its encoding and decoded
                # samples equal the freshly encoded crop, without rewriting it.
                existing_info = sf.info(path)
                if (existing_info.format, existing_info.subtype, existing_info.frames, existing_info.channels, existing_info.samplerate) != (info.format, info.subtype, info.frames, info.channels, info.samplerate):
                    raise FileExistsError(f"Existing crop differs; refusing to overwrite: {path}")
                existing_audio, _ = sf.read(path, dtype="float64", always_2d=True)
                encoded_audio, _ = sf.read(temporary, dtype="float64", always_2d=True)
                if not np.array_equal(existing_audio, encoded_audio):
                    raise FileExistsError(f"Existing crop differs; refusing to overwrite: {path}")
                digest = existing_digest
            action = "reused"
        else:
            temporary.replace(path)
            action = "written"
        return path, digest, action, {"audio_format": audio_format, "audio_subtype": subtype,
                                     "peak": peak, "samples_outside_pcm_range": int((np.abs(audio) > 1).sum()),
                                     "clipped_samples": 0, "gain_applied": False,
                                     "float_fallback": floating}
    finally:
        temporary.unlink(missing_ok=True)


def import_muljam_pt(annotations, metadata, output, *, audio_root=None, max_seconds=30.0,
                     max_gap=5.0, padding=0.25, validation_fraction=0.2, seed=42, progress=None) -> dict:
    import numpy as np
    import soundfile as sf
    options = [float(max_seconds), float(max_gap), float(padding), float(validation_fraction)]
    if not all(math.isfinite(value) for value in options) or max_seconds <= 0 or max_gap < 0 or padding < 0 or 2 * padding >= max_seconds or not 0 <= validation_fraction < 1:
        raise ValueError("Invalid crop/split options")
    annotation_path, metadata_path = Path(annotations).expanduser().resolve(), Path(metadata).expanduser().resolve()
    destination = _root(output)
    source, tracks = _load_tracks(metadata_path)
    annotation_hash = _sha256(annotation_path)
    declared_annotations = source.get("annotations_source", {})
    if isinstance(declared_annotations, dict) and declared_annotations.get("sha256"):
        expected = str(declared_annotations["sha256"])
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected) or expected.casefold() != annotation_hash:
            raise ValueError("Source annotation CSV SHA256 mismatch")
    selected = {track["song_id"] for track in tracks}
    annotations_by_song, counts = _read_annotations(annotation_path, selected)
    splits = _split_tracks(tracks, validation_fraction, seed)
    prepared = []
    # Validate all input audio and annotations before producing any crops.
    for track in tracks:
        audio_path = _audio_path(track, metadata_path, audio_root)
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        if track.get("size_bytes") is not None and audio_path.stat().st_size != int(track["size_bytes"]):
            raise ValueError(f"Source audio size mismatch: {track['song_id']}")
        digest = _sha256(audio_path)
        if digest.lower() != track["sha256"].lower():
            raise ValueError(f"Source audio SHA256 mismatch: {track['song_id']}")
        info = sf.info(audio_path)
        duration = info.frames / info.samplerate
        lines = annotations_by_song[track["song_id"]]
        for line in lines:
            if line["end"] > duration:
                raise ValueError(f"Timestamp exceeds decoded audio duration for {track['song_id']} at CSV row {line['source_row']}: {line['end']} > {duration}")
        groups, overlaps, skipped = _groups(lines, max_seconds=max_seconds, max_gap=max_gap, padding=padding)
        prepared.append((track, audio_path, info, lines, groups, overlaps, skipped))
    samples, originals, issues = [], [], []
    counts.update(tracks=len(tracks), source_sha256_verified=len(tracks), crops_written=0, crops_reused=0)
    for index, (track, audio_path, info, lines, groups, overlaps, skipped) in enumerate(prepared, 1):
        audio, sample_rate = sf.read(audio_path, dtype="float64", always_2d=True)
        if len(audio) != info.frames:
            raise OSError(f"Decoded frame count differs from source header for {track['song_id']}")
        artist_key = normalize_artist(track["artist"])
        artist_id = "artist:" + hashlib.sha256(artist_key.encode("utf-8")).hexdigest()[:24]
        shared = {"song_id": track["song_id"], "track_id": str(track.get("track_id", track["song_id"].removeprefix("muljam_"))),
                  "artist": track["artist"], "artist_id": artist_id, "artist_key": artist_key,
                  "source_metadata": {**track.get("source_metadata", {}), "artist_id": track.get("artist_id")},
                  "source_artist_id": track.get("artist_id"),
                  "title": track.get("title", ""), "language": "pt", "tags": track.get("tags", []),
                  "caption": _caption(track), "holdout_group": artist_id, "split": splits[track["song_id"]],
                  "license": track["license"], "license_url": track["license_url"], "track_url": track.get("track_url"),
                  "source_url": track.get("source_url"), "source_audio_sha256": track["sha256"],
                  "source_audio_path": str(audio_path), "source_audio_encoding": info.format,
                  "source_peak": float(np.max(np.abs(audio), initial=0)),
                  "source_audio_lossy": audio_path.suffix.casefold() in (".mp3", ".ogg", ".m4a", ".opus"),
                  "timestamp_source": "MusicAI Interspeech 2025 automatic line alignment",
                  "timestamps_human_verified": False, "sampling_rate": sample_rate, "channels": audio.shape[1]}
        originals.append({**track, **shared, "id": track["song_id"], "audio_path": str(audio_path),
                          "duration": len(audio) / sample_rate, "lyrics": "\n".join(line["text"] for line in lines),
                          "lines": lines, "sha256": track["sha256"]})
        for overlap in overlaps:
            issues.append({"song_id": track["song_id"], "kind": "overlapping_automatic_lines", **overlap})
        for skip in skipped:
            issues.append({"song_id": track["song_id"], "kind": "skipped_long_annotation", **skip})
        for segment_number, group in enumerate(groups, 1):
            first, last = group[0]["start"], max(line["end"] for line in group)
            row_ids = {line["source_row"] for line in group}
            preceding = [line["end"] for line in lines if line["source_row"] not in row_ids and line["end"] <= first]
            following = [line["start"] for line in lines if line["source_row"] not in row_ids and line["start"] >= last]
            start_frame = max(0, math.floor((first - padding) * sample_rate), math.ceil(max(preceding, default=0) * sample_rate))
            end_frame = min(len(audio), math.ceil((last + padding) * sample_rate), math.floor(min(following, default=len(audio) / sample_rate) * sample_rate))
            if not 0 <= start_frame < end_frame <= len(audio) or end_frame - start_frame > math.floor(max_seconds * sample_rate):
                raise ValueError(f"Invalid crop frame boundaries for {track['song_id']} segment {segment_number}")
            segment_id = f"{track['song_id']}_{segment_number:03d}"
            target = _inside(destination, f"audio/{segment_id}.flac")
            target, crop_sha256, action, encoding = _write_audio(target, audio[start_frame:end_frame], sample_rate)
            counts[f"crops_{action}"] += 1
            lyrics = "\n".join(line["text"] for line in group)
            condition = hashlib.sha256(json.dumps([shared["caption"], lyrics], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            samples.append({**shared, "id": segment_id, "audio_path": str(target), "sha256": crop_sha256,
                            "duration": (end_frame - start_frame) / sample_rate, **encoding,
                            "lyrics": lyrics, "group_id": f"{segment_id}:{condition[:24]}", "condition_sha256": condition,
                            "crop_start": start_frame / sample_rate, "crop_end": end_frame / sample_rate,
                            "crop_start_frame": start_frame, "crop_end_frame": end_frame,
                            "lines": [{**line, "segment_start": line["start"] - start_frame / sample_rate,
                                       "segment_end": line["end"] - start_frame / sample_rate} for line in group]})
        if progress:
            progress({"event": "import", "completed": index, "total": len(tracks), "song_id": track["song_id"], "segments": len(samples)})
    if not samples:
        raise ValueError("No complete annotation groups fit the segment duration limit")
    counts.update(segments=len(samples), lines_in_segments=sum(len(sample["lines"]) for sample in samples),
                  overlap_events=sum(issue["kind"] == "overlapping_automatic_lines" for issue in issues),
                  skipped_long_annotations=sum(issue["kind"] == "skipped_long_annotation" for issue in issues),
                  flac_pcm24_segments=sum(sample["audio_format"] == "FLAC" for sample in samples),
                  wav_float32_segments=sum(sample["float_fallback"] for sample in samples), clipped_samples=0)
    origin = {**source, "annotations_file": str(annotation_path), "annotations_sha256": annotation_hash,
              "audio_metadata_file": str(metadata_path), "audio_metadata_sha256": _sha256(metadata_path),
              "annotations": "Automatically aligned lines; no manual verification or fabricated word timings",
              "audio_processing": "Decode original audio and crop complete line groups. Store FLAC PCM24 with normal integer quantization, or WAV FLOAT32 when decoded peaks exceed integer PCM range. No gain, clipping, resampling or channel changes.",
              "lossy_sources_restored": False, "preferences_provided": False,
              "split_policy": "artist names normalized with NFKD, accents removed, casefold and whitespace normalization"}
    parameters = {"max_seconds": max_seconds, "max_gap": max_gap, "padding": padding,
                  "validation_fraction": validation_fraction, "seed": seed}
    manifest, full_manifest = destination / "dataset.json", destination / "full_tracks.json"
    info_payload = {"origin": origin, "parameters": parameters, "counts": counts,
                    "split_tracks": dict(Counter(track["split"] for track in originals)),
                    "split_segments": dict(Counter(sample["split"] for sample in samples)), "issues": issues,
                    "warnings": ["Automatic lyric timings may contain errors and require listening checks.",
                                 "Output retains the bandwidth and artifacts of the original lossy MP3.",
                                 "WAV FLOAT32 preserves decoded peaks above 1.0 instead of clipping or normalizing them.",
                                 "Keep this normalized artist grouping when combining with other datasets; do not resplit individual segments."]}
    _write_json(manifest, {"version": 1, "samples": samples, "metadata": info_payload})
    _write_json(full_manifest, {"version": 1, "samples": originals, "metadata": info_payload})
    report = {"manifest": str(manifest), "full_tracks_manifest": str(full_manifest), **info_payload}
    _write_json(destination / "import_report.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", required=True, help="Local dali_muljam_interspeech25.csv")
    parser.add_argument("--metadata", required=True, help="Local audio_metadata.json with verified hashes and track licenses")
    parser.add_argument("--audio-root", help="Override source audio folder; otherwise resolve paths relative to metadata")
    parser.add_argument("--output", required=True, help="Output directory for FLAC PCM24 / WAV FLOAT32 crops and manifests")
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--max-gap", type=float, default=5.0, help="Split between lyric lines separated by a longer pause")
    parser.add_argument("--padding", type=float, default=0.25, help="Padding bounded by the neighboring lyric annotations")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    report = import_muljam_pt(args.annotations, args.metadata, args.output, audio_root=args.audio_root,
                             max_seconds=args.max_seconds, max_gap=args.max_gap, padding=args.padding,
                             validation_fraction=args.validation_fraction, seed=args.seed,
                             progress=lambda item: print(json.dumps(item, ensure_ascii=True), flush=True))
    print(json.dumps({key: report[key] for key in ("manifest", "full_tracks_manifest", "counts", "split_tracks", "split_segments")}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
