"""Offline Muse archive/JSONL import into JK-Step's sample manifest.

Example:
  python scripts/import_muse.py --metadata train_en.jsonl --archive en_part01_of_35.tar \
    --audio-root datasets/muse/audio --output datasets/muse/imported --sidecars

No audio is generated, transcoded, cropped, scored, or preprocessed. Muse's
style_sim is retained as source metadata, never presented as musical quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import statistics
import tarfile
import tempfile


SOURCE_REPO = "bolshyC/Muse"
SOURCE_REVISION = "b1bf3bf906daab3a896e14f6dea58cc295848452"
SOURCE_CARD = f"https://huggingface.co/datasets/{SOURCE_REPO}/blob/{SOURCE_REVISION}/README.md"


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def _safe_relative(name: str) -> PurePosixPath:
    # Reject Windows paths/ADS/device aliases even when importing on Linux.
    normalized = name.replace("\\","/")
    path = PurePosixPath(normalized)
    if not normalized or "\0" in normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive/metadata path: {name!r}")
    for part in path.parts:
        if ":" in part or part.endswith((" ",".")) or re.match(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)",part,re.I):
            raise ValueError(f"Unsafe Windows path component: {part!r}")
    return path


def _root_path(value) -> Path:
    requested = Path(value).expanduser().absolute()
    for candidate in (requested,*requested.parents):
        if candidate.is_symlink() or (hasattr(candidate,"is_junction") and candidate.is_junction()):
            raise ValueError(f"Refusing an output/audio root through a link: {candidate}")
    return requested.resolve()


def _target_inside(root: Path, relative: str) -> Path:
    path = root.joinpath(*_safe_relative(relative).parts)
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path escapes the output/audio root: {relative!r}")
    # Existing symlinks/junctions within the root are also not followed.
    for candidate in (path,*path.parents):
        if candidate == root.parent:
            break
        if candidate.is_symlink() or (hasattr(candidate,"is_junction") and candidate.is_junction()):
            raise ValueError(f"Refusing existing link: {candidate}")
    return path


def _json_write(path: Path, data) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=path.parent,suffix=".writing",delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(data,handle,ensure_ascii=False,indent=2,allow_nan=False)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _bytes_write(path: Path, lines) -> None:
    """Preserve selected source JSONL rows byte for byte, atomically."""
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile("wb",dir=path.parent,suffix=".writing",delete=False) as handle:
        temporary = Path(handle.name)
        for line in lines:
            handle.write(line)
            if not line.endswith(b"\n"):
                handle.write(b"\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _audio_info(audio: Path) -> dict:
    """Read MP3 headers and fingerprint original bytes; do not decode/alter audio."""
    import soundfile as sf
    from mutagen import MutagenError
    from mutagen.mp3 import MP3
    try:
        info, mp3 = sf.info(str(audio)),MP3(str(audio)).info
    except MutagenError as error:
        raise ValueError(f"Invalid MP3 header: {audio}") from error
    duration = float(info.duration)
    if duration <= 0 or not math.isfinite(duration):
        raise ValueError(f"Invalid audio duration: {audio}")
    return {"audio_bytes":audio.stat().st_size,"audio_sha256":_sha256(audio),
            "duration_seconds":duration,"sample_rate":int(info.samplerate),
            "channels":int(info.channels),"bitrate_bps":int(mp3.bitrate),
            "mp3_header_duration_seconds":float(mp3.length),
            "average_file_bitrate_bps":round(audio.stat().st_size * 8 / duration),
            "audio_subtype":info.subtype}


def _audio_report(samples: list[dict]) -> dict:
    durations,bitrate = [],Counter()
    rate,channels,issues,sections = Counter(),Counter(),[],Counter()
    for sample in samples:
        duration = sample.get("duration_seconds")
        if duration is not None:
            durations.append(duration)
            bitrate[str(sample["bitrate_bps"])] += 1
            rate[str(sample["sample_rate"])] += 1
            channels[str(sample["channels"])] += 1
        previous_end = None
        for section in sample["sections"]:
            sections["total"] += 1
            has_words = bool(section.get("text","").strip())
            sections["with_lyrics" if has_words else "without_lyrics"] += 1
            if "startS" not in section or "endS" not in section:
                sections["without_timestamps"] += 1
                continue
            start,end = float(section["startS"]),float(section["endS"])
            sections["with_timestamps"] += 1
            if has_words and 24 <= end-start <= 35:
                sections["lyric_sections_24_to_35_seconds"] += 1
            if previous_end is not None and start < previous_end - .02:
                sections["overlap_over_20ms"] += 1
                if len(issues)<100:
                    issues.append({"id":sample["id"],"section":section.get("section"),
                                   "issue":"overlapping_source_sections","overlap_seconds":previous_end-start})
            if duration is not None and end > duration + .1:
                sections["end_beyond_audio_over_100ms"] += 1
                if len(issues)<100:
                    issues.append({"id":sample["id"],"section":section.get("section"),
                                   "issue":"section_end_beyond_audio","endS":end,"duration_seconds":duration})
            previous_end=end
    return {"inspected_audio":len(durations),"sample_rates":dict(rate),"channels":dict(channels),
            "bitrate_bps":dict(bitrate),"sections":dict(sections),"issues":issues,
            "duration_seconds":{"min":min(durations),"median":statistics.median(durations),
                                "max":max(durations),"total":sum(durations)} if durations else {},
            "alignment_granularity":"section; source supplies no word-level or line-level timestamps"}


def extract_archives(archives, audio_root, *, language="en", progress=None) -> list[dict]:
    """Extract only regular MP3 files; reject traversal, links and special files.

    All members are checked before extracting an archive. Files are written
    atomically; existing complete files are reused only if their bytes match
    the archive member. Nothing outside audio_root is changed.
    """
    root = _root_path(audio_root)
    root.mkdir(parents=True,exist_ok=True)
    prefix = "suno_en_songs" if language=="en" else "suno_cn_songs"
    reports = []
    for source in map(Path,archives):
        source = source.expanduser().resolve()
        report = {"archive":str(source),"sha256":_sha256(source),"extracted":0,"reused":0,"ignored":0}
        with tarfile.open(source,"r:*") as archive:
            selected, seen = [], set()
            for member in archive.getmembers():
                relative = _safe_relative(member.name)
                if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                    raise ValueError(f"Archive links/special files are forbidden: {member.name!r}")
                if member.isdir():
                    continue
                if relative.suffix.lower() != ".mp3":
                    report["ignored"] += 1
                    continue
                parts = relative.parts[1:] if relative.parts[0] == prefix else relative.parts
                target = _target_inside(root,"/".join((prefix,*parts)))
                identity = str(target).casefold()
                if identity in seen:
                    raise ValueError(f"Duplicate archive output path: {member.name!r}")
                seen.add(identity)
                selected.append((member,target))
            for index,(member,target) in enumerate(selected,1):
                target.parent.mkdir(parents=True,exist_ok=True)
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Cannot read archive member: {member.name}")
                with stream:
                    if target.exists():
                        if not target.is_file() or target.stat().st_size != member.size or hashlib.file_digest(stream,"sha256").hexdigest() != _sha256(target):
                            raise FileExistsError(f"Existing audio differs; refusing to overwrite: {target}")
                        report["reused"] += 1
                    else:
                        with tempfile.NamedTemporaryFile("wb",dir=target.parent,suffix=".writing",delete=False) as handle:
                            temporary = Path(handle.name)
                            shutil.copyfileobj(stream,handle,length=1024*1024)
                        try:
                            if temporary.stat().st_size != member.size:
                                raise OSError(f"Incomplete archive member: {member.name}")
                            temporary.replace(target)
                        finally:
                            temporary.unlink(missing_ok=True)
                        report["extracted"] += 1
                if progress and (index % 100 == 0 or index == len(selected)):
                    progress({"event":"extract","archive":source.name,"completed":index,"total":len(selected)})
        reports.append(report)
    return reports


def _sample(record: dict, audio: Path, language: str) -> dict:
    song_id = record.get("song_id")
    style = record.get("style","")
    sections = record.get("sections",[])
    if not isinstance(song_id,str) or not song_id or not isinstance(style,str) or not isinstance(sections,list):
        raise ValueError("Expected song_id/style strings and a sections list")
    blocks, ends, has_words = [], [], False
    for section in sections:
        label, text = section.get("section",""), section.get("text","")
        if not isinstance(label,str) or not isinstance(text,str):
            raise ValueError("Section labels/text must be strings")
        # Keep source section order and every word/repetition/punctuation. Do
        # not replace typographic apostrophes or invent line-level alignment.
        blocks.append((f"[{label}]\n" if label else "") + text)
        has_words |= bool(text.strip())
        if "startS" in section and "endS" in section:
            start, end = float(section["startS"]),float(section["endS"])
            if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start <= end:
                raise ValueError("Invalid section startS/endS")
            ends.append(end)
    lyrics = "\n\n".join(blocks) or "[Instrumental]"
    condition_hash = hashlib.sha256(json.dumps([style,lyrics],ensure_ascii=False,separators=(",",":")).encode("utf-8")).hexdigest()
    return {"id":f"{song_id}_{record.get('track_index',0)}","audio_path":str(audio),
            "caption":style,"style":style,"lyrics":lyrics,"language":language,
            "tags":[tag.strip() for tag in style.split(",") if tag.strip()],
            "song_id":song_id,"track_index":record.get("track_index",0),
            "group_id":f"{song_id}:{condition_hash[:24]}","holdout_group":song_id,
            "condition_sha256":condition_hash,"sections":sections,"is_instrumental":not has_words,
            "annotated_end_seconds":max(ends) if ends else None,
            "style_sim":record.get("style_sim"),"error":record.get("error"),
            "source_audio_path":record.get("audio_path"),"audio_format":"mp3","lossy_audio":True}


def import_muse(metadata, audio_root, output, *, archives=(), language="en", sidecars=False,
                include_errors=False, max_samples=None, revision=SOURCE_REVISION,
                inspect_audio=False, progress=None) -> dict:
    """Build dataset.json plus import_report.json from audio already present."""
    if language not in ("en","zh"):
        raise ValueError("Muse provides EN/CN; language must be en or zh")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive")
    metadata = Path(metadata).expanduser().resolve()
    root, destination = _root_path(audio_root),_root_path(output)
    extraction = extract_archives(archives,root,language=language,progress=progress)
    counts, samples, selected_lines, seen, conflicted = Counter(),{}, {},{},set()
    metadata_hash = hashlib.sha256()
    issues = []
    with metadata.open("rb") as handle:
        for number,line in enumerate(handle,1):
            metadata_hash.update(line)
            if not line.strip():
                continue
            counts["metadata_rows"] += 1
            try:
                record = json.loads(line)
                if record.get("error"):
                    counts["source_error_rows"] += 1
                    if not include_errors:
                        counts["skipped_error_rows"] += 1
                        continue
                audio = _target_inside(root,record["audio_path"])
                if not audio.is_file():
                    counts["audio_not_present"] += 1
                    continue
                sample = _sample(record,audio,language)
                # Enforce JSON-compatible finite source fields as well.
                identity = str(audio).casefold()
                digest = hashlib.sha256(json.dumps(record,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
                if identity in seen:
                    counts["duplicate_audio_rows"] += 1
                    if seen[identity] != digest:
                        counts["conflicting_audio_rows"] += 1
                        conflicted.add(identity)
                        samples.pop(identity,None)
                        selected_lines.pop(identity,None)
                    continue
                seen[identity] = digest
                if identity not in conflicted and (max_samples is None or len(samples) < max_samples):
                    samples[identity] = sample
                    selected_lines[identity] = line
            except (ValueError,KeyError,TypeError,AttributeError) as error:
                counts["invalid_rows"] += 1
                if len(issues) < 100:
                    issues.append({"line":number,"error":str(error)})
            if progress and number % 10000 == 0:
                progress({"event":"metadata","rows":number,"matched_audio":len(samples)})
    rows = list(samples.values())
    if inspect_audio:
        for index,sample in enumerate(rows,1):
            try:
                sample.update(_audio_info(Path(sample["audio_path"])))
            except (ValueError,OSError,RuntimeError) as error:
                counts["audio_inspection_failures"] += 1
                if len(issues)<100:
                    issues.append({"id":sample["id"],"error":str(error)})
                # Keep the source record for provenance, mark it unsuitable
                # for downstream training rather than silently declaring it valid.
                sample["audio_inspection_error"] = str(error)
            if progress and (index % 100 == 0 or index == len(rows)):
                progress({"event":"inspect_audio","completed":index,"total":len(rows)})
    songs, groups = Counter(s["song_id"] for s in rows),Counter(s["group_id"] for s in rows)
    counts.update(imported_samples=len(rows),unique_songs=len(songs),conditioning_groups=len(groups),
                  songs_with_multiple_tracks=sum(n>1 for n in songs.values()),
                  groups_with_pair_candidates=sum(n>1 for n in groups.values()))
    origin = {"repo":SOURCE_REPO,"revision":revision,"url":f"https://huggingface.co/datasets/{SOURCE_REPO}",
              "license_declared_by_card":"MIT","license_card":SOURCE_CARD.replace(SOURCE_REVISION,revision),
              "metadata_file":str(metadata),"metadata_sha256":metadata_hash.hexdigest(),"archives":extraction,
              "synthetic_audio_provider":"SunoV5","audio_encoding":"MP3 (lossy)",
              "annotations":"Automatically generated; section timestamps preserved without manual verification",
              "style_sim_is_musical_quality":False,"preferences_provided":False}
    subset = destination / "selected_metadata.jsonl"
    _bytes_write(subset,selected_lines.values())
    origin.update(selected_metadata_file=str(subset),selected_metadata_sha256=_sha256(subset),
                  selected_metadata_rows=len(selected_lines))
    _json_write(destination / "dataset.json",{"version":1,"samples":rows,"metadata":{"origin":origin}})
    report = {"manifest":str(destination / "dataset.json"),"counts":dict(counts),"origin":origin,
              "audio":_audio_report(rows),"issues":issues}
    _json_write(destination / "import_report.json",report)
    if sidecars:
        for sample in rows:
            audio = Path(sample["audio_path"])
            target = _target_inside(root,str(audio.relative_to(root).with_suffix(".json")))
            if target.exists():
                existing = json.loads(target.read_text(encoding="utf-8"))
                if existing != sample:
                    raise FileExistsError(f"Refusing to replace an existing different sidecar: {target}")
            else:
                _json_write(target,sample)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata",required=True,help="Local Muse train_en.jsonl/train_cn.jsonl")
    parser.add_argument("--audio-root",required=True,help="Root containing suno_en_songs/suno_cn_songs")
    parser.add_argument("--output",required=True,help="Directory for dataset.json and import_report.json")
    parser.add_argument("--archive",action="append",default=[],help="Optional local tar; may be repeated")
    parser.add_argument("--language",choices=("en","zh"),default="en")
    parser.add_argument("--sidecars",action="store_true",help="Also write non-overwriting JSON audio sidecars")
    parser.add_argument("--include-errors",action="store_true",help="Include source rows marked with error")
    parser.add_argument("--max-samples",type=int)
    parser.add_argument("--inspect-audio",action="store_true",
                        help="Verify MP3 headers, original SHA256, durations/rates/channels/bitrate (soundfile + mutagen)")
    parser.add_argument("--revision",default=SOURCE_REVISION,help="HF revision associated with your local inputs")
    args = parser.parse_args(argv)
    report = import_muse(args.metadata,args.audio_root,args.output,archives=args.archive,language=args.language,
                         sidecars=args.sidecars,include_errors=args.include_errors,max_samples=args.max_samples,
                         revision=args.revision,inspect_audio=args.inspect_audio,
                         progress=lambda event:print(json.dumps(event,ensure_ascii=True),flush=True))
    print(json.dumps(report,ensure_ascii=True,allow_nan=False),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
