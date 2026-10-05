"""Import downloaded Portuguese JamendoLyrics songs into JK-Step, offline.

python scripts/import_jamendolyrics_pt.py --root datasets/downloads/jamendolyrics_pt \
    --output datasets/jamendolyrics_pt --license-filter all

The full recording and the original lyrics are preserved. No word/line timing,
quality scores, BPM, instruments, vocalist attributes or license versions are
invented. Filters use source declarations; they are not rights clearance.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import tempfile
import unicodedata


REPO = "Felipehonorato/pt_it_jamendolyrics"
REVISION = "d3a53a2f145abf63351d658c4cbda6453d1c0dfc"
METADATA_GIT_BLOB = "68454f10b9df073fcc3d4c6822b404647ff3b5c1"
# Expected bytes/SHA256 from the pinned HF LFS pointers, not audio-derived scores.
EXPECTED_AUDIO = {
    "A_Feira_-_Pablo_Castro.mp3": (7504896,"319e040a958e65ffbf7de9270f816c743b977d2c8fc216c4596606b87282e3d2"),
    "Becos_de_Rua_-_pacto.mp3": (5950464,"ff20598413020b8abab388e6c1f8b6814958625ba3da4e789b64e227d3352879"),
    "Cristian_Romanus_-_Insensatez_-_Cristian_Romanus_(3).mp3": (3672064,"084020625cd59627920d0f2a0bd1fada695827919d6af277e31e3ea2cd9deb08"),
    "Curvas_-_Tulio_Borges.mp3": (5612032,"9170d1f29c95286059c30fbbf54cf3487540b476efa0f2d10ed4d6175ec64f7f"),
    "Dor_do_Mundo_-_Nicolas_Muma_Farruggia.mp3": (5573128,"b8f88d758198e32e7c66da06058be87543692ce1cdf3e56ef3619313bba7914c"),
    "E_Quando_a_Morte_Cruza_-_pangenianos.mp3": (9189712,"9170261eabf6072a180806d27f04ba0d021b7fe317ca5bfce4b21bf11513466d"),
    "Elefantes_-_Safari_de_Saturno.mp3": (3748937,"cd6fda0b2e12f5cf8b4bda5800abb40d37e2efb5ec84666ca1c82469a08739c4"),
    "Enxerida_no_contexto_-_Tulio_Borges.mp3": (6297088,"6ca8fdb8d83be416651529410488b73217fb7d2e07e135670164291d31403e82"),
    "Hoje_-_Taro.mp3": (4759552,"8ff9f9714ec71ebbcab60f4082f83cd1b4f6764d0090dd33ab4f4f0895e5ccc5"),
    "Impossível_Te_Esquecer-Di_Santos_-_DiSantos.mp3": (5939712,"eb33ac11996584bab382540ede1af42e2f85f75f3bdc852fb7f989277a5c78c2"),
    "Luar_do_Pontal_-_Distintivo_Blue.mp3": (5643264,"07d4bc41edd20c96ca2a219d34f53e8e3ba18b4da57a809e6d9a7e5de9547825"),
    "Massemba_-_Nicolas_Muma_Farruggia.mp3": (5427379,"3cf52fd78b65938cd94849bc6a680bd2e7df38ebc21decb089a8283426dac900"),
    "Morada_-_Confraria.mp3": (5548717,"86d01389ae14fba1cc677bd25eded9ff6f2c5a6dfb89dd25989d88056648c1f9"),
    "Morena_Beleza_-_Suinga.mp3": (3897388,"acd2d5e081d5c335e773ddd3c79624a8355ada81a224ee510d9503894bcd2e09"),
    "Música_-_BlueExcess.mp3": (4197888,"b3b6730a348f998c8da7487c534bf8456436713cc27965272d6a8a99c8a00276"),
    "Perto_Daqui_-_Pablo_Castro.mp3": (4733440,"2d3018b3784bbe73c4c61efe53ed74ff6e70ce621d01a9b5cd04a9130e628726"),
    "Portugues_-_pacto.mp3": (5210112,"36698ebac95fdef947739f881bfac4d8d9474727aa8bf668a83b2a4b95dff37c"),
    "Preciso_voar_-_Misturantes.mp3": (4012367,"c1749afbb17f2641a73fd7987d7841ea1bbfc21e81c916eca05d3af0b6a4d631"),
    "Um_novo_amanhecer_-_Raphael_Souza.mp3": (4016439,"161331ce2c7a243ca22e33992b4c40ed384f007a9ac13cf7d93a045d53a85ae4"),
    "Você_Roubou_o_Meu_Pendrive_-_Distintivo_Blue.mp3": (4687360,"f9b39a1adb6e85861aaa44a2738215060a87c7da1913ddd8d63068ec1f1995b6"),
}
GENRES = {"bossa nova":"bossa nova","jazz":"jazz","rock":"rock","indie":"indie music",
          "blues":"blues","hip-hop":"hip-hop","pop":"pop","folk":"folk","metal":"metal","axé":"axé music"}


def normalize_license(value: str) -> str | None:
    """Normalize CC SA-BY / SA-NC-BY token ordering without adding a version."""
    tokens = set(re.findall(r"[A-Z]+",str(value).upper()))
    if "CC" not in tokens or "BY" not in tokens or tokens - {"CC","BY","SA","NC","ND"}:
        return None
    if "SA" in tokens and "ND" in tokens:
        return None
    return "-".join(["CC","BY",*(part for part in ("NC","ND","SA") if part in tokens)])


def _hash_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def _safe_child(root: Path, relative: str) -> Path:
    normalized = str(relative).replace("\\","/")
    parts = PurePosixPath(normalized)
    if not normalized or parts.is_absolute() or ".." in parts.parts or "\0" in normalized or any(":" in part for part in parts.parts):
        raise ValueError(f"Unsafe source path: {relative!r}")
    target = root.joinpath(*parts.parts)
    if not target.resolve().is_relative_to(root):
        raise ValueError(f"Source path escapes dataset root: {relative!r}")
    if target.is_symlink():
        raise ValueError(f"Refusing source audio symlink: {target}")
    return target


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=path.parent,suffix=".writing",delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload,handle,ensure_ascii=False,indent=2,allow_nan=False)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def import_jamendolyrics_pt(root, output, *, license_filter="no-nc", verify_source=True, progress=None) -> dict:
    """Validate local full songs and write a manifest/report; uses no network/GPU."""
    import soundfile as sf
    if license_filter not in ("all","no-nc","commercial-compatible"):
        raise ValueError("license_filter must be all or no-nc (commercial-compatible is an alias)")
    root, output = Path(root).expanduser().resolve(),Path(output).expanduser().resolve()
    metadata = root / "subsets" / "pt" / "metadata.jsonl"
    if not metadata.is_file():
        metadata = root / "metadata.jsonl"
    raw = metadata.read_bytes()
    blob_hash = hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()
    if verify_source and blob_hash != METADATA_GIT_BLOB:
        raise ValueError("Metadata differs from the pinned Portuguese release. For an intentional custom copy use --skip-origin-check.")
    text = raw.decode("utf-8-sig",errors="strict")
    if "\ufffd" in text:
        raise ValueError("Metadata contains replacement characters; restore the original UTF-8 source")
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    samples, inspection, skipped, counts = [],[],[],Counter()
    seen = set()
    for index,record in enumerate(records,1):
        if record.get("language") != "pt":
            counts["non_portuguese_rows"] += 1
            continue
        audio = _safe_child(metadata.parent,record["file_name"])
        if not audio.is_file():
            raise FileNotFoundError(audio)
        identity = str(audio).casefold()
        if identity in seen:
            raise ValueError(f"Duplicate source audio record: {audio}")
        seen.add(identity)
        size, digest = audio.stat().st_size,_hash_file(audio)
        expected = EXPECTED_AUDIO.get(audio.name)
        if verify_source and (expected is None or (size,digest) != expected):
            raise ValueError(f"Audio size/SHA256 differs from the pinned HF release: {audio}")
        info = sf.info(str(audio))
        if info.frames <= 0 or info.samplerate <= 0 or info.channels < 1 or not math.isfinite(info.duration) or info.duration <= 0:
            raise ValueError(f"Invalid decoded audio metadata: {audio}")
        words = record.get("text")
        if not isinstance(words,str) or not words.strip():
            raise ValueError(f"Missing original Portuguese lyrics: {audio}")
        license_original = record.get("license_type","")
        license_normalized = normalize_license(license_original)
        nc = bool(license_normalized and "NC" in license_normalized.split("-"))
        genre = record.get("genre","")
        caption_genre = GENRES.get(str(genre).strip().casefold())
        caption = f"{caption_genre}, Portuguese vocals" if caption_genre else "Portuguese vocals"
        artist = str(record.get("artist","")).strip()
        if not artist:
            raise ValueError(f"Missing artist for held-out grouping: {audio}")
        decomposed = unicodedata.normalize("NFKD",artist)
        artist_key = " ".join("".join(char for char in decomposed if not unicodedata.combining(char)).casefold().split())
        holdout = "artist:"+hashlib.sha256(artist_key.encode("utf-8")).hexdigest()[:24]
        match = re.search(r"/track/(\d+)(?:/|$)",str(record.get("url","")))
        song = "jamendo:"+(match.group(1) if match else hashlib.sha256(record["name"].encode()).hexdigest()[:24])
        condition = hashlib.sha256(json.dumps([caption,words],ensure_ascii=False).encode("utf-8")).hexdigest()
        source_title = record.get("title")
        title = source_title if isinstance(source_title,str) and source_title.strip().casefold() not in ("","false","none","null") else record["name"]
        measured = {"duration_seconds":float(info.duration),"sample_rate":int(info.samplerate),
                    "channels":int(info.channels),"frames":int(info.frames),"format":info.format,
                    "subtype":info.subtype,"size_bytes":size,"sha256":digest,
                    "estimated_container_kbps":size*8/info.duration/1000}
        row = {"id":song,"song_id":song,"group_id":f"{song}:{condition[:24]}",
               "holdout_group":holdout,"artist_id":holdout,"artist":artist,"title":title,"source_title":source_title,
               "audio_path":str(audio),"caption":caption,"genre":genre,"lyrics":words,"language":"pt",
               "is_instrumental":False,"license_original":license_original,"license":license_normalized,
               "license_version":None,"noncommercial_declared":nc,"source_url":record.get("url"),
               "source_repo":REPO,"source_revision":REVISION,"audio_info":measured,"lossy_audio":True,
               "lines":record.get("lines",[]),"words":record.get("words",[]),
               "lyric_overlap":record.get("lyric_overlap"),"polyphonic":record.get("polyphonic"),
               "non_lexical":record.get("non_lexical"),"condition_sha256":condition}
        # Preserve all source fields for attribution and future manual review.
        row["source_metadata"] = record
        inspection.append({"id":song,"name":record["name"],"license":license_normalized,**measured})
        counts["verified_tracks"] += 1
        counts["noncommercial_tracks"] += int(nc)
        permitted = license_normalized in ("CC-BY","CC-BY-SA")
        if license_filter != "all" and not permitted:
            skipped.append({"id":song,"name":record["name"],"license":license_normalized,"reason":"NC/ND/unknown source declaration"})
        else:
            samples.append(row)
        if progress:
            progress({"event":"inspect","completed":index,"total":len(records),"file":audio.name})
    if not samples:
        raise ValueError("No Portuguese tracks remain after the license filter")
    durations = [s["audio_info"]["duration_seconds"] for s in samples]
    counts.update(imported_samples=len(samples),excluded_by_license=len(skipped),
                  artists=len({s["holdout_group"] for s in samples}),genres=len({s["genre"] for s in samples}),
                  missing_temporal_alignment=sum(not s["lines"] and not s["words"] for s in samples))
    provenance = {"repo":REPO,"revision":REVISION,"url":f"https://huggingface.co/datasets/{REPO}",
                  "metadata_file":str(metadata),"metadata_sha256":hashlib.sha256(raw).hexdigest(),
                  "metadata_git_blob_sha1":blob_hash,"pinned_hf_hashes_verified":bool(verify_source),
                  "license_filter":license_filter,"license_filter_basis":"Per-track source declarations only; no license version/rights clearance inferred",
                  "lyrics_processing":"Original full-song text; no correction, cropping or generated alignment",
                  "language_region":"Source labels pt; no pt-BR/pt-PT accent verification",
                  "audio_processing":"Original MP3 bytes; SHA256/size and libsndfile.info checked; no transcoding or postproduction",
                  "preferences_provided":False,"quality_scores_provided":False}
    report = {"manifest":str(output / "dataset.json"),"counts":dict(counts),"provenance":provenance,
              "license_counts":dict(Counter(r["license"] for r in samples)),
              "sample_rate_counts":dict(Counter(str(r["audio_info"]["sample_rate"]) for r in samples)),
              "channel_counts":dict(Counter(str(r["audio_info"]["channels"]) for r in samples)),
              "verified_source_license_counts":dict(Counter(r["license"] for r in inspection)),
              "verified_source_sample_rate_counts":dict(Counter(str(r["sample_rate"]) for r in inspection)),
              "verified_source_channel_counts":dict(Counter(str(r["channels"]) for r in inspection)),
              "total_imported_seconds":sum(durations),"min_duration_seconds":min(durations),
              "max_duration_seconds":max(durations),"inspection":inspection,"excluded":skipped}
    _write_json(output / "dataset.json",{"version":1,"samples":samples,"metadata":{"provenance":provenance}})
    _write_json(output / "import_report.json",report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root",required=True,help="Downloaded repo root, or its subsets/pt directory")
    parser.add_argument("--output",required=True,help="Directory for dataset.json and import_report.json")
    parser.add_argument("--license-filter",choices=("all","no-nc","commercial-compatible"),default="no-nc",
                        help="all includes NC declarations; no-nc keeps only declared CC BY/BY-SA")
    parser.add_argument("--skip-origin-check",action="store_true",help="Custom local metadata/audio: record actual hashes without asserting pinned HF identity")
    args = parser.parse_args(argv)
    report = import_jamendolyrics_pt(args.root,args.output,license_filter=args.license_filter,
                                    verify_source=not args.skip_origin_check,
                                    progress=lambda record:print(json.dumps(record,ensure_ascii=True),flush=True))
    print(json.dumps({k:v for k,v in report.items() if k not in ("inspection","excluded")},ensure_ascii=True,allow_nan=False),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
