"""Combine PT/EN quality datasets with group holdout and optional cached-pair balancing.

Samples are always unique: build pairs and preprocess them once first. The
--paired stage then repeats references to existing PT training tensors, never
audio files or validation examples. Source captions/lyrics are preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import tempfile
import unicodedata


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile("w",dir=path.parent,encoding="utf-8",suffix=".writing",delete=False) as handle:
        stage = Path(handle.name)
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
    try:
        stage.replace(path)
    finally:
        stage.unlink(missing_ok=True)


def _absolute(value, base: Path) -> str:
    if not isinstance(value,str) or not value:
        raise ValueError("Audio/tensor paths must be nonempty strings")
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base/path).resolve())


def _artist_id(name: str) -> str:
    normalized = " ".join("".join(c for c in unicodedata.normalize("NFKD",name)
                                  if not unicodedata.combining(c)).casefold().split())
    if not normalized:
        raise ValueError("Artist name cannot be blank")
    return "artist:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _value(row: dict, name: str, default=None):
    return row.get(name) or row.get("metadata",{}).get(name) or default


def _holdout(row: dict, language: str) -> str:
    if language=="pt":
        artist = _value(row,"artist")
        if artist:
            if not isinstance(artist,str):
                raise ValueError("artist must be a string")
            return _artist_id(artist)
        for field in ("artist_id","holdout_group"):
            value = _value(row,field)
            if value and re.fullmatch(r"artist:[0-9a-f]{24}",str(value)):
                return str(value)
        raise ValueError(f"PT record {row.get('id')} needs artist name or canonical artist:<hash> holdout")
    song = _value(row,"song_id")
    if not song:
        song = _value(row,"holdout_group")
        if song and str(song).startswith("artist:"):
            raise ValueError("EN Muse holdout requires song_id, not an invented shared artist")
    if not isinstance(song,str) or not song.strip():
        raise ValueError(f"EN record {row.get('id')} needs its original song_id")
    return song if song.startswith("song:") else "song:"+song


def _known_audio_sha(row: dict, branch: str | None) -> str | None:
    metadata = row.get("metadata",{})
    # source_audio_sha256 may describe the original full song, not its clip.
    # Never use that field to identify the current training audio.
    if branch=="rejected":
        candidates = (row.get("rejected_sha256"),metadata.get("rejected_sha256"))
    else:
        candidates = (row.get("chosen_sha256") if branch else None,row.get("sha256"),
                      row.get("audio_sha256"),row.get("audio_info",{}).get("sha256"),
                      metadata.get("sha256"),metadata.get("audio_sha256"),
                      metadata.get("audio_info",{}).get("sha256"))
    for candidate in candidates:
        if candidate:
            if not isinstance(candidate,str) or not re.fullmatch("[0-9a-fA-F]{64}",candidate):
                raise ValueError(f"Malformed current-audio SHA256 for record {row.get('id')}")
            return candidate.lower()
    return None


def _normalize(row, language, source: Path, paired: bool, hash_cache: dict, stats: Counter) -> dict:
    if not isinstance(row,dict):
        raise ValueError(f"Manifest {source} contains a non-object record")
    row = copy.deepcopy(row)
    if not isinstance(row.get("metadata",{}),dict):
        raise ValueError("Record metadata must be an object")
    metadata = row.setdefault("metadata",{})
    identity = row.get("id")
    if not isinstance(identity,str) or not identity:
        raise ValueError("Every record must have a nonempty string id")
    if row.get("replica_index",0) or metadata.get("balancing",{}).get("replica_index",0):
        raise ValueError(f"Input {identity} is already a replica; combine unique source manifests")
    if not isinstance(row.get("caption"),str) or not isinstance(row.get("lyrics"),str):
        raise ValueError(f"Record {identity} needs original caption and lyrics strings")
    origin_language = _value(row,"language")
    if origin_language and str(origin_language).lower().replace("_","-").split("-")[0] != language:
        raise ValueError(f"Record {identity} language {origin_language!r} conflicts with --{language}-input")
    origin_split = row.get("split",metadata.get("split"))
    paths = ("chosen","rejected") if paired else ("audio_path",)
    for name in paths:
        if name=="audio_path" and name not in row:
            row[name] = row.get("path") or row.get("filename")
        row[name] = _absolute(row.get(name),source.parent)
        audio = Path(row[name])
        if not audio.is_file():
            raise FileNotFoundError(f"Missing audio referenced by {identity}: {audio}")
        known = _known_audio_sha(row,name if paired else None)
        key = str(audio).casefold()
        cached = hash_cache.get(key)
        if cached and known and cached != known:
            raise ValueError(f"Conflicting SHA256 declarations for the same audio: {audio}")
        if not cached:
            cached = known or _digest(audio)
            stats["cached_sha_declarations" if known else "new_audio_hashes"] += 1
            if not known:
                stats["bytes_hashed"] += audio.stat().st_size
            hash_cache[key] = cached
        row[name+"_sha256" if paired else "audio_sha256"] = cached
    for name in ("audio_path","vocal_path","chosen_vocal_path","rejected_vocal_path",
                 "semantic_features","tensor_path"):
        if row.get(name):
            row[name] = _absolute(row[name],source.parent)
    for name in ("audio_path","vocal_path","semantic_features","source_audio_path"):
        if metadata.get(name):
            metadata[name] = _absolute(metadata[name],source.parent)
    if paired and not row.get("tensor_path"):
        raise ValueError(f"Pair {identity} has no tensor_path; --paired balances only preprocessed pairs")
    if paired and not Path(row["tensor_path"]).is_file():
        raise FileNotFoundError(f"Missing preprocessed tensor: {row['tensor_path']}")
    holdout = _holdout(row,language)
    row.update(language=language,holdout_group=holdout)
    metadata.update(language=language,holdout_group=holdout)
    if language=="pt":
        row["artist_id"] = holdout
        metadata["artist_id"] = holdout
    if not row.get("group_id"):
        # Conditioning groups describe comparable prompts; they are separate
        # from artist/song holdout groups and remain unchanged if supplied.
        condition = hashlib.sha256(json.dumps([row["caption"],row["lyrics"]],ensure_ascii=False,
                                              separators=(",",":")).encode()).hexdigest()
        row["group_id"] = identity+":"+condition[:24]
    metadata["combination"] = {"source_manifest":str(source),"source_id":identity,
                               "source_language":origin_language,"source_split":origin_split}
    return row


def _split(rows: list[dict], fraction: float, seed: int) -> dict:
    report = {}
    for language in ("pt","en"):
        groups = sorted({r["holdout_group"] for r in rows if r["language"]==language})
        # Independent stratification makes the PT artist split stable if the
        # number of EN songs grows in a later dataset version.
        rng_seed = int.from_bytes(hashlib.sha256(f"{seed}:{language}".encode()).digest()[:8],"big")
        random.Random(rng_seed).shuffle(groups)
        count = round(len(groups)*fraction)
        if fraction and len(groups)>1:
            count = max(1,min(count,len(groups)-1))
        else:
            count = 0
        validation = set(groups[:count])
        for row in rows:
            if row["language"]==language:
                row["split"] = "validation" if row["holdout_group"] in validation else "train"
                row["metadata"]["split"] = row["split"]
        names = {r["holdout_group"]:_value(r,"artist") for r in rows if r["language"]==language}
        songs = {split:{str(_value(r,"song_id",r["id"])) for r in rows
                        if r["language"]==language and r["split"]==split}
                 for split in ("train","validation")}
        report[language] = {"group_kind":"artist" if language=="pt" else "song",
                            "groups":len(groups),"train_groups":len(groups)-count,
                            "validation_groups":count,
                            "validation_holdout_groups":sorted(validation),
                            "validation_artist_names":sorted({str(names[g]) for g in validation if names.get(g)}),
                            "train_songs":len(songs["train"]),"validation_songs":len(songs["validation"]),
                            "single_group_no_validation":bool(fraction and len(groups)==1)}
    return report


def combine_quality_datasets(pt_inputs, en_inputs, output, *, paired=False, pt_fraction=.6,
                             validation_fraction=.15, seed=42, balance=None, max_pt_repeat=20) -> dict:
    if not 0 < pt_fraction < 1:
        raise ValueError("pt_fraction must be between 0 and 1")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0,1)")
    if not isinstance(max_pt_repeat,int) or max_pt_repeat<1:
        raise ValueError("max_pt_repeat must be a positive integer")
    balance = paired if balance is None else bool(balance)
    if balance and not paired:
        raise ValueError("Balance only after unique pairs are preprocessed, using --paired")
    output = Path(output).expanduser().resolve()
    manifest = output if output.suffix.lower()==".json" else output/("pairs.combined.json" if paired else "dataset.json")
    rows,sources,ids,hash_cache,duplicates = [],[],set(),{},[]
    stats = Counter()
    by_audio = {}
    preprocessing = []
    for language,inputs in (("pt",pt_inputs),("en",en_inputs)):
        for input_path in inputs:
            path = Path(input_path).expanduser().resolve()
            if path==manifest:
                raise ValueError("Output manifest cannot overwrite an input")
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(raw,list):
                raw = {"pairs" if paired else "samples":raw}
            field = "pairs" if paired else "samples"
            if not isinstance(raw,dict) or not isinstance(raw.get(field),list):
                raise ValueError(f"{path} must contain a {field} list")
            metadata = raw.get("metadata",{})
            if not isinstance(metadata,dict):
                raise ValueError(f"{path} metadata must be an object")
            sources.append({"manifest":str(path),"manifest_sha256":_digest(path),
                            "language":language,"input_rows":len(raw[field]),"metadata":metadata})
            if paired and metadata.get("preprocessing"):
                preprocessing.append(metadata["preprocessing"])
            for original in raw[field]:
                row = _normalize(original,language,path,paired,hash_cache,stats)
                identity = row["id"]
                if identity in ids:
                    raise ValueError(f"Duplicate record id across inputs: {identity}")
                ids.add(identity)
                stats["input_rows"] += 1
                fingerprint = ((row["chosen_sha256"],row["rejected_sha256"]) if paired
                               else (row["audio_sha256"],))
                existing = by_audio.get(fingerprint)
                if existing:
                    signature = lambda r:(r["language"],r["holdout_group"],r["caption"],r["lyrics"],
                                          r.get("pair_weight",1),r.get("primary_axis"))
                    if signature(existing)!=signature(row):
                        raise ValueError(f"Duplicate audio has conflicting conditioning/group/language: {existing['id']} / {identity}")
                    duplicates.append({"kept_id":existing["id"],"removed_id":identity,
                                       "source_manifest":str(path),"reason":"identical_audio_sha256_and_conditioning"})
                    continue
                by_audio[fingerprint] = row
                rows.append(row)
    if not rows:
        raise ValueError("No samples/pairs were supplied")
    if preprocessing:
        for key in ("model_variant","context_mode","shared_conditioning"):
            values = {json.dumps(p[key],sort_keys=True) for p in preprocessing if key in p}
            if len(values)>1:
                raise ValueError(f"Preprocessed pair sources disagree on {key}")
    split_report = _split(rows,validation_fraction,seed)
    counts = Counter((r["language"],r["split"]) for r in rows)
    train_pt,train_en = counts[("pt","train")],counts[("en","train")]
    desired = train_en*pt_fraction/(train_pt*(1-pt_fraction)) if train_pt else 0
    requested = max(1,math.ceil(desired-1e-12))
    repeat = min(max_pt_repeat,requested) if balance and train_pt and train_en else 1
    result_rows = []
    for row in rows:
        factor = repeat if row["language"]=="pt" and row["split"]=="train" else 1
        for index in range(factor):
            replica = copy.deepcopy(row)
            replica.update(base_id=row["id"],replica_index=index,replica_count=factor)
            if index:
                replica["id"] = f"{row['id']}::pt-repeat-{index:02d}"
                if replica["id"] in ids:
                    raise ValueError(f"Replica id collides with an input: {replica['id']}")
                ids.add(replica["id"])
            replica["metadata"]["balancing"] = {"base_id":row["id"],"replica_index":index,
                "repeat_factor":factor,"train_only":True,"audio_copied":False,"tensor_copied":False}
            result_rows.append(replica)
    emitted = Counter((r["language"],r["split"]) for r in result_rows)
    total_train = emitted[("pt","train")]+emitted[("en","train")]
    achieved = emitted[("pt","train")]/total_train if total_train else None
    report = {"manifest":str(manifest),"mode":"preprocessed_pairs" if paired else "unique_samples",
        "input_rows":stats["input_rows"],"unique_rows":len(rows),"emitted_rows":len(result_rows),
        "duplicates_removed":duplicates,"hashing":dict(stats),"group_split":split_report,
        "unique_counts":{language:{split:counts[(language,split)] for split in ("train","validation")}
                         for language in ("pt","en")},
        "emitted_counts":{language:{split:emitted[(language,split)] for split in ("train","validation")}
                          for language in ("pt","en")},
        "balancing":{"enabled":balance,"target_pt_train_fraction":pt_fraction,
            "required_repeat_factor":desired,"requested_integer_repeat":requested,"pt_train_repeat_factor":repeat,
            "max_pt_repeat":max_pt_repeat,"capped":bool(balance and requested>max_pt_repeat),
            "replicas_added":len(result_rows)-len(rows),"achieved_pt_train_fraction":achieved,
            "validation_repeated":False,"audio_files_copied":False,"tensor_files_copied":False},
        "seed":seed,"validation_fraction":validation_fraction,"sources":sources}
    common = {"combination":report,"scientific_scope":"Source audio/lyrics retained; preferences are not inferred by this combination tool."}
    if preprocessing:
        common["preprocessing"] = {"model_variant":preprocessing[0].get("model_variant"),
            "context_mode":preprocessing[0].get("context_mode"),
            "shared_conditioning":preprocessing[0].get("shared_conditioning"),
            "source_preprocessing":preprocessing}
    _write(manifest,{"version":1,"pairs" if paired else "samples":result_rows,"metadata":common})
    report_path = manifest.parent/"combine_report.json"
    _write(report_path,report)
    return {"manifest":str(manifest),"report":str(report_path),"unique_rows":len(rows),
            "emitted_rows":len(result_rows),"train_rows":total_train,
            "validation_rows":sum(emitted[(language,"validation")] for language in ("pt","en")),
            "pt_train_repeat_factor":repeat,"achieved_pt_train_fraction":achieved}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pt-input",action="append",default=[],help="PT sample/pair manifest; repeat for multiple sources")
    parser.add_argument("--en-input",action="append",default=[],help="EN sample/pair manifest; repeat for multiple sources")
    parser.add_argument("--output",required=True,help="Output directory or explicit JSON manifest")
    parser.add_argument("--paired",action="store_true",help="Use preprocessed tensor pairs; balance PT training references by default")
    parser.add_argument("--pt-fraction",type=float,default=.6)
    parser.add_argument("--validation-fraction",type=float,default=.15)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--max-pt-repeat",type=int,default=20)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--balance",dest="balance",action="store_true",default=None)
    group.add_argument("--no-balance",dest="balance",action="store_false")
    args = parser.parse_args(argv)
    result = combine_quality_datasets(args.pt_input,args.en_input,args.output,paired=args.paired,
        pt_fraction=args.pt_fraction,validation_fraction=args.validation_fraction,seed=args.seed,
        balance=args.balance,max_pt_repeat=args.max_pt_repeat)
    print(json.dumps(result,ensure_ascii=True,allow_nan=False),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
