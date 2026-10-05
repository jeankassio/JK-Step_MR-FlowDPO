"""Download the bounded Portuguese starter collection from its public publishers.

Run from the repository with .venv/Scripts/python.exe (Windows) or .venv/bin/python.
No songs are generated. The optional aligned subset uses full-quality originals,
not MTG's low-bitrate/mono version. It excludes ND and mixed-language candidates.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

HF_REPO = "Felipehonorato/pt_it_jamendolyrics"
HF_REVISION = "d3a53a2f145abf63351d658c4cbda6453d1c0dfc"
ANNOTATION_REVISION = "a07fb83b4eec4ed0c9ed3a5d7b79f00ff1fde828"
ANNOTATION_URL = (
    "https://raw.githubusercontent.com/weAreMusicAI/alt-datasets-interspeech2025/"
    + ANNOTATION_REVISION + "/dali_muljam_interspeech25.csv"
)
MTG_REVISION = "cafd8e20c265ed84f1e61f1c875327971f43a62f"
ANNOTATION_SHA256 = "5300db5bf79a9df0a1a191441b19b81fbbdea0100663e783721597a50158aa84"
MTG_BASE = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/" + MTG_REVISION + "/"
AUDIO_BASE = "https://cdn.freesound.org/mtg-jamendo/raw_30s/audio/"
TRACKS = [
    {"track_id": "1354721", "artist": "pacto", "artist_id": "artist_491018", "title": "Juiz",
     "path": "21/1354721.mp3", "size_bytes": 9442787,
     "sha256": "7170c6b9bb2701387e8380e03e99019001061f2f5d519da7ed45779fc79c8880",
     "tags": ["genre---hardcore", "genre---heavymetal", "genre---rock", "mood/theme---social"]},
    {"track_id": "1349643", "artist": "Diego OLandim", "artist_id": "artist_490520", "title": "Deságio",
     "path": "43/1349643.mp3", "size_bytes": 10213921,
     "sha256": "47bc1ac5a7c51b142918c2a3dacf21551c103ba407b0db5bdaaccfb91df4835c",
     "tags": ["genre---pop", "genre---popfolk"]},
    {"track_id": "1349657", "artist": "Diego OLandim", "artist_id": "artist_490520", "title": "Não é tão facil assim",
     "path": "57/1349657.mp3", "size_bytes": 11195080,
     "sha256": "d59727375768c0eb9a38e447bfbcfcad3e4cfdf3340eb918e0cf3aeedc84fb54",
     "tags": ["genre---pop", "genre---popfolk"]},
    {"track_id": "1206774", "artist": "Túlio Borges", "artist_id": "artist_461498", "title": "Adorável Trovador",
     "path": "74/1206774.mp3", "size_bytes": 9918215,
     "sha256": "d4a84bc29b30dfb3a6ca481c71f58bcc2833cc34fa0d13fd528007de5a83ac10",
     "tags": ["genre---latin", "genre---world", "mood/theme---brazil"]},
]


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _http_download(url: str, path: Path, *, size: int | None = None,
                   sha256: str | None = None) -> Path:
    import requests
    if path.is_file() and (size is None or path.stat().st_size == size):
        if sha256 and file_hash(path) == sha256:
            return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".downloading")
    try:
        with requests.get(url, stream=True, timeout=(20, 90)) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
        if size is not None and temporary.stat().st_size != size:
            raise ValueError(f"Unexpected file size: {path.name}")
        if sha256 and file_hash(temporary) != sha256:
            raise ValueError(f"SHA256 mismatch: {path.name}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def download_jamendolyrics(root: Path, workers: int = 4) -> dict:
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().dataset_info(HF_REPO, revision=HF_REVISION, files_metadata=True)
    files = [f for f in info.siblings if f.rfilename.startswith("subsets/pt/mp3/")
             and f.rfilename.endswith(".mp3")]
    if len(files) != 20:
        raise ValueError("Unexpected pinned Portuguese collection: expected 20 tracks")
    files += [f for f in info.siblings if f.rfilename in {
        "README.md", "JamendoLyrics_pt_it.csv", "subsets/pt/metadata.jsonl"}]
    target = root / "jamendolyrics_pt"

    def fetch(f):
        path = Path(hf_hub_download(HF_REPO, f.rfilename, repo_type="dataset",
                                   revision=HF_REVISION, local_dir=str(target)))
        if f.size is not None and path.stat().st_size != f.size:
            raise ValueError(f"Size mismatch: {f.rfilename}")
        digest = file_hash(path)
        expected = f.lfs.sha256 if f.lfs is not None else None
        if expected and digest != expected:
            raise ValueError(f"SHA256 mismatch: {f.rfilename}")
        return {"path": f.rfilename, "size_bytes": path.stat().st_size, "sha256": digest}

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fetch, f) for f in files]):
            row = future.result(); records.append(row)
            print(f"JamendoLyrics PT: {len(records)}/{len(files)} verified", flush=True)
    result = {"repo": HF_REPO, "revision": HF_REVISION,
              "files": sorted(records, key=lambda r: r["path"])}
    (target / "download_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def download_muljam(root: Path, workers: int = 4) -> dict:
    annotations = root / "musicai_interspeech2025"
    csv_path = _http_download(ANNOTATION_URL, annotations / "dali_muljam_interspeech25.csv",
                              size=61734413, sha256=ANNOTATION_SHA256)
    annotation_source = {"url": ANNOTATION_URL, "commit": ANNOTATION_REVISION,
                         "size_bytes": csv_path.stat().st_size, "sha256": file_hash(csv_path)}
    (annotations / "source.json").write_text(json.dumps(annotation_source, indent=2), encoding="utf-8")
    target = root / "muljam_pt"

    def fetch(track):
        relative = "audio/" + track["path"]
        url = AUDIO_BASE + track["path"]
        _http_download(url, target / relative, size=track["size_bytes"], sha256=track["sha256"])
        return {**track, "audio_path": relative, "song_id": "muljam_" + track["track_id"],
                "source_url": url, "track_url": "https://www.jamendo.com/track/" + track["track_id"],
                "license": "CC BY 3.0", "license_url": "http://creativecommons.org/licenses/by/3.0/",
                "language": "pt"}

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fetch, t) for t in TRACKS]):
            row = future.result(); records.append(row)
            print(f"MulJam PT: {len(records)}/{len(TRACKS)} verified", flush=True)
    metadata = {"metadata": {
        "source": "https://github.com/MTG/mtg-jamendo-dataset",
        "mtg_revision": MTG_REVISION, "annotations_source": annotation_source,
        "original_metadata_urls": [MTG_BASE + p for p in (
            "data/raw.meta.tsv", "data/raw.tsv", "audio_licenses.txt",
            "data/download/raw_30s_audio_sha256_tracks.txt")],
        "scope": "MTG source specifies noncommercial research and academic use; track licenses do not override source terms.",
        "lyrics_alignment": "Music.AI automatic refined line-level annotations; manual review still needed.",
        "excluded": {"muljam_7095": "ND recording license", "muljam_1060118": "Mixed-language lyrics require review"},
    }, "tracks": sorted(records, key=lambda r: r["track_id"])}
    (target / "audio_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets/downloads"))
    parser.add_argument("--collection", choices=("all", "jamendolyrics", "muljam"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    root = args.output.expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    if args.collection in ("all", "jamendolyrics"):
        download_jamendolyrics(root, args.workers)
    if args.collection in ("all", "muljam"):
        download_muljam(root, args.workers)
    print(json.dumps({"downloads": str(root), "collection": args.collection}), flush=True)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
