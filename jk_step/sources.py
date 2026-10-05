"""Dataset catalog and deliberately selected HF downloads (not an unbounded crawler)."""
from pathlib import Path

SOURCES = [
    {"name": "JamendoLyrics Portuguese", "url": "https://huggingface.co/datasets/Felipehonorato/pt_it_jamendolyrics",
     "kind": "20 real Portuguese songs, complete human-reviewed lyrics, 15 artists and 10 genres",
     "notes": "About 106 MB. Download subsets/pt/mp3, not root mp3 reference files. Per-song BY/BY-SA/BY-NC-SA licenses; no timestamps. Designed as an evaluation corpus: training on it prevents independent benchmark evaluation.",
     "repo": "Felipehonorato/pt_it_jamendolyrics", "revision": "d3a53a2f145abf63351d658c4cbda6453d1c0dfc",
     "example_files": ["subsets/pt/metadata.jsonl"]},
    {"name": "MulJam Portuguese starter", "url": "https://github.com/weAreMusicAI/alt-datasets-interspeech2025",
     "kind": "4 additional Portuguese originals with automatically refined line-level lyrics/timestamps",
     "notes": "40.8 MB of 320 kbps MP3 originals from the official MTG mirror, plus 61.7 MB multilingual annotation CSV. Bounded downloader and aligned FLAC importer in scripts/. MTG source specifies noncommercial research/academic use."},
    {"name": "Muse", "url": "https://huggingface.co/datasets/bolshyC/Muse",
     "kind": "Existing generated songs with lyrics and descriptions",
     "notes": "Suno V5 dataset, not Turbo. MIT declared by its dataset card. A single English archive is about 11 GB; select files explicitly.",
     "repo": "bolshyC/Muse", "example_files": ["en_part01_of_35.tar"]},
    {"name": "MUSDB18-HQ", "url": "https://sigsep.github.io/datasets/musdb.html",
     "kind": "Real songs with vocal and instrument stems",
     "notes": "150 songs; suitable for vocal-only degradation pairs. Academic access request required; no supplied exact lyrics."},
    {"name": "MTG-Jamendo", "url": "https://github.com/MTG/mtg-jamendo-dataset",
     "kind": "Real songs with genre, instrument and mood tags",
     "notes": "55k songs. Use stereo full-quality files for clarity work. Dataset requires noncommercial research use; individual licenses vary."},
    {"name": "JamendoLyrics", "url": "https://huggingface.co/datasets/jamendolyrics/jamendolyrics",
     "kind": "Word-aligned singing benchmark",
     "notes": "79 songs in EN/FR/DE/ES, test split. Keep as holdout when assessing diction; per-song licenses vary.",
     "repo": "jamendolyrics/jamendolyrics"},
]


def download_files(repo: str, files: list[str], output: str, revision: str = "main") -> dict:
    if not files:
        raise ValueError("Supply --file for each desired archive/metadata file; whole-dataset downloads are not implicit")
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().dataset_info(repo, revision=revision, files_metadata=True)
    available = {f.rfilename: f for f in info.siblings}
    for name in files:
        if name not in available:
            raise ValueError(f"File not present in {repo}: {name}")
    target = Path(output).resolve(); target.mkdir(parents=True, exist_ok=True)
    paths = [hf_hub_download(repo, name, repo_type="dataset", revision=info.sha,
                            local_dir=str(target)) for name in files]
    return {"repo": repo, "revision": info.sha, "files": paths,
            "size_bytes": sum(available[name].size or 0 for name in files)}
