# Portuguese starter datasets

This collection uses existing real recordings. It requires no Turbo or other song generation. Run the commands from the JK-Step repository after installing it. On Linux, replace `.venv\Scripts\python.exe` with `.venv/bin/python` and `jk-step.bat` with `jk-step.sh`.

## Sources

| Collection | Audio and lyrics | Relevant limitations |
| --- | --- | --- |
| [JamendoLyrics PT](https://huggingface.co/datasets/Felipehonorato/pt_it_jamendolyrics) | 20 complete Portuguese recordings, 15 artists, 10 genres, approximately 73 minutes and 106 MB; human-reviewed lyrics. | Original MP3 quality varies. No word/line timestamps or preference labels. Includes BY, BY-SA and BY-NC-SA declarations. Originally an evaluation corpus. |
| [MulJam / Music.AI](https://github.com/weAreMusicAI/alt-datasets-interspeech2025) | Four additional Portuguese recordings, approximately 17 minutes and 41 MB; official MTG 320 kbps originals, line-level lyrics and timestamps. | Alignment/transcription was refined automatically. Review it before serious training. MTG specifies noncommercial research/academic use; track BY 3.0 declarations do not remove source terms. |

The [Interspeech 2025 paper](https://arxiv.org/html/2506.02339v1) describes the annotations and Portuguese extension. The downloader pins publisher revisions and checks audio size/SHA256. It excludes an ND-licensed candidate and a mixed-language candidate requiring review.

These are small pilot collections, not enough to establish a general musical-quality improvement. Preserve artists across all splits. If training on JamendoLyrics PT, do not subsequently describe evaluation on those same songs as an independent benchmark.

## Download and import

```powershell
.\.venv\Scripts\python.exe scripts/download_portuguese_datasets.py --collection all

.\.venv\Scripts\python.exe scripts/import_jamendolyrics_pt.py --root datasets/downloads/jamendolyrics_pt --output datasets/jamendolyrics_pt --license-filter all
# Optional manifest excluding tracks declared NC/ND/unknown; this is a metadata filter, not rights clearance.
.\.venv\Scripts\python.exe scripts/import_jamendolyrics_pt.py --root datasets/downloads/jamendolyrics_pt --output datasets/jamendolyrics_pt_no_nc --license-filter no-nc

.\.venv\Scripts\python.exe scripts/import_muljam_pt.py --annotations datasets/downloads/musicai_interspeech2025/dali_muljam_interspeech25.csv --metadata datasets/downloads/muljam_pt/audio_metadata.json --output datasets/muljam_pt
```

Use `--collection jamendolyrics` or `--collection muljam` to download only one collection. Downloads are bounded: approximately 146 MB of music plus the 62 MB multilingual annotation CSV. No access key is required. The original bytes remain under `datasets/downloads/`.

The JamendoLyrics importer writes `dataset.json` and `import_report.json`, preserving full tracks and exact lyrics. The alternative `no_nc` manifest currently contains nine of the same recordings; it is not an independent dataset. Nineteen originals are stereo and one is mono. All are 44.1 kHz; none is certified as a high-quality positive merely because it is present.

The MulJam importer writes 30 short aligned segments (approximately 10.5 minutes) and a sample manifest. Its cuts respect annotated lines and preserve sample rate, channels and gain. The output uses FLAC PCM24 for 16 segments and WAV FLOAT32 for 14 whose decoded peaks exceed integer PCM range; those peaks are preserved instead of clipped or normalized. Converting MP3 to FLAC does not restore lost source quality. Reports preserve original line timestamps and identify two small overlapping annotations. Listen to segments and check the words around every cut.

Do not attach a full song's lyrics to an arbitrary 30-second crop. The unaligned JamendoLyrics recordings need verified timing before this kind of slicing. The provided short-segment pilot uses only the aligned MulJam subset.

## Acoustic preference pilot

```powershell
.\jk-step.bat pairs build --input datasets/muljam_pt/dataset.json --output datasets/muljam_pt_acoustic_pilot --options configs/portuguese_acoustic_pairs.example.json
.\jk-step.bat pairs validate --manifest datasets/muljam_pt_acoustic_pilot/pairs.json
```

This explicitly labelled experiment pairs each original segment with a 6 kHz lowpass version of the **complete mix**. Artist-level splitting keeps the same singer/recording out of both train and validation. Listen to the chosen originals and remove poor examples before training. These pairs are not naturally ranked MRSD candidates and do not isolate vocal muffling. Vocal-only comparisons require aligned vocal stems.

The current seed produces 30 pairs: 23 training and seven validation, with an entire artist held out. The two Diego OLandim recordings remain in the same split.

They test acoustic preference learning, not a guarantee of improved composition, diction or ACE-Step SFT generation. Natural MRSD preferences require comparable candidates, real reward values on the desired axes, and appropriate selection. No reward scores are fabricated for these recordings.

## Encode and train after reviewing the data

```powershell
.\jk-step.bat pairs preprocess --manifest datasets/muljam_pt_acoustic_pilot/pairs.json --checkpoint-dir checkpoints --output datasets/muljam_pt_tensors --options configs/portuguese_preprocess.example.json
.\jk-step.bat pairs validate --manifest datasets/muljam_pt_tensors/pairs.preprocessed.json --check-tensors

.\jk-step.bat train --config configs/portuguese_pilot.example.json
# For two GPUs, after confirming each card can fit the chosen sequence/rank:
.\jk-step.bat train --config configs/portuguese_pilot.example.json --multi-gpu --gpu-ids 0,1
```

The preprocessing options keep full aligned segments, use the VAE posterior mean and disable gain normalization. Both preference branches share the chosen recording's semantic codes and text/lyrics conditioning. The rank 32/alpha 64 pilot settings are editable examples, not calibrated optima. `max_latent_length=0` avoids another random crop of these already aligned short segments.

The program and release ZIP include download/import scripts and examples, **not** the recordings, trained adapters or local caches. Every recipient obtains the data from its publisher with the original attribution and terms.

## Other language collections

For a downloaded [Muse](https://huggingface.co/datasets/bolshyC/Muse) archive and metadata, the offline importer is:

```powershell
.\.venv\Scripts\python.exe scripts/import_muse.py --metadata datasets/downloads/muse/train_en.jsonl --audio-root datasets/downloads/muse --archive datasets/downloads/muse/en_part01_of_35.tar --output datasets/muse
```

This is optional English/Chinese material, not part of the Portuguese download. A single English archive is about 11 GB and its train metadata about 652 MB. The importer preserves existing lyrics/sections and groups comparisons only when conditioning matches; it does not assign preferences automatically.
