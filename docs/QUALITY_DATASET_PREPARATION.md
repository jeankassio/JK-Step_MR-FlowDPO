# Prepare PT/EN acoustic-quality datasets

[Português (Brasil)](QUALITY_DATASET_PREPARATION_PTBR.md) · [Dataset sources](DATASETS.md)

This workflow uses existing recordings: full songs → complete aligned lyric groups/sections → unique PT/EN samples → controlled degraded pairs → preprocess each unique pair once → balance PT training references to the cached tensors. It generates no Turbo/LM songs. The resulting lowpass comparisons are an **experimental acoustic preference adaptation**, not complete MRSD reward selection or evidence of improved composition, diction or vocal clarity.

Run commands from the repository root. Examples use Windows PowerShell; on Linux substitute `.venv/bin/python` and `./jk-step.sh`. Source acquisition/import is described in [Portuguese starter datasets](PORTUGUESE_DATASETS.md). Preserve the original downloads, full-song manifests, attribution and reports.

## 1. Keep full songs and create aligned Portuguese segments

`datasets/jamendolyrics_pt/dataset.json` contains original audio and human source lyrics without timestamps. Force-align that text before cropping:

```powershell
.\.venv\Scripts\python.exe scripts/align_portuguese_lyrics.py --manifest datasets/jamendolyrics_pt/dataset.json --output datasets/jamendolyrics_pt_aligned --device cuda:0 --dtype float16 --model-cache .cache/mms_fa --min-seconds 8 --max-seconds 30 --max-gap 5 --padding 0.25 --min-line-score 0.35 --max-low-character-fraction 0.25 --max-word-seconds 8
```

Use `--device cpu --dtype float32` for CPU inference. The regular alignment command downloads the pinned approximately 1.26 GB MMS_FA checkpoint if it is missing, and verifies its size/SHA256. The script preserves source words, accents and punctuation; a normalized copy is used only for CTC alignment. It does not replace lyrics with an ASR transcript. Source section labels and numeric lyrics requiring pronunciation clarification are quarantined rather than silently removed.

MMS_FA is a speech model used experimentally for singing. Its CTC scores are **uncalibrated support scores**, not probabilities of correct words, human-verified alignment or musical quality. The example thresholds reject weak/implausible spans and preserve whole line groups; they do not certify the remaining clips. Listen to boundary words and review `quarantine.json` and `alignment_report.json` before selecting positives.

Outputs include `dataset.json`, `full_tracks.json`, `alignments/`, `quarantine.json` and `alignment_report.json`. The audio fed to alignment is an inference-only mono 16 kHz copy. Training crops retain the original gain, sample rate and channels, using FLAC PCM24 or WAV FLOAT32 when decoded peaks exceed the integer PCM range. Original files remain in place.

To rebuild crops with an existing FP16 alignment cache, without inference or new model download:

```powershell
.\.venv\Scripts\python.exe scripts/align_portuguese_lyrics.py --manifest datasets/jamendolyrics_pt/dataset.json --output datasets/jamendolyrics_pt_aligned --model-cache .cache/mms_fa --cache-only --dtype float16 --min-seconds 8 --max-seconds 30 --min-line-score 0.35
```

The pinned model and matching source/text/inference cache must already exist. `--dtype` must match the cached run; missing/incompatible caches abort the rebuild. Alignment has no `--seed` argument. Seed 42 below controls selection/splits, not alignment confidence.

The separate `datasets/muljam_pt/dataset.json` already contains crops of complete automatically aligned lyric lines from `import_muljam_pt.py`; it can join the accepted Portuguese segments. Keep its annotation/source reports and the original per-track license fields.

## 2. Select whole English Muse sections

Start with the local full-song manifest created by `scripts/import_muse.py`, here `datasets/muse_english/dataset.json`:

```powershell
.\.venv\Scripts\python.exe scripts/segment_muse_dataset.py --input datasets/muse_english/dataset.json --output datasets/muse_english_sections_30s --max-samples 1000 --max-seconds 30 --min-seconds 8 --max-per-song 2 --seed 42 --validation-fraction 0.1
```

The offline script selects one source variant per `song_id`, at most two clips per song, and one or two consecutive complete sections per clip. Gaps cannot exceed five seconds. Empty lyrics and marker-only text such as `[Instrumental]` are excluded. A section over 30 seconds is reported and excluded; it is not truncated. Selected sections must fit decoded source audio and cannot cut an unselected overlapping section.

Captions retain the original style and known selected-section `desc`; lyrics retain the exact section text with labels. No word timestamps or quality scores are invented. Both source variants/any later degraded variants share song-level holdout. `crop_report.json` records source/crop hashes, frame bounds, exclusions and output encoding. Original MP3 bandwidth/artifacts remain after lossless storage; automatic Muse descriptions/timestamps and positive quality remain unverified.

## 3. Combine unique samples before pairs or oversampling

```powershell
.\.venv\Scripts\python.exe scripts/combine_quality_datasets.py --pt-input datasets/jamendolyrics_pt_aligned/dataset.json --pt-input datasets/muljam_pt/dataset.json --en-input datasets/muse_english_sections_30s/dataset.json --output datasets/quality_pt_en_unique --validation-fraction 0.15 --seed 42 --no-balance
```

The combination tool preserves captions/lyrics, records source manifests, deduplicates current audio by SHA256 and rejects conflicting conditioning. PT holdout uses normalized artist names/canonical `artist:<hash>` IDs across sources; EN uses the original Muse `song_id`. Conditioning `group_id` remains separate from holdout. Splits are stratified independently by language, so adding EN songs does not change PT artist holdout.

The local preparation snapshot contains **83 PT segments + 1,000 EN segments = 1,083 unique samples**: 53 accepted JamendoLyrics alignment crops plus 30 MulJam crops. These are observed local counts, not files bundled with the repository or guaranteed outputs for different filters/caches. Check `combine_report.json` for your actual counts and held-out artists/songs. Do not repeat sample rows at this stage.

## 4. Build controlled acoustic pairs

After reviewing the originals:

```powershell
.\jk-step.bat pairs build --input datasets/quality_pt_en_unique/dataset.json --output datasets/quality_pt_en_unique_pairs --options configs/portuguese_acoustic_pairs.example.json
.\jk-step.bat pairs validate --manifest datasets/quality_pt_en_unique_pairs/pairs.json
```

The referenced example uses seed 42 and one 6 kHz lowpass variation per original. `chosen` is the existing clip; `rejected` is its filtered **full mix**, with the same duration, caption and lyrics. It does not isolate the vocalist. A poor original remains a poor positive; the source is not certified by this comparison, and no reward values are fabricated. Vocal-only experiments require aligned vocal stems. Natural MRSD requires comparable candidates and measured scores on the selected reward axes; this workflow does not supply them.

The example pair options have their own intermediate split fraction. The final combination in step 6 assigns the authoritative artist/song split with the explicit seed/fraction; finish preparation before training from it.

## 5. Preprocess each unique pair once

```powershell
.\jk-step.bat pairs preprocess --manifest datasets/quality_pt_en_unique_pairs/pairs.json --checkpoint-dir checkpoints --model-variant xl-sft --output datasets/quality_pt_en_tensors --options configs/portuguese_preprocess.example.json
.\jk-step.bat pairs validate --manifest datasets/quality_pt_en_tensors/pairs.preprocessed.json --check-tensors
```

The example uses VAE posterior mean, `normalize=none` and `max_duration=0`, keeping the complete aligned clips. Model-required preprocessing can adapt audio to the ACE encoder's format; this is distinct from preserving the original source/crop files. `context_mode=chosen_semantic` extracts a plan from the existing chosen recording and shares it with both branches. It generates no song. Text/lyrics conditioning is also shared; the rejected branch contributes its VAE target.

Encode unique pairs before PT balancing. Compatible caches are reusable, while changed audio/conditioning/preprocessing require new fingerprints. Do not preprocess replicated training rows again.

## 6. Balance only references to PT training tensors

`--paired` requires one-language preprocessed input manifests. If step 5 produced a single mixed manifest, split **only its JSON records**, retaining the same audio paths, tensor paths and preprocessing metadata:

```powershell
@'
import json
from pathlib import Path
root = Path("datasets/quality_pt_en_tensors")
raw = json.loads((root / "pairs.preprocessed.json").read_text(encoding="utf-8-sig"))
groups = {language: [] for language in ("pt", "en")}
for pair in raw["pairs"]:
    language = pair.get("language") or pair.get("metadata", {}).get("language")
    if language not in groups:
        raise ValueError(f"Unknown language for pair {pair['id']}")
    groups[language].append(pair)
for language, pairs in groups.items():
    if not pairs:
        raise ValueError(f"No {language} preprocessed pairs")
    target = root / f"pairs.{language}.json"
    target.write_text(json.dumps({**raw, "pairs": pairs}, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
'@ | .\.venv\Scripts\python.exe -

.\.venv\Scripts\python.exe scripts/combine_quality_datasets.py --paired --pt-input datasets/quality_pt_en_tensors/pairs.pt.json --en-input datasets/quality_pt_en_tensors/pairs.en.json --output datasets/quality_pt_en_training --pt-fraction 0.6 --max-pt-repeat 20 --validation-fraction 0.15 --seed 42
.\jk-step.bat pairs validate --manifest datasets/quality_pt_en_training/pairs.combined.json --check-tensors
```

For Linux, run the Python block with a shell here-document or save it as a temporary `.py` file. If PT/EN were preprocessed separately once, supply those two manifests directly instead.

Only PT **training** rows are repeated. Replicas keep the same `tensor_path`; audio/tensor files are neither copied nor regenerated, and validation rows are not repeated. The integer repeat factor is capped at 20, so the achieved PT fraction may differ from 0.6; inspect `combine_report.json`. Use the final `pairs.combined.json` for training. With manifest balancing, keep `dataset_repeats=1`; retain `max_latent_length=0` for these whole aligned clips to avoid an additional arbitrary crop paired with unchanged full-clip lyrics.

## Licenses and redistribution

Retain each recording's license, attribution, URLs and source restrictions. Jamendo/MTG terms and per-track declarations must be considered individually; combining or cropping does not grant new rights. Muse's dataset card declares MIT for its published collection; preserve its provenance rather than treating this as clearance for every downstream use.

The alignment checkpoint has a separate [MMS CC-BY-NC 4.0 license](https://github.com/facebookresearch/fairseq/blob/main/examples/mms/README.md#license). The aligner records its publisher, pinned SHA256 and license in `model.source.json`. It is not bundled with JK-Step, and its noncommercial terms remain separate from the trainer's license and the recordings' licenses. Release packages contain scripts/docs, not downloaded recordings, alignment weights, tensor caches or trained adapters.
