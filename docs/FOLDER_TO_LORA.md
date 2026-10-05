# From a music folder to LoRA: automatic MR-FlowDPO or SFT

**English** · [Português (Brasil)](FOLDER_TO_LORA_PTBR.md) · [Español](FOLDER_TO_LORA_ES.md) · [Project README](../README.md)

Place existing recordings in a folder, choose it in JK-Step and run preparation. The application produces descriptions, singing transcripts, short audio/text clips and dataset JSON. Choose **automatic MR-FlowDPO** to create controlled synthetic preference pairs and encode both branches, or **normal SFT** to encode one target per recording. The UI prepares the required manifests automatically: no Turbo songs, manually written pair JSON, preference scores or Genius/API key are needed.

Automatic labels are estimates. Whisper can omit, repeat or invent sung words; its timestamps can be inaccurate. Qwen can misidentify instruments or vocals. Preparation and lower training loss are not evidence of an audible quality improvement or perfect diction.

## Use the browser UI

Run `start_ui.bat` on Windows, or `bash jk-step.sh gui` on Linux. Choose **English**, **Português** or **Español** in the UI language selector; the browser saves your choice. This preference is separate from the music's lyrics/transcription language, so the UI and songs can use different languages.

1. In the dataset tab, select the music-folder workflow and its objective: MR-FlowDPO (the UI default) or normal SFT.
2. Choose the music folder, or paste its absolute path. Subfolders are scanned. Use the folder inspection control for a lightweight inventory.
3. Choose vocals, instrumentals or automatic detection in the content field. Leave the language on `auto`, or select the known language, such as `pt`, `en` or `es`.
4. Start dataset preparation. Missing models download automatically when enabled. Follow progress in the preparation tab; the models analyze local audio, then the pipeline builds clips and tensors.
5. Review the sample preview and preparation report, especially any excluded material. Listen to the corresponding clips when checking the words and boundaries.
6. Continue to training. The correct prepared manifest and objective preset are filled in automatically. Choose a new output directory, adjust the options and start training.

Preparation leaves the original recordings unchanged. It writes clips and sidecars to a separate output folder. No gain normalization or mastering is applied; ACE preprocessing converts its working copy to the sample rate/channel layout required by the model.

Local captions/transcription are included in the core installation, including `-CoreOnly`. Audio decoding uses SoundFile, with the bundled `imageio-ffmpeg` executable as a fallback; this folder workflow needs no external FFmpeg configuration. Optional legacy toolkit functions can have different decoding dependencies.

A bounded functional test completed folder input, local Qwen/Whisper annotation, JSON/tensor preparation, UI-triggered SFT training and export. A separate automatic MR test prepared three controlled pairs from Portuguese audio, encoded both branches, completed one Flow-DPO update and exported finite adapter weights. Two RTX 3090 GPUs were exercised in another SFT execution test. These are checks of software operation, not evidence of better music, word accuracy or general dataset performance.

## What preparation does

| Stage | Result |
| --- | --- |
| Scan | Recursive audio inventory, existing sidecars and source hashes. |
| Lyrics | Local Whisper large-v3 transcription with estimated phrase times. Existing reviewed lyrics are preserved when they can be matched safely. |
| Clips | Groups complete timed phrases up to the configured duration, normally 30 seconds, with only those phrases' words attached. Overlong or unusable phrases may be excluded. |
| Captions | Local Qwen2.5-Omni-7B listens to each prepared clip and writes a musical description. Supported existing captions can be reused. |
| Dataset | Audio/text JSON, provenance, group-based train/validation assignment and a report of excluded items. |
| Preferences, MR only | Compares an original clip with copied audio containing controlled defects; each pair shares its lyrics and caption. |
| Preprocessing | VAE/text/ACE tensors: a single target for SFT, or chosen/rejected branches with shared conditioning for MR-FlowDPO. |

Phrase boundaries are derived from automatic recognition, not human-verified alignment. A full song's lyrics are not attached to an arbitrary short crop. The SFT preset keeps the prepared clips intact with `max_latent_length=0`; the trainer rejects a shorter latent crop of a vocal sample rather than silently retaining unrelated words.

## Automatic MR-FlowDPO from the folder

This mode uses the existing annotated clip as **chosen** and a copied version with a controlled acoustic degradation as **rejected**. The original is not certified excellent: the label says it is preferable to that specific artificially damaged version. Starts, lengths, lyrics and caption stay aligned; source files remain unchanged. Both branches are encoded and used by the real Flow-DPO objective, with the frozen original SFT as reference.

| `pair_options` | Default / purpose |
| --- | --- |
| `degradation` | `mixed`: lowpass, noise and clipping; or select `lowpass`, `noise`, `clipping`. |
| `variations_per_audio` | 3 for mixed, 1 for a single selected defect. |
| `cutoff_hz` | 6000 Hz lowpass cutoff. |
| `noise_snr_db` | 24 dB signal-to-noise ratio. |
| `clip_threshold` | 0.15 amplitude threshold. |
| `context_mode` | `chosen_semantic`, or `silence` for standard text/lyrics conditioning. |

These are synthetic acoustic preferences, **not the paper's MRSD reward evaluations** or evidence of global musical quality. Lowpass affects the complete mix; it does not isolate vocals. Avoiding these defects is not the same task as better composition, more natural singing or perfect pronunciation. Review the comparisons and test generated music before judging the adapter.

Human-selected pairs and score-based MRSD selection remain available in the separate pair workflow; they are not replaced by this automatic experiment. SFT folder mode creates no rejected recordings or preference scores.

### Vocals and instrumentals

- `content_mode="auto"`: recognition and audio classification are used together. An empty transcript alone is not proof of an instrumental; uncertain or contradictory examples are excluded for review.
- `content_mode="vocal"`: prepare sung words and timing. A failed transcription does not receive a fabricated `[Instrumental]` label.
- `content_mode="instrumental"`: the user explicitly declares the folder instrumental. Lyrics are `[Instrumental]`; a conflicting existing vocal lyric is rejected.

Use a consistent folder when choosing vocals-only or instrumentals-only. Automatic detection is available for mixed folders. Classification can still be wrong, so inspect the resulting examples.

### Existing or corrected lyrics

Metadata is optional. Recognized sidecars include `song.caption.txt`, `song.lyrics.txt`, or a key/value `song.txt`; `song` must match the audio stem. Reviewed clips no longer than the preparation limit can use their supplied full-clip lyrics directly. Longer recordings require timing; if supplied lyrics disagree with the transcript, the material is separated for review instead of forcing that text onto mismatched audio.

For a precise correction, prepare a short source clip with the words actually sung in that clip and the corresponding sidecar, then rerun preparation. The sample preview is a review tool; it does not certify or automatically correct lyrics. Do not edit only a cached tensor's metadata and assume its encoded conditioning changed.

## Models and first use

The default pipeline uses pure SFT XL, its matching architecture, the ACE VAE and Qwen3 text embedding, plus **Qwen2.5-Omni-7B** for audio descriptions and **Whisper large-v3** for lyrics. ACE models use the configured `checkpoint_dir`; annotation models use the local Hugging Face cache. Completed downloads and compatible annotations/tensors are reused. The first preparation can spend considerable time downloading models; later runs reuse them.

Allow enough disk space beyond the approximately 22 GB ACE model set: the annotation models, environment, clips and tensor caches add more. Models run locally after downloading. Preparation releases annotation models before ACE tensor preprocessing, and uses one selected device; multi-GPU synchronization is a training option.

Official sources: [Qwen2.5-Omni-7B model card](https://huggingface.co/Qwen/Qwen2.5-Omni-7B), [Whisper large-v3 model card](https://huggingface.co/openai/whisper-large-v3), and [ACE-Step models](https://huggingface.co/ACE-Step/Ace-Step1.5). Qwen's card identifies Apache 2.0. The Whisper Hugging Face card labels that repository Apache 2.0; the [upstream Whisper software license](https://github.com/openai/whisper/blob/main/LICENSE) is MIT. Keep the actual downloaded licenses and the project's [CC BY-NC-SA 4.0](../LICENSE) attribution when redistributing.

## Command-line workflow

Use the project's activated environment (`.venv\Scripts\Activate.ps1` on Windows or `source .venv/bin/activate` on Linux). Alternatively, replace `python -m jk_step` with `jk-step.bat` or `bash jk-step.sh`.

```powershell
# Inventory only: no models loaded or downloaded.
python -m jk_step dataset inspect --audio-dir "E:/My music"

# Automatic MR: annotations, controlled comparisons and paired tensors.
python -m jk_step dataset prepare --audio-dir "E:/My music" --output datasets/my_music_mr --objective flow_dpo
python -m jk_step train --preset conservative --pairs-manifest "<returned pairs_manifest>" --max-latent-length 0 --output-dir output/my_mr_lora

# Alternatively, normal SFT: a single target per prepared clip.
python -m jk_step dataset prepare --audio-dir "E:/My music" --output datasets/my_music --objective sft
python -m jk_step config --preset sft_lora --output configs/my_sft.json
python -m jk_step train --config configs/my_sft.json --dataset-manifest datasets/my_music/supervised_manifest.json --output-dir output/my_sft_lora
```

Preparation writes a final result with `objective`, `ready`, `dataset_json`, `tensor_dir` and `report`. **MR uses its returned `pairs_manifest`**; **SFT uses `dataset_manifest`**. Substitute the actual path for the angle-bracket placeholder above; the UI fills it automatically. Specify the objective explicitly on the CLI: its historical preparation default remains SFT, whereas the UI starts with MR selected. A ready `partial` result has valid samples but some exclusions; inspect the report. Failed/cancelled preparation is not ready.

To customize preparation, save this example as `configs/my_preparation.json`:

```json
{
  "audio_dir": "E:/My music",
  "output_dir": "datasets/my_music",
  "checkpoint_dir": "checkpoints",
  "model_variant": "xl-sft",
  "objective": "flow_dpo",
  "pair_options": {
    "degradation": "mixed",
    "variations_per_audio": 3,
    "cutoff_hz": 6000,
    "noise_snr_db": 24,
    "clip_threshold": 0.15
  },
  "caption_model": "Qwen/Qwen2.5-Omni-7B",
  "caption_tier": "auto",
  "lyrics_model": "openai/whisper-large-v3",
  "language": "auto",
  "content_mode": "auto",
  "clip_seconds": 30,
  "validation_fraction": 0.1,
  "seed": 42,
  "allow_download": true,
  "reuse": true,
  "preprocess": true,
  "normalize": "none"
}
```

```powershell
python -m jk_step dataset prepare --config configs/my_preparation.json
```

The `--options` flag accepts an additional JSON object or file. For example, `{"language":"pt"}` fixes Portuguese transcription; `{"objective":"flow_dpo","pair_options":{"degradation":"noise"}}` selects automatic noise comparisons. Set `objective` to `sft` for single-target preparation. `preprocess=false` produces annotations/raw comparisons without a ready tensor dataset; include preprocessing before training. Use `--help` for CLI arguments.

## Prepared files

| Path inside the selected output | Purpose |
| --- | --- |
| `audio/` | Prepared audio clips and matching sidecars. |
| `annotation_cache/` | Reusable recognition/caption results keyed by sources and settings. |
| `dataset.json` | Audio samples, captions, clip lyrics, times and provenance for review/preprocessing. |
| `sources.json` | Original source paths/hashes and original-file preservation record. |
| `tensors/<fingerprint>/` | SFT only: Side-Step-compatible `.pt` training caches. |
| `supervised_manifest.json` | SFT only: validated single-target tensor paths/hashes and group splits. |
| `preferences/<fingerprint>/` | MR only: controlled comparison audio, raw pairs and paired tensor caches. |
| `preparation_report.json` | Counts, model provenance, exclusions and warnings. |

`dataset.json` alone is not the training cache. For SFT use `dataset_manifest=supervised_manifest.json`; existing Side-Step tensor manifests/directories are also supported. For MR use the returned preprocessed `pairs_manifest`, not raw `pairs.json`. Grouped recording/artist variants stay together across splits, and repeats apply only to training.

## Editable training settings

Automatic MR uses `conservative`: rank 32/alpha 64, LR `1e-6`, beta 100, chosen-FM regularization 0.1, batch 1, accumulation 8 and two epochs. Folder UI keeps complete clips (`max_latent_length=0`) and CFG dropout 0. `rank64` is the corresponding preference rank-64 preset. These starting settings are not a demonstrated quality optimum.

The `sft_lora` preset starts with:

| Setting | Initial value |
| --- | --- |
| Objective | `sft` |
| Rank / alpha | 32 / 64 |
| Learning rate / optimizer | `1e-5` / AdamW |
| Batch per GPU / accumulation | 1 / 4 |
| Epochs / warmup updates | 100 / 50 |
| CFG dropout | 0.1 |
| Extra latent crop | 0: retain each prepared clip |

These are starting values, not a quality guarantee or a required duration of training. Review held-out results and generated songs. Choose `sft_rank64` for rank 64/alpha 128, or edit any supported setting. Adapter modules/layers, dropout, precision, optimizer/scheduler, steps, checkpointing and hardware controls remain available. Preference-only `beta`, reference/FM regularization, label smoothing and pair weights do not affect SFT.

For two GPUs:

```powershell
python -m jk_step train --config configs/my_sft.json --dataset-manifest datasets/my_music/supervised_manifest.json --output-dir output/my_sft_2gpu --multi-gpu --gpu-ids 0,1
```

Each GPU must fit the model and its local batch; VRAM is not pooled. Effective batch size is batch × accumulation × GPU count. For this preset it is 4 on one GPU or 8 on two. Longer clips, rank 64, extra targets or a larger batch can increase memory usage. See the [main multi-GPU notes](../README.md#experimental-multi-gpu-training).

## Stop, resume and use the LoRA

The UI stop control or CLI Ctrl+C requests cancellation at an optimizer boundary and saves a resumable `stopped` checkpoint. A GPU operation or model download may need to finish before stopping.

```powershell
python -m jk_step train --config output/my_sft_lora/training_config.json --resume-from output/my_sft_lora/stopped
```

Use the actual checkpoint path in `latest.json`. Resume restores training state and rejects incompatible data, objective or settings changes. Use a new output directory for a new run. By default the run exports `quality_comfyui.safetensors` and keeps the PEFT adapter in `final/`.

Apply the exported LoRA to the same SFT base family used for training. In the ComfyUI SFT LoRA loader, route its model output to Generate; keep the original text encoder for TextEncode, because CLIP/text-encoder weights were not trained. Start comparing model strengths such as 0.5 and 1, with text-encoder strength 0, and keep seed, prompt, lyrics, CFG, sampler and LM-code settings equal. Evaluate voices, words, instruments and musical structure separately; falling loss does not prove better sound.

For human-selected or scored MR preferences, use the separate pair workflow with `objective="flow_dpo"` and `pairs_manifest`. Automatic folder comparisons are explicitly synthetic; they do not claim those human/reward labels. See [preference dataset instructions](../README.md#prepare-preference-datasets).
