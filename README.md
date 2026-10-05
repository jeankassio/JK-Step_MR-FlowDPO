# JK-Step MR-FlowDPO

**English** · [Português (Brasil)](README_PTBR.md) · [Español](README_ES.md)

Supervised and preference-based LoRA training for **ACE-Step 1.5 SFT and SFT XL**, with a command-line interface, a local browser UI, and a dataset toolkit derived from [Side-Step](https://github.com/koda-dernet/Side-Step). The default browser workflow starts with **a folder of music** and prepares captions, lyrics, clips, JSON and training tensors automatically.

The goal is to investigate general musical and acoustic quality while preserving lyric fidelity. You can build datasets from existing recordings; **generating music with Turbo is not required**. The adapter changes the generation model itself, without applying mastering, equalization, or other post-processing to generated songs.

![JK-Step MR-FlowDPO browser interface](assets/Screenshots/JK-Step_MR-FlowDPO.png)

**Research status:** the preference mode is an experimental ACE-Step adaptation of [MR-FlowDPO](https://arxiv.org/abs/2512.10264). The paper evaluated instrumental generation with other flow-matching models. SFT mode uses ordinary supervised flow matching, without preference comparisons. Software tests establish that the trainer executes, updates LoRA weights, and freezes the base; they do not establish improved music or guaranteed pronunciation preservation. See [validation results](docs/VALIDATION.md).

## Included features

- Folder-to-LoRA preparation with local Qwen2.5-Omni-7B captions, Whisper large-v3 lyrics/timestamps, matched clips, JSON and tensors. Choose automatic MR-FlowDPO preferences or normal SFT. No Turbo songs, Genius/API key or manually written JSON is needed in the UI.
- Automatic MR-FlowDPO pairs compare each existing clip with copied audio degraded by lowpass, noise or clipping. These labelled synthetic comparisons teach specific degradation avoidance; they do not establish general musical-quality preferences or reproduce the paper's MRSD rewards.
- Single-target SFT with masked FP32 flow-matching loss, group holdout before training repeats, and single/multi-GPU support in the same checkpoint/resume/export engine.
- A real Flow-DPO objective with the original frozen SFT as its reference, without keeping a second XL model on each GPU.
- Configurable LoRA rank, alpha, dropout, rsLoRA, per-module overrides, decoder layers, self/cross attention, and MLP targets.
- Multi-reward pair selection, human-pair import, controlled degradation pairs, shared-conditioning preprocessing, and tensor caches.
- Configuration through CLI flags, JSON, or UI; monitoring, cancellation, checkpoints, resume, and ComfyUI export.
- Dataset organization, captions, lyrics, sidecars, audio analysis, stems, PP++, and supervised adapter tools inherited from Side-Step. Some integrations need optional packages, API credentials, or additional models.
- Automatic downloads of the default pure SFT XL and preprocessing models, using this trainer's environment and checkpoint directory.

## Installation

### Windows

1. Clone or extract this repository into a writable folder.
2. Run `install_windows.bat`. It creates `.venv`, installs dependencies, and downloads the default models.
3. Run `start_ui.bat`. The UI opens at `http://127.0.0.1:8771` with a session token.

No administrator privileges or ComfyUI installation are required. The installer uses its own Python environment and does not update ComfyUI packages.

```powershell
.\jk-step.bat doctor
.\jk-step.bat --help
.\jk-step.bat train --help
```

The tested stack uses Python 3.12, PyTorch 2.10, CUDA 12.8, and TorchCodec 0.10. [constraints.txt](constraints.txt) pins the verified dependency versions.

| Installer option | Effect |
| --- | --- |
| `install_windows.bat -CoreOnly` | Install the trainer/UI and local folder caption/transcription workflow, without optional remote-API, stem-separation, and terminal-UI integrations. |
| `install_windows.bat -Rewards` | Also install optional CLAP and Audiobox scorers. |
| `install_windows.bat -SkipModels` | Skip model downloads; use local models or run `models setup` later. |
| `install_windows.bat -Backend cpu` | Install CPU PyTorch for development or small tests. XL training on CPU is slow. |

Folder preparation decodes audio with SoundFile and falls back to the bundled `imageio-ffmpeg` executable when needed; the default folder workflow needs no external FFmpeg configuration. Optional inherited toolkit functions that use TorchCodec may still require compatible shared FFmpeg libraries; see the [official TorchCodec compatibility instructions](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions).

### Linux

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
bash install_linux.sh
bash jk-step.sh doctor
bash jk-step.sh gui
```

`JK_BACKEND=cpu` selects CPU wheels; `JK_SKIP_MODELS=1` skips model downloads. Windows and Linux installers are provided. The engine accepts MPS devices, but XL training has not been validated on macOS.

### Storage and GPU memory

Reserve approximately **22 GB for the ACE generation/preprocessing models**, plus the environment, audio, cached tensors, adapters, and the additional Qwen caption/Whisper transcription downloads used by folder preparation. The SFT XL file itself is about 20 GB. Its standard Hugging Face filename is created as a hardlink where supported; other filesystems may need another approximately 20 GB copy for toolkit compatibility.

A 24 GB RTX 3090 was used for software smoke tests. Actual VRAM depends on crop length, batch, rank, target layers, and precision. Start with batch 1, mixed precision, offload, and gradient checkpointing. The default selects GPU 0; a supported device can be chosen explicitly, for example `--device cuda:1` or `--device cpu`.

## Models

```powershell
# The default installer already performs this step.
.\jk-step.bat models setup --checkpoint-dir checkpoints
```

| Component | Source |
| --- | --- |
| Pure SFT XL weights | [jeankassio/acestep_v1.5_sft_xl](https://huggingface.co/jeankassio/acestep_v1.5_sft_xl), at the pinned revision recorded by the application. |
| Matching architecture, config, and silence latent | [ACE-Step/acestep-v15-xl-sft](https://huggingface.co/ACE-Step/acestep-v15-xl-sft), pinned separately. |
| VAE and Qwen3-Embedding-0.6B | [ACE-Step/Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5), with the resolved revision recorded locally. |
| Local folder captions | [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B), downloaded when needed for audio annotation. |
| Local folder lyrics/timestamps | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3), downloaded when needed for transcription. |

Downloads are resumable and completed files are reused. Folder preparation downloads missing annotation and ACE models automatically when downloads are enabled. This workflow needs no Turbo/Merge checkpoint, music-generating language model, Genius token, or remote caption/transcription API. Reward/toolkit models are additional downloads.

For an existing safetensors file, set `checkpoint_file` and its matching `model_config_dir`. Parameter names and shapes are checked before allocating weights. `--no-auto-download` disables the default automatic download path. Standard Hugging Face model directories can instead use `checkpoint_dir` and `model_variant` (`xl-sft`, `sft`, `xl-base`, or `base`). Distilled Turbo preference training is not enabled in this engine.

## Browser UI and dataset toolkit

```powershell
.\jk-step.bat gui
# Optional: change the port and do not open a browser automatically.
.\jk-step.bat gui --port 8772 --no-browser
```

Use the UI language selector to choose **English**, **Português** or **Español**. The browser remembers this preference. It changes interface labels, independently of the lyrics/transcription language (`auto`, `pt`, `en`, `es`, etc.); English UI can prepare Portuguese music.

The main page defaults to music-folder preparation with the MR-FlowDPO objective selected. Normal SFT is another folder option. Dataset, preparation and training tabs cover music selection, automatic preparation, review, configuration, logs and cancellation. Importing human pairs and selecting scored candidates remain available separately. The toolkit link opens audio organization, sidecars, captions/lyrics, analysis, stems, PP++ and inherited adapter tools.

```powershell
.\jk-step.bat toolkit --help
.\jk-step.bat toolkit dataset --help
.\jk-step.bat toolkit captions --help
.\jk-step.bat toolkit preprocess --help
```

`sidestep` remains a compatibility alias for `toolkit`. [UPSTREAM.md](UPSTREAM.md) preserves the historical project documentation and attribution; [docs/toolkit](docs/toolkit) contains inherited guides. The current Python toolkit package is `jk_engine`; its optional terminal-UI entry point is `jk_step_tui.py`.

## Quick start: a music folder to LoRA

1. Open `start_ui.bat`. In **Dataset**, choose your music folder and the training objective: automatic **MR-FlowDPO** or normal **SFT**.
2. Start dataset preparation. The application downloads missing models, annotates the audio locally, prepares lyric-matched clips and writes the dataset/tensors outside the originals.
3. In the preparation tab, review the sample captions/lyrics and any excluded items. Automatic singing transcription and timestamps can be wrong.
4. Continue to training. The correct prepared manifest and preset are filled in automatically; select a new output folder, adjust your options and start the LoRA.

MR-FlowDPO uses the `conservative` preset: rank 32, alpha 64, LR `1e-6`, beta 100, chosen-FM regularization 0.1, batch 1, accumulation 8 and two epochs. Folder preparation keeps complete prepared clips (`max_latent_length=0`, CFG dropout 0). Its default mixed comparisons use lowpass, noise and clipping variants; those are known artificial defects, not human evaluations of composition.

Normal SFT uses `sft_lora`: rank 32, alpha 64, LR `1e-5`, batch 1, accumulation 4, 100 epochs, warmup 50, CFG dropout 0.1 and `max_latent_length=0`. These are editable starting settings, not demonstrated optima. `sft_rank64` provides rank 64/alpha 128. Use the content option for vocals, instrumentals or automatic detection.

The same workflow is available from the activated project environment:

```powershell
python -m jk_step dataset inspect --audio-dir "E:/My music"

# Automatic MR-FlowDPO; use the pairs_manifest returned in the final result.
python -m jk_step dataset prepare --audio-dir "E:/My music" --output datasets/my_music_mr --objective flow_dpo
python -m jk_step train --preset conservative --pairs-manifest "<returned pairs_manifest>" --max-latent-length 0 --output-dir output/my_mr_lora

# Alternatively, normal single-target SFT.
python -m jk_step dataset prepare --audio-dir "E:/My music" --output datasets/my_music --objective sft
python -m jk_step config --preset sft_lora --output configs/my_sft.json
python -m jk_step train --config configs/my_sft.json --dataset-manifest datasets/my_music/supervised_manifest.json --output-dir output/my_sft_lora
```

Use the actual returned paths: **`pairs_manifest` for MR-FlowDPO**, **`dataset_manifest` for SFT**. Both write `dataset.json` annotations; only SFT writes `supervised_manifest.json`, without preference scores or rejected audio. Automatic MR creates labelled controlled-degradation copies and paired tensors; no manual pair JSON is required. See the guide in [English](docs/FOLDER_TO_LORA.md), [Português](docs/FOLDER_TO_LORA_PTBR.md) or [Español](docs/FOLDER_TO_LORA_ES.md) for options, reviewed lyrics, limits, multi-GPU, resume and ComfyUI.

A bounded functional test completed source-folder preparation with local annotation models, JSON/tensors, UI-triggered SFT training and adapter export. A separate automatic MR test prepared three controlled pairs from Portuguese audio, encoded both branches, performed one genuine Flow-DPO update and exported finite adapter weights. Two RTX 3090 GPUs were exercised in another SFT execution test. These checks verify operation and saved artifacts; they do not measure improved audible quality, lyric accuracy, or performance on arbitrary datasets.

## Prepare preference datasets

Each pair contains a **chosen** and **rejected** recording representing the same intended caption and lyrics. Keep their starts aligned and durations equal. The chosen side must actually be better under the criteria you want to teach.

Supply the exact sung words, a useful musical caption, and correct metadata. Do not label a vocal song `[Instrumental]`. Review automatic captions/transcriptions. For a general-quality adapter, vary genres, languages, singers, tempos, instruments, and arrangements. A narrow dataset can still teach a narrow style.

### Human-selected pairs

Edit [configs/pair_import.example.json](configs/pair_import.example.json), then import it. Paths resolve relative to the manifest. `group_id` identifies the recording/prompt; `holdout_group` can identify an artist or larger validation group. `pair_weight` optionally controls the pair's training weight.

```powershell
.\jk-step.bat pairs build --input configs/pair_import.example.json --output datasets/human_pairs --options configs/import.example.json
```

Keep variants of the same recording or artist out of both training and validation simultaneously.

### Multi-reward selection from existing candidates

Create a manual score template, group comparable candidates, fill their scores, and apply MRSD selection:

```powershell
.\jk-step.bat pairs score --input my_audio --output datasets/scores.json --template
.\jk-step.bat pairs build --input datasets/scores.json --output datasets/mrsd --options configs/mrsd.example.json
```

The example uses text alignment, production quality, and semantic consistency, plus protected `lyric_fidelity`. Fill every required score or explicitly configure fewer axes. Missing scores are not invented. Selection requires a strong primary-axis improvement, improvements in other selected axes, and protected-axis constraints; quantiles, quality floors, outlier filters, and axis balancing are configurable.

Optional automatic backends include musical CLAP and Audiobox Production Quality. The paper's semantic reward needs a music-trained HuBERT representation and matching centroids; they are not bundled. Supply compatible resources or manual/precomputed scores. CLAP measures audio/text association, not every sung word.

```json
{"providers":["clap","audiobox"],"allow_download":true,"device":"cuda:0"}
```

Save these options in JSON and pass that file to `pairs score --options`. Two automatic scores alone do not fill all axes in the three-reward example.

### Controlled acoustic pairs

```powershell
.\jk-step.bat pairs build --input my_audio --output datasets/acoustic_pairs --options configs/degraded.example.json
```

The generator supports labelled lowpass, noise, clipping, and vocal-lowpass comparisons. Vocal-only degradation needs aligned vocal stems and `vocal_stems_dir`; filtering a complete mix does not isolate its singer.

These comparisons are an acoustic-quality experiment, not evidence of better composition or a solved SFT vocal issue. Degradations prepare training examples, without post-processing generated songs.

### Dataset sources

Your own licensed recordings can be used. The built-in catalog also links to:

| Source | Relevant content and use |
| --- | --- |
| [JamendoLyrics PT](https://huggingface.co/datasets/Felipehonorato/pt_it_jamendolyrics) | 20 Portuguese originals with human-reviewed full lyrics; per-track licenses, no timestamps. |
| [MulJam PT annotations](https://github.com/weAreMusicAI/alt-datasets-interspeech2025) | Four selected Portuguese originals with line-level timestamps; import into aligned short segments. MTG source is for noncommercial research/academic use. |
| [Muse](https://huggingface.co/datasets/bolshyC/Muse) | Existing Suno V5 songs with lyrics/style metadata; needs curation and preferences. Its card declares MIT. |
| [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html) | Real songs with aligned vocal/instrument stems; useful for vocal comparisons. Academic access must be requested; supply lyrics separately. |
| [MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset) | Full tracks with genre/instrument/mood tags. The source specifies noncommercial research/academic use and track-specific licenses. |
| [JamendoLyrics](https://huggingface.co/datasets/jamendolyrics/jamendolyrics) | Word-aligned singing benchmark; preserve an external evaluation split when assessing diction. |

```powershell
.\jk-step.bat sources
# Download only this selected archive, not the whole dataset.
.\jk-step.bat sources --download bolshyC/Muse --file en_part01_of_35.tar --output datasets/downloads/muse
```

This Muse archive is approximately 11 GB. Archives are not extracted or universally converted into captions/lyrics automatically. Follow the source layout and import the desired audio/metadata through the toolkit. See [dataset notes](docs/DATASETS.md).

For the bounded Portuguese starter collection (24 existing songs, approximately 146 MB of MP3 plus annotation metadata), use:

```powershell
.\.venv\Scripts\python.exe scripts/download_portuguese_datasets.py
.\.venv\Scripts\python.exe scripts/import_jamendolyrics_pt.py --root datasets/downloads/jamendolyrics_pt --output datasets/jamendolyrics_pt --license-filter all
.\.venv\Scripts\python.exe scripts/import_muljam_pt.py --annotations datasets/downloads/musicai_interspeech2025/dali_muljam_interspeech25.csv --metadata datasets/downloads/muljam_pt/audio_metadata.json --output datasets/muljam_pt
```

On Linux, substitute `.venv/bin/python`. These importers preserve lyrics, source hashes and licenses; the aligned importer writes FLAC or WAV FLOAT32 segments without changing gain or sample rate. Full-track lyrics must not be attached to an arbitrary short crop. These are source samples, not naturally ranked MRSD preferences. See [Portuguese dataset guide](docs/PORTUGUESE_DATASETS.md) for the pilot and its limits.

The [PT/EN quality dataset preparation guide](docs/QUALITY_DATASET_PREPARATION.md) covers experimental Portuguese forced alignment, whole Muse sections, unique sample combination and balancing PT references after preprocessing each pair once.

## Validate, preprocess, and train

Run commands from the project directory, using manifests returned by your pair-building commands:

```powershell
.\jk-step.bat pairs validate --manifest datasets/acoustic_pairs/pairs.json
.\jk-step.bat pairs preprocess --manifest datasets/acoustic_pairs/pairs.json --checkpoint-dir checkpoints --output datasets/tensors
.\jk-step.bat pairs validate --manifest datasets/tensors/pairs.preprocessed.json --check-tensors

.\jk-step.bat config --preset conservative --output configs/my_training.json
.\jk-step.bat train --config configs/my_training.json --pairs-manifest datasets/tensors/pairs.preprocessed.json --output-dir output/my_quality_lora
```

Preprocessing caches aligned VAE, text/lyrics, and ACE conditioning tensors. Both branches share conditioning and receive the same crop. The VAE posterior mean is used by default and audio normalization is disabled. Cache fingerprints track source/config changes so compatible tensors can be reused.

Default `context_mode=chosen_semantic` extracts codes from the chosen **existing recording** and shares its detokenized plan with both branches. It generates no Turbo/LM song. `context_mode=silence` uses standard text/lyrics conditioning without this plan. For independent candidates with different melodies, a winner-only plan can bias comparisons; prefer silence unless a shared semantic plan is justified. Pass preprocessing overrides as a JSON file through `--options`.

The `conservative` preset starts with rank 32, alpha 64, LR `1e-6`, beta 100, FM regularization 0.1, batch 1, accumulation 8, and two epochs. These are pilot settings, not established ACE-Step optima. `paper_beta` exposes beta 2000 from the paper's setup.

```powershell
# Training fields are available through flags, JSON, and the UI.
.\jk-step.bat train --config configs/my_training.json --rank 64 --alpha 128 --layers 0-15 --learning-rate 0.000001 --pairs-manifest datasets/tensors/pairs.preprocessed.json --output-dir output/rank64
```

CLI overrides take precedence over loaded JSON. `--set KEY=VALUE` accepts JSON-valued overrides; misspelled or unsupported keys are rejected.

| Area | Configurable controls |
| --- | --- |
| Adapter | Rank/alpha/dropout, rsLoRA, per-module rank/alpha, projections, decoder layers, self/cross attention, MLP. |
| Objective | `sft`: single-target flow matching. `flow_dpo`: beta, chosen-FM/reference-velocity regularization, label smoothing and pair weights. Continuous timesteps and CFG dropout apply to both. |
| Optimization | Batch/accumulation, epochs or max steps, AdamW/AdamW8bit/Adafactor, LR, weight decay/moments, warmup, constant/linear/cosine scheduler, clipping. |
| Hardware/data | Device, precision, offload, gradient checkpointing, workers, seeds, aligned crops, repeats, cache verification, group-level validation. |
| Outputs | Evaluation/logging frequency, TensorBoard, checkpoint frequency/retention, resume, adapter initialization, ComfyUI export. |

In Flow-DPO, batch size counts **pairs**, each containing two audio branches, and the frozen reference adds a forward pass. In SFT, it counts **single audio examples** with no reference pass. Accumulation increases effective batch size without keeping all GPU activations. Optional optimizer packages must be installed when selected; the inherited toolkit also has other optimizer/adapter choices.

### Experimental multi-GPU training

```powershell
.\jk-step.bat train --config configs/my_training.json --pairs-manifest datasets/tensors/pairs.preprocessed.json --output-dir output/two_gpus --multi-gpu --gpu-ids 0,1 --distributed-backend auto
```

`batch_size` is per GPU. Effective batch size is `batch_size × gradient_accumulation × number_of_GPUs`. Each GPU holds a decoder replica; **two 24 GB cards do not become one 48 GB memory pool**. Crop length and adapter settings must fit each card individually.

On Linux, automatic backend selection uses NCCL for GPU communication. On Windows, it uses the built-in `local` backend: workers communicate LoRA gradients through CPU memory and a multiprocessing manager, without requiring Gloo or NCCL. The optional `gloo` backend remains available for PyTorch builds that support it. Performance depends on hardware/interconnect and communication overhead. Hardware validation and its limits are recorded in [VALIDATION.md](docs/VALIDATION.md); experimental support is not a speedup guarantee. The UI exposes the corresponding hardware options.

## Checkpoints, stopping, and ComfyUI

Outputs include `training_config.json`, `metrics.jsonl`, optional TensorBoard events, checkpoints, and PEFT adapters. The final directory contains `adapter_model.safetensors`, `adapter_config.json`, and `training_state.pt`; `latest.json` records actual paths.

```powershell
# Continue the same run with its saved configuration.
.\jk-step.bat train --config output/my_quality_lora/training_config.json --resume-from output/my_quality_lora/checkpoint-00000100

# Export an existing adapter, preserving effective scaling.
.\jk-step.bat export --adapter output/my_quality_lora/final --output output/quality.safetensors --target native
```

Resume restores optimizer, scheduler, scaler, RNG, epoch, and dataset cursor. Incompatible base/model/data/settings changes are rejected. `init_adapter` starts another experiment from compatible existing weights; rank/alpha/dropout/rsLoRA must match that adapter. Use a new output directory for a new run.

Ctrl+C in the CLI or Stop in the UI requests a cooperative stop at an optimizer-update boundary and saves `stopped`. An active GPU kernel cannot be interrupted instantaneously.

With `export_comfyui=true`, successful training also writes `quality_comfyui.safetensors`. Export preserves effective per-module scaling, including rsLoRA and alpha overrides. Use `native` for the ACE-specific ComfyUI loader mapping or `generic` if required by your loader. Apply the LoRA to the SFT base family it was trained on and compare strengths; Turbo does not imply equivalent behavior.

## Method and evaluation limits

For `objective=sft`, the trainer learns the single recording's flow velocity (`noise - target_latents`) from a noisy interpolation, using masked FP32 squared error. Only decoder LoRA weights train. Prepared vocals cannot be randomly shortened while keeping their full lyrics. `beta`, preference regularization, smoothing and pair weights do not affect SFT.

For `objective=flow_dpo`, the core objective for each chosen/rejected pair is:

```text
softplus(beta * ((chosen_error - rejected_error)
               - (reference_chosen_error - reference_rejected_error)))
```

Errors are masked FP32 velocity errors. Noise, timestep, caption, lyrics, context, and pairwise dropout are shared. Reference inference disables adapters and gradients; only decoder LoRA parameters are updated. ACE's data-to-noise velocity convention is used consistently.

This implements Flow-DPO and multi-reward selection with explicit ACE adaptations. **The paper's numerical reward-prompting mechanism is not implemented.** Adding `quality=10` to captions does not reproduce it. The paper's results cannot be transferred directly to this singing model or arbitrary datasets. Read the [paper](https://arxiv.org/abs/2512.10264), [authors' code](https://github.com/lonzi/mrflow_dpo), and [training notes](docs/TRAINING.md).

Lower loss or better preference margins do not prove better audible music. Evaluate held-out songs/artists/languages/prompts with matching seeds and conditions, comparing SFT with/without LoRA. Judge clarity, artifacts, naturalness, harmony, rhythm, and exact sung words separately. Singing transcription needs human review. A frozen base or lyric-fidelity pair filter does not guarantee unchanged pronunciation after training.

## Development and distribution

```text
jk_step/          SFT/preference engine, folder pipeline, CLI, UI and models
jk_engine/        Adapted dataset/supervised toolkit from Side-Step
frontend/         Browser assets
configs/          Presets and pair examples
docs/             Method, datasets, validation and inherited toolkit guides
scripts/          Packaging and smoke tests
tests/            Automated tests
UPSTREAM.md       Historical upstream README
```

```powershell
.venv\Scripts\python.exe -m pip install pytest httpx
.venv\Scripts\python.exe -m pytest -q tests
.venv\Scripts\python.exe scripts/package_release.py
```

The release ZIP excludes environments, models, datasets, outputs, tokens, and sessions. Distribute source/installers rather than local caches or `.venv`. A wheel can be built with `uv build --wheel`; use the repository/source ZIP and provided installers for the full standalone toolkit.

## Credits and license

JK-Step MR-FlowDPO is a modified distribution of **[Side-Step](https://github.com/koda-dernet/Side-Step), by koda-dernet**, extended with ACE preference training, pairing/scoring, a new interface/CLI, model setup, and documentation. The inherited license is **[CC BY-NC-SA 4.0](LICENSE)**. Preserve [NOTICE.md](NOTICE.md), the license, and upstream attribution when redistributing; commercial use is not granted by that license.

ACE runtime/architecture and models come from [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5). Pure SFT XL is downloaded from [jeankassio/acestep_v1.5_sft_xl](https://huggingface.co/jeankassio/acestep_v1.5_sft_xl). The objective follows the equations in [MR-FlowDPO](https://arxiv.org/abs/2512.10264); this is an ACE adaptation, not the authors' Audiocraft trainer. Dependencies, weights, and datasets keep their own licenses. No upstream endorsement is implied.
