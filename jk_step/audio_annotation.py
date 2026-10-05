"""Local audio annotation. Model outputs are automatic, never human verification.

Whisper is speech-trained; transcription/timestamps on singing can be wrong.
Qwen captions likewise require listening evaluation. No API credentials are used.
"""
from __future__ import annotations
import ast
import gc
import json
import math
import re
from pathlib import Path

ANNOTATION_VERSION = 2
MODEL_REVISIONS = {"Qwen/Qwen2.5-Omni-7B": "ae9e1690543ffd5c0221dc27f79834d0294cba00",
                   "openai/whisper-large-v3": "06f233fe06e710322aca913c1bc4249a0d71fce1"}
CAPTION_SYSTEM_PROMPT = "You analyze audible music conservatively. Follow the user's schema and return only one valid JSON object with double-quoted keys and strings. Do not add markdown, prose outside JSON, or guesses about inaudible details."
CAPTION_PROMPT = """Listen to the entire attached audio clip. Return one JSON object only:
{"caption":"brief factual English description of the audible music", "genre":"or unknown",
"vocal_status":"singing|speech|no_vocals|uncertain", "bpm":null,"key":null,"signature":null}.
Describe only audible instruments, groove, timbre, vocal delivery, and texture.
Do not invent instruments, lyrics, song sections, artist identities or production equipment.
Do not describe an opening, drop, bridge, climax or outro outside this clip.
Use uncertain when you cannot decide whether there is a voice. No guessed exact BPM/key/meter.
Caption must not contain an artist's name, song title or invented lyrics.
"""


class PreparationCancelled(RuntimeError):
    pass


def check_stop(event):
    if event is not None and event.is_set():
        raise PreparationCancelled("Preparation cancelled")


def resolve_device(value="auto"):
    import torch
    if value == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in ("cuda", "cpu", "mps", "xpu"):
        raise ValueError("Unsupported annotation device")
    return str(device)


def read_audio(path):
    """Decode original channels/amplitude. Optional ffmpeg only decodes formats SF cannot read."""
    import numpy as np
    import soundfile as sf
    try:
        audio, sr = sf.read(path, dtype="float64", always_2d=True)
    except (RuntimeError, OSError) as original_error:
        import shutil
        import subprocess
        import tempfile
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                raise ValueError(f"Cannot decode {Path(path).name}; install imageio-ffmpeg or ffmpeg for this format") from original_error
        with tempfile.TemporaryDirectory(prefix="jk_decode_") as tmp:
            target = Path(tmp) / "decoded.wav"
            subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(path),
                            "-vn", "-c:a", "pcm_f32le", str(target)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           creationflags=subprocess.CREATE_NO_WINDOW if __import__('os').name == 'nt' else 0)
            audio, sr = sf.read(target, dtype="float64", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all() or sr < 1:
        raise ValueError("Audio is empty or contains nonfinite samples")
    if audio.shape[1] > 2:
        raise ValueError("More than two channels: supply a deliberate mono/stereo mix; no channels are silently dropped")
    return audio, int(sr)


def validate_phrases(chunks, duration):
    """Reject missing, overlapping or out-of-range automatic timestamps."""
    result, end = [], 0.0
    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        times = chunk.get("timestamp", (chunk.get("start"), chunk.get("end")))
        if len(times) != 2 or any(t is None or not math.isfinite(float(t)) for t in times):
            raise ValueError("ASR returned incomplete phrase timestamps; refusing guessed boundaries")
        first, last = map(float, times)
        if not 0 <= first < last <= duration + .025 or first < end - .025:
            raise ValueError("ASR returned overlapping or out-of-range phrase timestamps")
        first, last = max(end, first), min(duration, last)
        if first >= last:
            raise ValueError("ASR phrase contains no audio interval")
        result.append({"start": first, "end": last, "text": text})
        end = last
    return result


def whisper_segments(result, tokenizer, timestamp_begin, duration):
    """Decode native Whisper segments, accepting only predicted time boundaries.

    Transformers can use a window boundary when no end timestamp was predicted.
    Such a fallback is unsuitable for training lyrics and is rejected here.
    """
    if not isinstance(result, dict) or "segments" not in result:
        raise ValueError("Native Whisper did not return timestamped segments")
    batches = result["segments"]
    if len(batches) != 1:
        raise ValueError("Native Whisper returned an unexpected audio batch")
    phrases = []
    for segment in batches[0]:
        tokens = segment["tokens"]
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        text = tokenizer.decode([int(t) for t in tokens if int(t) < timestamp_begin], skip_special_tokens=True).strip()
        if not text:
            continue
        if not tokens or int(tokens[0]) < timestamp_begin or int(tokens[-1]) < timestamp_begin:
            raise ValueError("ASR returned incomplete phrase timestamps; refusing a window-end fallback")
        phrases.append({"text": text, "start": float(segment["start"]), "end": float(segment["end"])})
    return validate_phrases(phrases, duration)


def parse_caption(raw):
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Caption model did not return the requested JSON metadata")
    payload = text[start:end + 1]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        # Some local generations use Python-style quotes. This accepts only
        # literal data, never evaluates code, and applies the same schema.
        try:
            data = ast.literal_eval(payload)
        except (SyntaxError, ValueError) as error:
            raise ValueError("Caption model returned malformed metadata") from error
    if not isinstance(data, dict) or not isinstance(data.get("caption"), str) or not isinstance(data.get("vocal_status"), str):
        raise ValueError("Caption metadata must contain string caption and vocal_status fields")
    if not isinstance(data.get("genre", ""), str):
        raise ValueError("Caption genre must be a string")
    caption = data["caption"].strip()
    status = data["vocal_status"].strip().lower()
    if len(caption) < 10 or caption.lower() in ("unknown", "n/a", "none"):
        raise ValueError("Caption model returned no usable description")
    if status not in ("singing", "speech", "no_vocals", "uncertain"):
        raise ValueError("Caption model returned an unsupported vocal classification")
    # Exact musical fields are not accepted from this generative classifier.
    return {"caption": caption, "genre": "" if str(data.get("genre", "")).lower() in ("unknown", "n/a", "none") else str(data.get("genre", "")),
            "vocal_status": status, "bpm": None, "keyscale": "", "timesignature": ""}


class LocalAudioAnnotator:
    """Lazy, sequential model use; unload_asr must precede the caption phase."""
    def __init__(self, config, progress=None, stop_event=None):
        self.config, self.progress, self.stop_event = config, progress, stop_event
        self.device = None
        self.asr = None
        self.origin = {"lyrics_model": config["lyrics_model"], "caption_model": config["caption_model"],
                       "lyrics_model_card": "https://huggingface.co/" + config["lyrics_model"],
                       "caption_model_card": "https://huggingface.co/" + config["caption_model"],
                       "human_verified": False, "confidence_is_calibrated": False}

    def _event(self, message):
        check_stop(self.stop_event)
        if self.progress:
            self.progress({"event": "model", "stage": "annotation", "message": message})

    def transcribe(self, audio, sample_rate):
        import numpy as np
        import torch
        from scipy.signal import resample_poly
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, StoppingCriteria
        check_stop(self.stop_event)
        if self.asr is None:
            self.device = self.device or resolve_device(self.config["device"])
            dtype = torch.float32 if self.device == "cpu" else torch.bfloat16 if self.config["precision"] == "bf16" else torch.float16
            self._event("Loading local Whisper; uncached weights download when allowed")
            kwargs = {"local_files_only": not self.config["allow_download"],
                      "revision": MODEL_REVISIONS.get(self.config["lyrics_model"])}
            model = AutoModelForSpeechSeq2Seq.from_pretrained(self.config["lyrics_model"],
                torch_dtype=dtype, low_cpu_mem_usage=True, **kwargs).to(self.device).eval()
            processor = AutoProcessor.from_pretrained(self.config["lyrics_model"], **kwargs)
            self.origin["lyrics_revision"] = getattr(model.config, "_commit_hash", None)
            # Direct array feature extraction avoids the pipeline's TorchCodec
            # decoder import and its shared-FFmpeg DLL requirement on Windows.
            self.asr = (model, processor, dtype)
        mono = np.asarray(audio, dtype=np.float32).mean(axis=1)
        divisor = math.gcd(sample_rate, 16000)
        if sample_rate != 16000:
            mono = resample_poly(mono, 16000 // divisor, sample_rate // divisor).astype(np.float32)
        event = self.stop_event
        class Stop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return event is not None and event.is_set()
        generation = {"task": "transcribe", "do_sample": False, "num_beams": 1,
                      "condition_on_prev_tokens": False, "stopping_criteria": [Stop()],
                      "return_timestamps": True, "return_segments": True}
        if self.config["language"] != "auto":
            generation["language"] = self.config["language"]
        model, processor, dtype = self.asr
        inputs = processor(mono, sampling_rate=16000, return_tensors="pt", truncation=False,
                           padding="longest", return_attention_mask=True)
        check_stop(self.stop_event)
        def monitor_progress(progress):
            check_stop(self.stop_event)
            if self.progress:
                current, total = progress[0].tolist()
                self.progress({"event": "progress", "stage": "transcription",
                    "message": "Whisper is transcribing automatic phrase times",
                    "current": int(current), "total": int(total)})
        with torch.inference_mode():
            result = model.generate(input_features=inputs.input_features.to(self.device, dtype=dtype),
                attention_mask=inputs.attention_mask.to(self.device), monitor_progress=monitor_progress,
                **generation)
        check_stop(self.stop_event)
        timestamp_begin = model.generation_config.no_timestamps_token_id + 1
        self.origin["lyrics_inference"] = "native Whisper generate; predicted segment boundaries only"
        return whisper_segments(result, processor.tokenizer, timestamp_begin, len(audio) / sample_rate)

    def unload_asr(self):
        self.asr = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def caption(self, path):
        from jk_engine.data.caption_provider_local import generate_caption
        check_stop(self.stop_event)
        self.device = self.device or resolve_device(self.config["device"])
        tier = self.config["caption_tier"]
        if tier == "auto":
            if self.device.startswith("cuda"):
                import torch
                free, _total = torch.cuda.mem_get_info(self.device)
                bf16_minimum = (10 if "3B" in self.config["caption_model"] else 18) * 1024 ** 3
                if free >= bf16_minimum:
                    tier = "16gb"
                elif free >= 8 * 1024 ** 3:
                    tier = "8-10gb"
                else:
                    raise ValueError("Insufficient free GPU memory for local captioning; stop other GPU work or choose another device")
            else:
                tier = "16gb"
        else:
            tier = "8-10gb" if tier in ("8gb", "8-10gb") else "16gb"
        self._event("Analyzing attached audio with local Qwen Omni")
        raw = generate_caption("", "", audio_path=Path(path), tier=tier, max_new_tokens=400,
            temperature=0, repetition_penalty=1.05, stop_event=self.stop_event,
            model_id=self.config["caption_model"], device=self.device,
            revision=MODEL_REVISIONS.get(self.config["caption_model"]), text_only=True,
            allow_download=self.config["allow_download"], full_audio=True,
            prompt_instructions=CAPTION_PROMPT, system_prompt=CAPTION_SYSTEM_PROMPT)
        from jk_engine.data import caption_provider_local
        model = caption_provider_local._model
        if model is not None:
            self.origin["caption_revision"] = getattr(model.config, "_commit_hash", None)
        check_stop(self.stop_event)
        return parse_caption(raw)

    def close(self):
        self.unload_asr()
        from jk_engine.data.caption_provider_local import unload_model
        unload_model()
