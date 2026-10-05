"""Optional real reward models and manual score templates.

MR-FlowDPO's semantic reward uses a music-trained HuBERT and learned centroids.
Its private checkpoint cannot be recreated by labelling spectral brightness or
MusicFM similarity as 'musicality'. Supply that axis manually/precomputed, or
provide a compatible music-HuBERT and centroid set for the paper-style formula.
"""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path

from .pairs import _read_samples, write_manifest, is_cancelled

MUSIC_CLAP_CHECKPOINT = "music_audioset_epoch_15_esc_90.14.pt"
REWARD_SOURCES = {
    "clap": "https://github.com/LAION-AI/CLAP",
    "audiobox": "https://github.com/facebookresearch/audiobox-aesthetics",
    "semantic_formula": "https://arxiv.org/html/2512.10264v2#S3.SS3.SSS3",
}


def semantic_consistency_reward(features, centroids, temperature: float = .1, chunk_size: int = 2048) -> float:
    """Equation (4): average maximum log softmax over cosine centroids.

    Features and centroids MUST come from the same trained representation/layer.
    This function does not train HuBERT or pretend arbitrary embeddings reproduce
    the paper's private music representation. Higher values are preferred.
    """
    import torch
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("Semantic temperature must be finite and positive.")
    if chunk_size < 1:
        raise ValueError("Semantic chunk_size must be positive.")
    features = torch.as_tensor(features, dtype=torch.float32)
    centroids = torch.as_tensor(centroids, dtype=torch.float32, device=features.device)
    if features.ndim != 2 or centroids.ndim != 2 or features.shape[-1] != centroids.shape[-1]:
        raise ValueError("Semantic features [T,D] and centroids [K,D] must share D.")
    if features.shape[0] == 0 or centroids.shape[0] < 2:
        raise ValueError("Semantic reward needs nonempty features and at least two centroids.")
    if not torch.isfinite(features).all() or not torch.isfinite(centroids).all():
        raise ValueError("Semantic inputs contain NaN or infinity.")
    features = torch.nn.functional.normalize(features, dim=-1)
    centroids = torch.nn.functional.normalize(centroids, dim=-1)
    total = 0.0
    for start in range(0, len(features), chunk_size):
        logits = features[start:start+chunk_size] @ centroids.T / temperature
        values = logits.max(dim=-1).values - torch.logsumexp(logits, dim=-1)
        total += float(values.sum())
    return total / len(features)


def create_score_template(input_manifest_or_audio_dir: str, output_file: str,
                          axes=("text_alignment", "production_quality", "semantic_consistency", "lyric_fidelity")) -> dict:
    """Create a candidate manifest for auditable human/precomputed scores."""
    samples, metadata = _read_samples(input_manifest_or_audio_dir)
    axes = axes or ("text_alignment", "production_quality", "semantic_consistency", "lyric_fidelity")
    if isinstance(axes,str):
        axes = [a.strip() for a in axes.split(",") if a.strip()]
    for sample in samples:
        sample["scores"] = {axis: sample.get("scores", {}).get(axis) for axis in axes}
        sample.setdefault("group_id", sample.get("song_id", ""))
    manifest = write_manifest(Path(output_file).expanduser().resolve(),
        {"version":1, "metadata":{**metadata, "score_template":True,
          "instructions":"Fill scores and group_id. Candidates in one group must have identical caption/lyrics. Higher scores are better; missing scores are never invented."},
         "samples":samples})
    return {"manifest":manifest,"samples":len(samples),"axes":list(axes)}


def _cleanup(model):
    import torch
    if model is not None and hasattr(model, "to"):
        model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_candidates(input_manifest_or_audio_dir: str, output_file: str, **options) -> dict:
    """Score candidates with optional CLAP/Audiobox/music-HuBERT providers.

    providers: list ('clap', 'audiobox', 'hubert'). All model downloads are
    explicit via allow_download. Scores may then be edited before MRSD pairing.
    CLAP never certifies lyric correctness; add an independently curated axis.
    """
    import numpy as np
    import torch
    import soundfile as sf
    samples, metadata = _read_samples(input_manifest_or_audio_dir)
    if not samples:
        raise ValueError("No candidate audio was found.")
    providers = options.get("providers", ["clap", "audiobox"])
    if isinstance(providers, str):
        providers = [p.strip() for p in providers.split(",") if p.strip()]
    if not providers or any(p not in ("clap", "audiobox", "hubert") for p in providers):
        raise ValueError("Choose clap, audiobox and/or hubert providers.")
    device = options.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    allow_download = bool(options.get("allow_download", False))
    callback = options.get("progress_callback")
    cancel = lambda: is_cancelled(options)
    warnings, provider_meta = [], {}
    batch_size = int(options.get("batch_size", 1))
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    for sample in samples:
        sample["scores"] = dict(sample.get("scores", {}))
        if not Path(sample["audio_path"]).is_file():
            raise FileNotFoundError(sample["audio_path"])
    if "clap" in providers and not cancel():
        try:
            import laion_clap
        except ImportError as exc:
            raise RuntimeError("Optional CLAP scorer is missing. Install laion-clap or import manual/precomputed scores.") from exc
        checkpoint = options.get("clap_checkpoint")
        if not checkpoint:
            if not allow_download:
                raise ValueError("Provide clap_checkpoint or enable allow_download to download the verified music CLAP checkpoint.")
            from huggingface_hub import hf_hub_download
            checkpoint = hf_hub_download("lukewys/laion_clap", MUSIC_CLAP_CHECKPOINT)
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        model = None
        try:
            # Music checkpoint requires HTSAT-base, unlike the package default.
            model = laion_clap.CLAP_Module(enable_fusion=False, device=device,
                                         amodel=options.get("clap_amodel", "HTSAT-base"))
            model.load_ckpt(str(checkpoint), verbose=False)
            model.eval()
            for start in range(0,len(samples),batch_size):
                if cancel():
                    break
                rows = samples[start:start+batch_size]
                with torch.inference_mode():
                    audio = model.get_audio_embedding_from_filelist([r["audio_path"] for r in rows],use_tensor=True)
                    text = model.get_text_embedding([r.get("caption","") for r in rows],use_tensor=True)
                    values = torch.nn.functional.cosine_similarity(audio.float(),text.float(),dim=-1).cpu().tolist()
                for row,value in zip(rows,values):
                    row["scores"]["text_alignment"] = float(value)
                if callback:
                    callback(min(start+batch_size,len(samples)),len(samples),"CLAP text alignment")
            provider_meta["clap"] = {"checkpoint":str(checkpoint),"axis":"text_alignment",
                                     "source":REWARD_SOURCES["clap"],"does_not_verify_lyrics":True}
        finally:
            _cleanup(model)
            del model
    if "audiobox" in providers and not cancel():
        try:
            from audiobox_aesthetics.infer import initialize_predictor
        except ImportError as exc:
            raise RuntimeError("Optional quality scorer is missing. Install audiobox_aesthetics or import manual/precomputed scores.") from exc
        checkpoint = options.get("audiobox_checkpoint")
        if not checkpoint and not allow_download:
            raise ValueError("Provide audiobox_checkpoint or enable allow_download for facebook/audiobox-aesthetics.")
        predictor = None
        try:
            predictor = initialize_predictor(ckpt=checkpoint)
            predictor.device = torch.device(device)
            predictor.model.to(device)
            for start in range(0,len(samples),batch_size):
                if cancel():
                    break
                rows = samples[start:start+batch_size]
                # soundfile tensors avoid torchcodec/torchaudio-load incompatibility.
                batch = []
                for row in rows:
                    wav,sr = sf.read(row["audio_path"],dtype="float32",always_2d=True)
                    batch.append({"path":torch.from_numpy(np.ascontiguousarray(wav.T)),"sample_rate":sr})
                scores = predictor.forward(batch)
                for row,values in zip(rows,scores):
                    row["scores"]["production_quality"] = float(values["PQ"])
                    row["audiobox_axes"] = {key:float(value) for key,value in values.items()}
                if callback:
                    callback(min(start+batch_size,len(samples)),len(samples),"Audiobox production quality")
            provider_meta["audiobox"] = {"checkpoint":checkpoint or "facebook/audiobox-aesthetics",
                                        "axis":"production_quality","source":REWARD_SOURCES["audiobox"]}
        finally:
            _cleanup(predictor.model if predictor else None)
            del predictor
    if "hubert" in providers and not cancel():
        centroid_path, model_dir = options.get("semantic_centroids"), options.get("hubert_model_dir")
        if not centroid_path:
            raise ValueError("Semantic reward requires semantic_centroids learned from the SAME music-HuBERT/layer. The paper's checkpoint is not public.")
        if not model_dir and not all(s.get("semantic_features") for s in samples):
            raise ValueError("Provide hubert_model_dir trained on music or semantic_features for every candidate.")
        path = Path(centroid_path)
        centroids = np.load(path) if path.suffix.lower()==".npy" else torch.load(path,map_location="cpu",weights_only=True)
        if isinstance(centroids,dict):
            centroids = centroids.get("centroids")
        model = None
        try:
            if model_dir:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(model_dir,local_files_only=not allow_download).to(device).eval()
            for index,sample in enumerate(samples):
                if cancel():
                    break
                if sample.get("semantic_features"):
                    features = np.load(sample["semantic_features"])
                else:
                    from scipy.signal import resample_poly
                    wav,sr = sf.read(sample["audio_path"],dtype="float32",always_2d=True)
                    rate = int(options.get("hubert_sample_rate",16000))
                    mono = wav.mean(axis=1)
                    if sr != rate:
                        divisor = math.gcd(sr,rate)
                        mono = resample_poly(mono,rate//divisor,sr//divisor)
                    mono = (mono-mono.mean()) / max(float(mono.std()),1e-7)
                    with torch.inference_mode():
                        result = model(torch.from_numpy(np.ascontiguousarray(mono)).unsqueeze(0).to(device),output_hidden_states=True)
                        features = result.hidden_states[int(options.get("hubert_layer",12))].squeeze(0)
                sample["scores"]["semantic_consistency"] = semantic_consistency_reward(features,centroids,
                    temperature=float(options.get("semantic_temperature",.1)))
                if callback:
                    callback(index+1,len(samples),"Music-HuBERT centroid consistency")
            provider_meta["hubert"] = {"model":model_dir,"centroids":str(path),
                "layer":int(options.get("hubert_layer",12)),"axis":"semantic_consistency",
                "checkpoint_equivalence_to_paper":"not established; user supplied representation",
                "source":REWARD_SOURCES["semantic_formula"]}
        finally:
            _cleanup(model)
            del model
    available = sorted(set.intersection(*(set(s["scores"]) for s in samples)))
    missing = sorted(set(("text_alignment","production_quality","semantic_consistency"))-set(available))
    if missing:
        warnings.append("Partial reward adaptation; missing paper axes: "+", ".join(missing)+". Supply manual/precomputed values or explicitly choose fewer axes.")
    warnings.append("CLAP/audio-quality/semantic rewards do not certify sung-word accuracy. Curate diction separately or add lyric_fidelity scores.")
    if any(not math.isfinite(float(v)) for s in samples for v in s["scores"].values() if v is not None):
        raise ValueError("A reward provider produced a nonfinite score.")
    result = {"version":1,"metadata":{**metadata,"reward_providers":provider_meta,
              "paper_axes_complete":not missing,"warnings":warnings},"samples":samples}
    output = write_manifest(Path(output_file).expanduser().resolve(),result)
    return {"manifest":output,"samples":len(samples),"axes":available,"warnings":warnings,
            "providers":provider_meta,"cancelled":bool(cancel())}
