"""Generate an original tone fixture and encode a real ACE preference pair."""
import argparse
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output", default=".jk_step/smoke")
    args = parser.parse_args()
    import numpy as np
    import soundfile as sf
    from jk_step.pairs import build_preference_pairs
    from jk_step.preprocess import preprocess_pairs
    root = Path(args.output).resolve()
    audio_dir = root / "audio"; audio_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(4 * 48000) / 48000
    signal = .08 * np.sin(2 * np.pi * 440 * t) + .02 * np.sin(2 * np.pi * 3300 * t)
    stereo = np.stack([signal, signal * .9], axis=-1).astype(np.float32)
    sf.write(audio_dir / "tone.wav", stereo, 48000)
    (audio_dir / "tone.json").write_text(json.dumps({"caption": "An instrumental electronic sine tone", "lyrics": "[Instrumental]", "bpm": 120, "keyscale": "A major", "language": "en"}), encoding="utf-8")
    pairs = build_preference_pairs(str(audio_dir), str(root / "pairs"),
                mode="degraded", degradation="lowpass", cutoff_hz=1500,
                validation_fraction=0, seed=1234)
    print(json.dumps({"event":"pairs", "result":pairs}), flush=True)
    result = preprocess_pairs(pairs["manifest"], args.checkpoint_dir, "xl-sft", str(root / "tensors"),
                device="cuda:0", precision="bf16", max_duration=4,
                progress_callback=lambda current, total, message: print(json.dumps(
                    {"event":"progress", "current":current, "total":total, "message":message}), flush=True))
    print(json.dumps({"event":"result", "result":result}), flush=True)
    assert result["processed"] == 1 and not result["failed"], result


if __name__ == "__main__":
    main()
