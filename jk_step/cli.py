"""CLI shared by the browser UI and standalone users; ML imports are lazy."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any

from . import APP_NAME, __version__

ROOT = Path(__file__).resolve().parent.parent


def emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, default=str), flush=True)


def get_presets() -> dict:
    # The paper's beta is 2000 on its own latent/loss scale. Lower values here
    # are explicit pilot settings, not claimed optimal ACE-Step hyperparameters.
    return {
        "sft_lora": {"objective": "sft", "rank": 32, "alpha": 64,
                     "learning_rate": 1e-5, "optimizer": "adamw",
                     "batch_size": 1, "gradient_accumulation": 4,
                     "epochs": 100, "max_latent_length": 0,
                     "cfg_dropout": 0.1, "warmup_steps": 50},
        "sft_rank64": {"objective": "sft", "rank": 64, "alpha": 128,
                       "learning_rate": 1e-5, "batch_size": 1,
                       "gradient_accumulation": 4, "epochs": 100,
                       "max_latent_length": 0, "cfg_dropout": 0.1,
                       "warmup_steps": 50},
        "conservative": {"objective": "flow_dpo", "rank": 32, "alpha": 64, "learning_rate": 1e-6,
                         "beta": 100, "fm_regularization": 0.1,
                         "target_mlp": False, "batch_size": 1,
                         "gradient_accumulation": 8, "epochs": 2},
        "rank64": {"objective": "flow_dpo", "rank": 64, "alpha": 128, "learning_rate": 1e-6,
                   "beta": 100, "fm_regularization": 0.1,
                   "batch_size": 1, "gradient_accumulation": 8, "epochs": 2},
        "paper_beta": {"objective": "flow_dpo", "rank": 32, "alpha": 64, "learning_rate": 1e-6,
                       "beta": 2000, "fm_regularization": 0.0,
                       "batch_size": 1, "gradient_accumulation": 8, "epochs": 10},
    }


def _load_json(value: str) -> dict:
    if value.startswith("{"):
        data = json.loads(value)
    else:
        data = json.loads(Path(value).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    return data


def _typed(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _stop_event(stop_file: str | None) -> threading.Event:
    event = threading.Event()
    def stop(*_):
        event.set()
        emit({"event": "stop_requested"})
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    if stop_file:
        target = Path(stop_file)
        def watch():
            while not event.wait(0.3):
                if target.exists():
                    event.set()
                    break
        threading.Thread(target=watch, daemon=True).start()
    return event


def doctor() -> dict:
    packages = {}
    for name in ("torch", "torchaudio", "transformers", "peft", "accelerate",
                 "safetensors", "fastapi", "uvicorn", "soundfile"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"name": APP_NAME, "version": __version__, "python": sys.version,
              "executable": sys.executable, "root": str(ROOT), "packages": packages}
    if packages.get("torch"):
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["distributed"] = {
            "available": torch.distributed.is_available(),
            "gloo": torch.distributed.is_gloo_available(),
            "nccl": torch.distributed.is_nccl_available(),
            "local": True,
        }
        if result["distributed"]["gloo"]:
            try:
                torch.distributed.ProcessGroupGloo.create_device(hostname="127.0.0.1")
                result["distributed"]["gloo_transport_usable"] = True
            except (RuntimeError, AttributeError) as exc:
                result["distributed"]["gloo_transport_usable"] = False
                result["distributed"]["gloo_transport_error"] = str(exc)
        if torch.cuda.is_available():
            result["gpus"] = []
            for index in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(index)
                result["gpus"].append({"name": torch.cuda.get_device_name(index), "index": index,
                                       "total_gib": round(total / 2**30, 2),
                                       "free_gib": round(free / 2**30, 2)})
            result["gpu"] = result["gpus"][0]
    result["ready"] = all(packages.values())
    return result


def launch_gui(port: int, open_browser: bool = True) -> None:
    import uvicorn
    from jk_engine.gui.security import generate_token
    from jk_engine.gui.server import create_app
    token = generate_token()
    app = create_app(token=token, port=port)
    url = f"http://127.0.0.1:{port}/quality?token={token}"
    print(f"{APP_NAME}\n{url}\nCtrl+C to stop.", flush=True)
    if open_browser:
        import webbrowser
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


def _parser() -> argparse.ArgumentParser:
    from .config import schema_fields
    parser = argparse.ArgumentParser(prog="jk-step", description=APP_NAME)
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check runtime and CUDA without loading ACE-Step")
    gui = sub.add_parser("gui", help="Launch browser UI including the Side-Step tools")
    gui.add_argument("--port", type=int, default=8771)
    gui.add_argument("--no-browser", action="store_true")
    cfg = sub.add_parser("config", help="Write a full editable training configuration")
    cfg.add_argument("--preset", choices=list(get_presets()), default="conservative")
    cfg.add_argument("--output", default="configs/my_training.json")
    train = sub.add_parser("train", help="Train an ACE-Step supervised or preference LoRA")
    train.add_argument("--config", help="JSON configuration")
    train.add_argument("--preset", choices=list(get_presets()))
    train.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    train.add_argument("--stop-file", help="Cooperatively stop and save when this file appears")
    for field in schema_fields():
        name = field["name"]
        option = "--" + name.replace("_", "-")
        kw = {"default": argparse.SUPPRESS, "help": field.get("help", "")}
        kind = field["type"]
        if kind == "bool":
            kw["action"] = argparse.BooleanOptionalAction
        elif kind == "int":
            kw["type"] = int
        elif kind == "float":
            kw["type"] = float
        elif kind == "json":
            kw["type"] = json.loads
        elif kind == "list":
            kw["type"] = lambda text: [x.strip() for x in text.split(",") if x.strip()]
        if field.get("choices"):
            kw["choices"] = field["choices"]
        train.add_argument(option, dest=name, **kw)
    dataset = sub.add_parser("dataset", help="Prepare a music folder: captions, lyrics, JSON and SFT tensors")
    ds = dataset.add_subparsers(dest="dataset_command", required=True)
    inspect = ds.add_parser("inspect", help="Scan an audio folder without loading models")
    inspect.add_argument("--audio-dir", required=True)
    prepare = ds.add_parser("prepare", help="Automatically prepare a music folder for supervised or preference LoRA")
    prepare.add_argument("--audio-dir")
    prepare.add_argument("--objective", choices=['sft','flow_dpo'])
    prepare.add_argument("--output", dest="output_dir")
    prepare.add_argument("--config", help="Optional complete preparation JSON configuration")
    prepare.add_argument("--options", default="{}", help="Optional JSON object or file with preparation options")
    prepare.add_argument("--stop-file")
    pairs = sub.add_parser("pairs", help="Create, validate, score and encode preference datasets")
    pair_sub = pairs.add_subparsers(dest="pair_command", required=True)
    build = pair_sub.add_parser("build", help="Import pairs, select MRSD pairs, or degrade existing audio")
    build.add_argument("--input", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--options", default="{}", help="JSON object or JSON file")
    build.add_argument("--stop-file")
    validate = pair_sub.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--check-tensors", action="store_true")
    validate.add_argument("--no-check-audio", action="store_true")
    prep = pair_sub.add_parser("preprocess")
    prep.add_argument("--manifest", required=True)
    prep.add_argument("--checkpoint-dir", required=True)
    prep.add_argument("--model-variant", default="xl-sft")
    prep.add_argument("--output", required=True)
    prep.add_argument("--options", default="{}")
    prep.add_argument("--stop-file")
    score = pair_sub.add_parser("score")
    score.add_argument("--input", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--options", default="{}")
    score.add_argument("--stop-file")
    score.add_argument("--template", action="store_true", help="Create a manual scoring manifest instead of running models")
    models = sub.add_parser("models", help="Inspect a checkpoint or fetch small official architecture files")
    ms = models.add_subparsers(dest="model_command", required=True)
    inspect = ms.add_parser("inspect")
    inspect.add_argument("--file", required=True)
    metadata = ms.add_parser("metadata")
    metadata.add_argument("--repo", default="ACE-Step/acestep-v15-xl-sft")
    metadata.add_argument("--revision", default="main")
    metadata.add_argument("--output", default="checkpoints/acestep-v15-xl-sft")
    setup = ms.add_parser("setup", help="Download SFT XL and its preprocessing models, reusing cached files")
    setup.add_argument("--checkpoint-dir", default="checkpoints")
    setup.add_argument("--weights-only", action="store_true")
    source = sub.add_parser("sources", help="Documented dataset sources; no automatic bulk downloads")
    source.add_argument("--download", help="HF dataset repository ID")
    source.add_argument("--file", action="append", default=[])
    source.add_argument("--output", default="datasets/downloads")
    source.add_argument("--revision", default="main")
    export = sub.add_parser("export", help="Export a saved LoRA to ComfyUI with its effective scaling")
    export.add_argument("--adapter", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--target", choices=["native", "generic"], default="native")
    side = sub.add_parser("toolkit", aliases=["sidestep"], help="Dataset, captions, SFT, PP++ and adapter tools inherited from Side-Step")
    side.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"toolkit", "sidestep"}:
        # Forward even --help to the inherited command parser, so users see
        # its complete command reference rather than the wrapper's placeholder.
        import subprocess
        return subprocess.call([sys.executable, str(ROOT / "train.py"), *arguments[1:]], cwd=ROOT)
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor(); emit(result)
            return 0 if result["ready"] else 1
        if args.command == "gui":
            launch_gui(args.port, not args.no_browser); return 0
        if args.command == "config":
            from .config import schema_defaults
            config = schema_defaults(); config.update(get_presets()[args.preset])
            output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            emit({"event": "result", "config": str(output.resolve())}); return 0
        if args.command == "train":
            from .config import schema_defaults
            from .distributed import launch_training
            config = schema_defaults()
            if args.preset:
                config.update(get_presets()[args.preset])
            if args.config:
                config.update(_load_json(args.config))
            schema_names = set(schema_defaults())
            config.update({k: v for k, v in vars(args).items() if k in schema_names})
            for override in args.set:
                key, sep, value = override.partition("=")
                if not sep or key not in schema_names:
                    raise ValueError(f"Invalid override: {override}")
                config[key] = _typed(value)
            event = _stop_event(args.stop_file)
            result = launch_training(config, progress=emit, stop_event=event)
            emit({"event": "result", "result": result}); return 0
        if args.command == "dataset":
            from .dataset_pipeline import inspect_folder, prepare_folder
            if args.dataset_command == "inspect":
                emit(inspect_folder(args.audio_dir)); return 0
            config = _load_json(args.config) if args.config else {}
            config.update(_load_json(args.options))
            for name in ("audio_dir", "output_dir", "objective"):
                if getattr(args, name, None):
                    config[name] = getattr(args, name)
            if not str(config.get("audio_dir", "")).strip():
                raise ValueError("Specify --audio-dir or audio_dir in the configuration")
            result = prepare_folder(config, progress=emit, stop_event=_stop_event(args.stop_file))
            emit({"event": "result", "result": result})
            if result.get("cancelled"):
                return 0
            return 0 if result.get("ready") or (not config.get("preprocess", True)
                                              and result.get("dataset_json")) else 2
        if args.command == "pairs":
            from .pairs import build_preference_pairs, validate_pairs
            if args.pair_command == "validate":
                result = validate_pairs(args.manifest, check_audio=not args.no_check_audio,
                                        check_tensors=args.check_tensors)
                emit(result); return 0 if result["valid"] else 2
            options = _load_json(args.options)
            options["progress_callback"] = lambda current, total, message: emit(
                {"event": "progress", "current": current, "total": total, "message": message})
            options["stop_event"] = _stop_event(args.stop_file)
            if args.pair_command == "build":
                result = build_preference_pairs(args.input, args.output, **options)
            elif args.pair_command == "score":
                from .rewards import score_candidates, create_score_template
                if args.template:
                    result = create_score_template(args.input, args.output, axes=options.get("axes"))
                else:
                    result = score_candidates(args.input, args.output, **options)
            else:
                from .preprocess import preprocess_pairs
                result = preprocess_pairs(args.manifest, args.checkpoint_dir, args.model_variant,
                                          args.output, **options)
            emit({"event": "result", "result": result}); return 0
        if args.command == "models":
            from .models import inspect_checkpoint, prepare_metadata, ensure_models
            if args.model_command == "inspect":
                emit(inspect_checkpoint(args.file))
            elif args.model_command == "setup":
                emit({"event": "result", "result": ensure_models(
                    args.checkpoint_dir, include_preprocess=not args.weights_only, progress=emit)})
            else:
                emit(prepare_metadata(args.repo, args.output, args.revision))
            return 0
        if args.command == "sources":
            from .sources import SOURCES, download_files
            emit(download_files(args.download, args.file, args.output, args.revision)
                 if args.download else SOURCES)
            return 0
        if args.command == "export":
            from .training import export_adapter
            result = export_adapter(args.adapter, args.output, target=args.target)
            emit({"event": "result", "result": result}); return 0
        if args.command in {"toolkit", "sidestep"}:
            import subprocess
            return subprocess.call([sys.executable, str(ROOT / "train.py"), *args.args], cwd=ROOT)
    except (Exception, KeyboardInterrupt) as exc:
        logging.exception("Command failed")
        emit({"event": "error", "error": str(exc), "type": type(exc).__name__})
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    return 0
