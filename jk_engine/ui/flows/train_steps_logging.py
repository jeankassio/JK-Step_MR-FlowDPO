"""
Logging, checkpoint, run-name, and chunk-duration wizard steps.

Extracted from ``train_steps.py`` to meet the module LOC policy.
"""

from __future__ import annotations

import os
from jk_engine.training_defaults import DEFAULT_SAVE_EVERY

from jk_engine.ui.flows.train_steps_helpers import smart_save_best_default as _smart_save_best_default
from jk_engine.ui.prompt_helpers import (
    _esc,
    ask,
    ask_bool,
    ask_output_path,
    ask_path,
    print_message,
    print_rich,
    section,
)


def step_logging(a: dict) -> None:
    """Logging and checkpoint settings.

    In basic mode, only save_every and save_best are prompted.
    Resume, heavy logging, and strict_resume are skipped (defaults applied).
    """
    _is_basic = a.get("config_mode") == "basic"

    section("Logging & Checkpoints (press Enter for defaults)")
    a["save_every"] = ask(
        "Save checkpoint every N epochs",
        default=a.get("save_every", DEFAULT_SAVE_EVERY),
        type_fn=int,
        allow_back=True,
    )

    # Basic mode: auto-default save_best, logging, and resume
    if _is_basic:
        a.setdefault("save_best", True)
        a["save_best_after"] = _smart_save_best_default(a)
        a.setdefault("early_stop_patience", 0)
        a.setdefault("target_loss", 0.0)
        a.setdefault("target_loss_floor", 0.01)
        a.setdefault("target_loss_warmup", 50)
        a.setdefault("target_loss_smoothing", 0.98)
        a.setdefault("log_every", 10)
        a.setdefault("log_heavy_every", 50)
        a.setdefault("resume_from", None)
        return

    a["save_best"] = ask_bool("Auto-save best model (smoothed loss)", default=a.get("save_best", True), allow_back=True)
    if a["save_best"]:
        _sba_default = a.get("save_best_after") or _smart_save_best_default(a)
        a["save_best_after"] = ask("Start best-model tracking after epoch", default=_sba_default, type_fn=int, allow_back=True)
        a["early_stop_patience"] = ask("Early stop patience (0=disabled)", default=a.get("early_stop_patience", 0), type_fn=int, allow_back=True)
        a["target_loss"] = ask("Target loss cruise control (0=disabled)", default=a.get("target_loss", 0.0), type_fn=float, allow_back=True)
        if a["target_loss"] > 0:
            _sched_warmup = a.get("warmup_steps", 0)
            _cruise_default = max(a.get("target_loss_warmup", 50), _sched_warmup) if _sched_warmup > 0 else a.get("target_loss_warmup", 50)
            _cruise_min = _sched_warmup if _sched_warmup > 0 else 0
            _cruise_hint = f" (min {_sched_warmup}, linked to scheduler warmup)" if _sched_warmup > 0 else ""
            a["target_loss_warmup"] = ask(
                f"  Cruise warmup steps{_cruise_hint}",
                default=_cruise_default, type_fn=int, allow_back=True,
                validate_fn=lambda v, _m=_cruise_min: f"Must be >= {_m} (scheduler warmup)" if v < _m else None,
            )
            a["target_loss_smoothing"] = ask("  Cruise smoothing (EMA beta)", default=a.get("target_loss_smoothing", 0.98), type_fn=float, allow_back=True)
    else:
        a["save_best_after"] = a.get("save_best_after") or _smart_save_best_default(a)
        a["early_stop_patience"] = 0
        a["target_loss"] = 0.0

    a["log_every"] = ask("Log metrics every N steps", default=a.get("log_every", 10), type_fn=int, allow_back=True)
    a["log_heavy_every"] = ask(
        "Log gradient norms every N steps (0=disabled)",
        default=a.get("log_heavy_every", 50),
        type_fn=int,
        allow_back=True,
    )
    if a["log_heavy_every"] < 0:
        a["log_heavy_every"] = 0
    resume_raw = ask("Resume from checkpoint path (leave empty to skip)", default=a.get("resume_from"), allow_back=True)
    if resume_raw in (None, "None", ""):
        a["resume_from"] = None
    else:
        # Normalize: if user pointed to a file (e.g. adapter_config.json),
        # use the containing directory instead.
        if os.path.isfile(resume_raw):
            parent = os.path.dirname(resume_raw)
            print_message(
                f"That's a file -- using checkpoint directory: {parent}",
                kind="warn",
            )
            if ask_bool("Use this directory for resume?", default=True, allow_back=True):
                resume_raw = parent
            else:
                resume_raw = ask_path(
                    "Resume checkpoint directory",
                    default=parent,
                    must_exist=True,
                    allow_back=True,
                )
        else:
            # Directory path: validate it exists
            resume_raw = ask_path(
                "Resume checkpoint directory",
                default=resume_raw,
                must_exist=True,
                allow_back=True,
            )
        a["resume_from"] = resume_raw
        a["strict_resume"] = ask_bool(
            "Strict resume? (abort on state mismatch)",
            default=a.get("strict_resume", True),
            allow_back=True,
        )


def step_chunk_duration(a: dict) -> None:
    """Latent chunking for data augmentation and VRAM savings.

    In basic mode, chunking is disabled by default without prompting.
    """
    if a.get("config_mode") == "basic":
        a.setdefault("chunk_duration", None)
        a.setdefault("max_latent_length", None)
        a.setdefault("crop_mode", None)
        return

    section("Sequence Cropping (optional)")

    print_message(
        "Sequence cropping slices preprocessed tensors into random\n"
        "  fixed-length windows each iteration, providing data augmentation\n"
        "  (the model sees different parts of each song every epoch) and\n"
        "  reducing VRAM usage for long songs.\n\n"
        "  Crop modes:\n"
        "    full    — no cropping (use entire song)\n"
        "    seconds — crop by duration in seconds (recommended: 60)\n"
        "    latent  — crop by latent frame count (advanced)",
        kind="dim",
    )

    mode = ask(
        "Crop mode (full / seconds / latent)",
        default=a.get("crop_mode", "full"),
        allow_back=True,
    ).strip().lower()
    if mode not in ("full", "seconds", "latent"):
        print_rich(f"[bold yellow]Unknown mode '{_esc(mode)}', defaulting to full.[/]")
        mode = "full"

    a["crop_mode"] = mode

    if mode == "full":
        a["chunk_duration"] = None
        a["max_latent_length"] = None
        return

    if mode == "latent":
        a["chunk_duration"] = None
        while True:
            latent_len = ask(
                "Max latent length in frames (0 = disabled)",
                default=a.get("max_latent_length", 0),
                type_fn=int, allow_back=True,
            )
            if latent_len <= 0:
                a["max_latent_length"] = None
                a["crop_mode"] = "full"
                return
            a["max_latent_length"] = latent_len
            break
    else:
        # seconds mode
        a["max_latent_length"] = None
        while True:
            chunk = ask(
                "Chunk duration in seconds (0 = disabled, recommended: 60)",
                default=a.get("chunk_duration", 0),
                type_fn=int, allow_back=True,
            )

            if chunk <= 0:
                a["chunk_duration"] = None
                a["crop_mode"] = "full"
                return

            if 0 < chunk < 10:
                print_rich(
                    f"[bold red]Rejected:[/] {chunk}s is below the hard minimum of 10 seconds.\n"
                    "  Chunks this short produce degenerate latents. Use at least 10s."
                )
                continue

            if chunk < 60:
                print_rich(
                    f"[bold yellow]Caution:[/] {chunk}s chunks are below the recommended\n"
                    "  60s minimum. This may reduce training quality, especially for\n"
                    "  full-length inference. Consider using 60s or higher."
                )
                if not ask_bool("Use this chunk size anyway?", default=False, allow_back=True):
                    continue

            a["chunk_duration"] = chunk
            break

    # Coverage decay: configurable in advanced, auto in basic
    if a.get("config_mode") != "basic":
        a["chunk_decay_every"] = ask(
            "Coverage decay interval (epochs, 0=off)",
            default=a.get("chunk_decay_every", 10),
            type_fn=int, allow_back=True,
        )
    else:
        a.setdefault("chunk_decay_every", 10)
