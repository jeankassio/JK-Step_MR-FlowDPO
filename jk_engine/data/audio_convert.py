"""
Lightweight audio format conversion for API caption providers.

Converts lossless audio (FLAC, WAV, etc.) to a temporary MP3 before
uploading to external APIs, reducing bandwidth and per-request cost.
Uses ``ffmpeg`` via subprocess; falls back gracefully to the original
file when ffmpeg is unavailable.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Formats that are already compact enough to send as-is
_PASSTHROUGH_FORMATS = frozenset({"mp3", "m4a", "aac", "ogg", "opus"})

_TARGET_BITRATE = "256k"


def ensure_mp3(audio_path: Path) -> tuple[Path, bool]:
    """Return an MP3-converted path suitable for API upload.

    If the file is already in a compact format, returns it unchanged.
    Otherwise, converts to a temporary MP3 at 256 kbps via ffmpeg.

    Args:
        audio_path: Path to the source audio file.

    Returns:
        ``(path_to_use, is_temp)`` — *is_temp* is ``True`` when the
        returned path is a temporary file that the caller must delete
        after use (see :func:`cleanup_temp`).
    """
    fmt = audio_path.suffix.lstrip(".").lower()
    if fmt in _PASSTHROUGH_FORMATS:
        return audio_path, False

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning(
            "ffmpeg not found — sending original %s file to API (install "
            "ffmpeg to auto-convert to MP3 and save bandwidth)",
            fmt.upper(),
        )
        return audio_path, False

    fd, tmp_path = tempfile.mkstemp(prefix="sidestep_api_", suffix=".mp3")
    os.close(fd)

    cmd = [
        ffmpeg, "-y",
        "-i", str(audio_path),
        "-vn",
        "-map_metadata", "-1",
        "-ac", "1",
        "-b:a", _TARGET_BITRATE,
        tmp_path,
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        src_mb = audio_path.stat().st_size / (1024 * 1024)
        dst_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
        logger.info(
            "Converted %s → MP3 for API upload: %.1f MB → %.1f MB (%s)",
            audio_path.name, src_mb, dst_mb, _TARGET_BITRATE,
        )
        return Path(tmp_path), True
    except Exception as exc:
        Path(tmp_path).unlink(missing_ok=True)
        logger.warning(
            "MP3 conversion failed for %s, sending original: %s",
            audio_path.name, exc,
        )
        return audio_path, False


def cleanup_temp(path: Path, is_temp: bool) -> None:
    """Delete a temporary file created by :func:`ensure_mp3`.

    No-op when *is_temp* is ``False``.
    """
    if is_temp:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("Could not remove temp file %s: %s", path, exc)
