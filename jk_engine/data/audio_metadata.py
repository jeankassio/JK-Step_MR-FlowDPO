"""
Extract song metadata (title, artist) from embedded audio tags.

Uses ``mutagen`` to read ID3 (MP3), Vorbis comments (FLAC/OGG),
MP4 atoms (M4A/AAC), and other tag formats.  Falls back to filename
parsing when tags are absent or unreadable.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import NamedTuple, Optional

logger = logging.getLogger(__name__)

# Common filename patterns: "Artist - Title", "Title"
_FILENAME_RE = re.compile(r"^(.+?)\s*[-–—]\s*(.+)$")


class SongMeta(NamedTuple):
    """Metadata resolved from an audio file."""

    title: str
    artist: str
    source: str  # "tags" or "filename"


def parse_filename(audio_path: Path) -> tuple[str, str]:
    """Extract artist and title from an audio filename.

    Tries ``"Artist - Title"`` patterns first, falls back to using
    the stem as the title with an empty artist.

    Returns:
        ``(artist, title)`` tuple.  Artist may be empty.
    """
    stem = audio_path.stem
    match = _FILENAME_RE.match(stem)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", stem.strip()


def _sanitize_tag(value: str) -> str:
    """Normalize a tag value for safe downstream use.

    Strips BOM, zero-width characters, control characters, and
    normalizes Unicode via NFKC (compatibility + composed form) so
    that fullwidth letters, ligatures, and other compatibility forms
    are converted to their basic equivalents.
    """
    # NFKC: compatibility decomposition then canonical composition
    # e.g. fullwidth "Ｓ" → "S", ligature "ﬁ" → "fi"
    value = unicodedata.normalize("NFKC", value)
    # Strip BOM and common invisible Unicode
    value = (
        value
        .replace("\ufeff", "")   # BOM
        .replace("\ufffe", "")   # reversed BOM
        .replace("\u200b", "")   # zero-width space
        .replace("\u200c", "")   # zero-width non-joiner
        .replace("\u200d", "")   # zero-width joiner
        .replace("\u200e", "")   # left-to-right mark
        .replace("\u200f", "")   # right-to-left mark
        .replace("\u202a", "")   # left-to-right embedding
        .replace("\u202c", "")   # pop directional formatting
    )
    # Remove control characters (Unicode category C) except whitespace
    value = "".join(
        c for c in value
        if c in ("\n", "\r", "\t", " ") or unicodedata.category(c)[0] != "C"
    )
    return value.strip()


def _read_tags(audio_path: Path) -> tuple[Optional[str], Optional[str]]:
    """Try to read title and artist from embedded audio tags via mutagen.

    Returns:
        ``(title_or_None, artist_or_None)``.
    """
    try:
        import mutagen
    except ImportError:
        logger.debug("mutagen not available — skipping tag read for %s", audio_path.name)
        return None, None

    try:
        mf = mutagen.File(str(audio_path))
        if mf is None or mf.tags is None:
            return None, None
    except Exception as exc:
        logger.debug("mutagen could not read %s: %s", audio_path.name, exc)
        return None, None

    title: Optional[str] = None
    artist: Optional[str] = None

    # ID3 tags (MP3, AIFF, etc.)
    for key in ("TIT2",):
        val = mf.tags.get(key)
        if val:
            title = str(val)
            break
    for key in ("TPE1", "TPE2"):
        val = mf.tags.get(key)
        if val:
            artist = str(val)
            break

    # Vorbis comments (FLAC, OGG) and MP4 atoms
    if title is None:
        for key in ("title", "\xa9nam"):
            vals = mf.tags.get(key)
            if vals:
                title = str(vals[0]) if isinstance(vals, list) else str(vals)
                break
    if artist is None:
        for key in ("artist", "\xa9ART", "albumartist", "aART"):
            vals = mf.tags.get(key)
            if vals:
                artist = str(vals[0]) if isinstance(vals, list) else str(vals)
                break

    # Sanitize tag values (normalize Unicode, strip invisible chars)
    if title:
        title = _sanitize_tag(title) or None
    if artist:
        artist = _sanitize_tag(artist) or None

    return title, artist


def resolve_metadata(audio_path: Path) -> SongMeta:
    """Resolve song title and artist for an audio file.

    Tries embedded tags first (via mutagen), falling back to filename
    parsing when tags are missing or unreadable.

    Returns:
        A :class:`SongMeta` with ``title``, ``artist``, and ``source``.
    """
    tag_title, tag_artist = _read_tags(audio_path)

    if tag_title:
        # Tags found — use them, artist may still be empty
        logger.debug(
            "Using embedded tags for %s: title=%r, artist=%r",
            audio_path.name, tag_title, tag_artist,
        )
        return SongMeta(
            title=tag_title,
            artist=tag_artist or "",
            source="tags",
        )

    # No title tag — fall back to filename
    fn_artist, fn_title = parse_filename(audio_path)
    logger.debug(
        "No tags for %s, using filename: title=%r, artist=%r",
        audio_path.name, fn_title, fn_artist,
    )
    return SongMeta(title=fn_title, artist=fn_artist, source="filename")
