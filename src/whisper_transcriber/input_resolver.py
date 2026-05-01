from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS = (
    ".mp3",
    ".m4a",
    ".wav",
    ".flac",
    ".ogg",
    ".aac",
    ".mp4",
    ".webm",
)


def resolve_input_audio(inbox_dir: Path) -> Path | None:
    for ext in SUPPORTED_EXTENSIONS:
        candidate = inbox_dir / f"input{ext}"
        if candidate.exists() and candidate.is_file():
            return candidate

    matches = sorted(p for p in inbox_dir.glob("input.*") if p.is_file())
    if matches:
        return matches[0]
    return None
