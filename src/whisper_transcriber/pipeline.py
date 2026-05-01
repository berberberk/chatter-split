from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from whisper_transcriber.formatter import render_markdown_dialogue


@dataclass(slots=True)
class Segment:
    start: float
    end: float
    text: str


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> list[Segment]: ...


class Diarizer(Protocol):
    def assign_speakers(self, segments: list[Segment], audio_path: Path) -> list[tuple[str, Segment]]: ...


class TranscriptionPipeline:
    def __init__(self, transcriber: Transcriber, diarizer: Diarizer) -> None:
        self._transcriber = transcriber
        self._diarizer = diarizer

    def run(self, audio_path: Path) -> str:
        segments = self._transcriber.transcribe(audio_path)
        speaker_segments = self._diarizer.assign_speakers(segments, audio_path)
        turns = [(speaker, segment.text) for speaker, segment in speaker_segments]
        return render_markdown_dialogue(turns)
