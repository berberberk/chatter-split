from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from whisper_transcriber.formatter import render_markdown_dialogue


@dataclass(slots=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(slots=True)
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass(slots=True)
class TranscriptionResult:
    markdown: str
    source_segments: list[Segment]
    speaker_segments: list[tuple[str, Segment]]


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> list[Segment]: ...


class Diarizer(Protocol):
    def assign_speakers(self, segments: list[Segment], audio_path: Path) -> list[tuple[str, Segment]]: ...


class TranscriptionPipeline:
    def __init__(self, transcriber: Transcriber, diarizer: Diarizer) -> None:
        self._transcriber = transcriber
        self._diarizer = diarizer

    def run(self, audio_path: Path) -> str:
        return self.run_detailed(audio_path).markdown

    def run_detailed(self, audio_path: Path) -> TranscriptionResult:
        prepare = getattr(self._diarizer, "prepare", None)
        if callable(prepare):
            prepare()
        segments = self._transcriber.transcribe(audio_path)
        speaker_segments = self._diarizer.assign_speakers(segments, audio_path)
        markdown = render_markdown_dialogue(speaker_segments)
        return TranscriptionResult(markdown=markdown, source_segments=segments, speaker_segments=speaker_segments)
