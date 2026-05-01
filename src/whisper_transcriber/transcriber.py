from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel

from whisper_transcriber.pipeline import Segment


class WhisperTranscriber:
    def __init__(self, model_name: str = "small", device: str = "auto", compute_type: str = "int8") -> None:
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: Path) -> list[Segment]:
        segments, _ = self._model.transcribe(
            str(audio_path),
            language="ru",
            vad_filter=True,
            beam_size=5,
        )
        return [Segment(start=s.start, end=s.end, text=s.text.strip()) for s in segments if s.text.strip()]
