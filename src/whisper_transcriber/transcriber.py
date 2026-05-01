from __future__ import annotations

import logging
from pathlib import Path

from faster_whisper import WhisperModel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from whisper_transcriber.pipeline import Segment

logger = logging.getLogger("chatter_split.transcriber")


class WhisperTranscriber:
    def __init__(self, model_name: str = "small", device: str = "auto", compute_type: str = "int8") -> None:
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: Path) -> list[Segment]:
        logger.info("Transcribing audio with Whisper")
        segments, info = self._model.transcribe(
            str(audio_path),
            language="ru",
            vad_filter=True,
            beam_size=5,
        )
        total_seconds = float(getattr(info, "duration", 0.0) or 0.0)
        result: list[Segment] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Whisper progress"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task_id = progress.add_task("transcribe", total=total_seconds if total_seconds > 0 else None)
            last_end = 0.0

            for s in segments:
                text = s.text.strip()
                if text:
                    result.append(Segment(start=s.start, end=s.end, text=text))

                if total_seconds > 0:
                    current_end = max(float(s.end), last_end)
                    progress.advance(task_id, current_end - last_end)
                    last_end = current_end
                else:
                    progress.advance(task_id, 1.0)

            if total_seconds > 0:
                progress.update(task_id, completed=total_seconds)

        logger.info("Speech recognition complete: %s text segments", len(result))
        logger.debug("Raw segments: %s", result)
        return result
