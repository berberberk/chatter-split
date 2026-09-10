from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from whisper_transcriber.pipeline import Segment, TranscriptionResult

UNKNOWN_SPEAKER = "UNKNOWN"


def build_transcript_metrics(
    result: TranscriptionResult,
    *,
    runtime_seconds: float,
    diarizer_backend: str,
    speaker_count_mode: str,
) -> dict[str, Any]:
    speaker_segments = result.speaker_segments
    speaker_labels = {speaker for speaker, _ in speaker_segments if speaker != UNKNOWN_SPEAKER}
    turn_count = len(speaker_segments)
    unknown_turn_count = sum(1 for speaker, _ in speaker_segments if speaker == UNKNOWN_SPEAKER)
    word_count = _word_count([segment for _, segment in speaker_segments])
    unknown_word_count = _word_count([segment for speaker, segment in speaker_segments if speaker == UNKNOWN_SPEAKER])
    zero_duration_word_count = sum(
        1
        for segment in result.source_segments
        for word in segment.words
        if float(word.end) <= float(word.start)
    )

    return {
        "speaker_count": len(speaker_labels),
        "turn_count": turn_count,
        "unknown_turn_count": unknown_turn_count,
        "unknown_turn_ratio": _ratio(unknown_turn_count, turn_count),
        "word_count": word_count,
        "unknown_word_count": unknown_word_count,
        "unknown_word_ratio": _ratio(unknown_word_count, word_count),
        "avg_words_per_turn": _ratio(word_count, turn_count),
        "zero_duration_word_count": zero_duration_word_count,
        "runtime_seconds": round(runtime_seconds, 3),
        "diarizer_backend": diarizer_backend,
        "speaker_count_mode": speaker_count_mode,
    }


def write_transcript_metrics(metrics_path: Path, metrics: dict[str, Any]) -> None:
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _word_count(segments: list[Segment]) -> int:
    count = 0
    for segment in segments:
        count += len(segment.words) if segment.words else len(segment.text.split())
    return count


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 3)
