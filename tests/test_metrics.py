import json
from pathlib import Path

from whisper_transcriber.metrics import build_transcript_metrics, write_transcript_metrics
from whisper_transcriber.pipeline import Segment, TranscriptionResult, Word


def test_build_transcript_metrics_counts_turns_words_and_unknowns() -> None:
    source = [
        Segment(
            start=0.0,
            end=2.0,
            text="A B C",
            words=[
                Word(start=0.0, end=0.4, text="A"),
                Word(start=0.5, end=0.5, text="B"),
                Word(start=1.0, end=1.3, text="C"),
            ],
        )
    ]
    speaker_segments = [
        ("Speaker 1", Segment(start=0.0, end=0.4, text="A", words=[source[0].words[0]])),
        ("UNKNOWN", Segment(start=0.5, end=0.5, text="B", words=[source[0].words[1]])),
        ("Speaker 2", Segment(start=1.0, end=1.3, text="C", words=[source[0].words[2]])),
    ]
    result = TranscriptionResult(markdown="", source_segments=source, speaker_segments=speaker_segments)

    metrics = build_transcript_metrics(
        result,
        runtime_seconds=12.34567,
        diarizer_backend="pyannote",
        speaker_count_mode="auto",
    )

    assert metrics["speaker_count"] == 2
    assert metrics["turn_count"] == 3
    assert metrics["unknown_turn_count"] == 1
    assert metrics["unknown_turn_ratio"] == 0.333
    assert metrics["word_count"] == 3
    assert metrics["unknown_word_count"] == 1
    assert metrics["unknown_word_ratio"] == 0.333
    assert metrics["avg_words_per_turn"] == 1.0
    assert metrics["zero_duration_word_count"] == 1
    assert metrics["runtime_seconds"] == 12.346
    assert metrics["diarizer_backend"] == "pyannote"
    assert metrics["speaker_count_mode"] == "auto"


def test_write_transcript_metrics_writes_json(tmp_path: Path) -> None:
    metrics_path = tmp_path / "transcript.metrics.json"

    write_transcript_metrics(metrics_path, {"speaker_count": 1})

    assert json.loads(metrics_path.read_text(encoding="utf-8")) == {"speaker_count": 1}
