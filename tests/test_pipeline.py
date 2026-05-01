from pathlib import Path

from whisper_transcriber.pipeline import Segment, TranscriptionPipeline


class StubTranscriber:
    def transcribe(self, _: Path):
        return [
            Segment(0.0, 2.0, "Привет"),
            Segment(2.0, 4.0, "Здравствуйте"),
            Segment(4.0, 5.0, "Пока"),
        ]


class StubDiarizer:
    def assign_speakers(self, segments, _):
        labels = ["Speaker 1", "Speaker 1", "Speaker 2"]
        return list(zip(labels, segments))


def test_pipeline_builds_markdown() -> None:
    pipeline = TranscriptionPipeline(transcriber=StubTranscriber(), diarizer=StubDiarizer())

    text = pipeline.run(Path("inbox/input.mp3"))

    assert "Speaker 1:" in text
    assert "- Привет" in text
    assert "Speaker 2:" in text
