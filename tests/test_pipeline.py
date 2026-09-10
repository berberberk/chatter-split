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
    assert "- [00:00.00-00:04.00] Привет Здравствуйте" in text
    assert "Speaker 2:" in text


def test_pipeline_prepares_diarizer_before_transcribing() -> None:
    class FailingPrepareDiarizer:
        def prepare(self) -> None:
            raise RuntimeError("prepare failed")

        def assign_speakers(self, segments, _):
            return [("Speaker 1", segment) for segment in segments]

    class TrackingTranscriber:
        called = False

        def transcribe(self, _: Path):
            self.called = True
            return [Segment(0.0, 1.0, "Привет")]

    transcriber = TrackingTranscriber()
    pipeline = TranscriptionPipeline(transcriber=transcriber, diarizer=FailingPrepareDiarizer())

    try:
        pipeline.run(Path("inbox/input.mp3"))
    except RuntimeError as exc:
        assert "prepare failed" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")

    assert transcriber.called is False
