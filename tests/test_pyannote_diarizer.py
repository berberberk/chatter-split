from pathlib import Path

from whisper_transcriber.pipeline import Segment, Word
from whisper_transcriber.pyannote_diarizer import PyannoteDiarizer, assign_speakers_by_overlap


def test_assign_speakers_by_overlap_maps_words_to_best_overlap() -> None:
    segments = [
        Segment(
            start=0.0,
            end=3.0,
            text="A B C",
            words=[
                Word(start=0.0, end=0.8, text="A"),
                Word(start=1.0, end=1.8, text="B"),
                Word(start=2.0, end=2.8, text="C"),
            ],
        ),
    ]
    speaker_turns = [
        (0.0, 0.9, "SPEAKER_00"),
        (0.9, 1.9, "SPEAKER_01"),
        (1.9, 3.0, "SPEAKER_00"),
    ]

    labeled = assign_speakers_by_overlap(segments, speaker_turns)

    assert [speaker for speaker, _ in labeled] == ["Speaker 1", "Speaker 2", "Speaker 1"]
    assert [segment.text for _, segment in labeled] == ["A", "B", "C"]


def test_assign_speakers_by_overlap_uses_unknown_on_gap() -> None:
    segments = [
        Segment(
            start=0.0,
            end=2.0,
            text="A B",
            words=[
                Word(start=0.0, end=0.7, text="A"),
                Word(start=1.0, end=1.8, text="B"),
            ],
        ),
    ]
    speaker_turns = [(0.0, 0.7, "SPEAKER_99")]

    labeled = assign_speakers_by_overlap(segments, speaker_turns)

    assert [speaker for speaker, _ in labeled] == ["Speaker 1", "UNKNOWN"]


def test_assign_speakers_by_overlap_splits_on_long_gap() -> None:
    segments = [
        Segment(
            start=0.0,
            end=4.0,
            text="A B",
            words=[
                Word(start=0.0, end=0.5, text="A"),
                Word(start=3.0, end=3.5, text="B"),
            ],
        ),
    ]
    speaker_turns = [(0.0, 4.0, "SPEAKER_00")]

    labeled = assign_speakers_by_overlap(segments, speaker_turns, max_gap_seconds=1.0)

    assert [speaker for speaker, _ in labeled] == ["Speaker 1", "Speaker 1"]
    assert [segment.text for _, segment in labeled] == ["A", "B"]


def test_pyannote_diarizer_passes_exact_speaker_count() -> None:
    received: dict[str, int | str] = {}
    diarizer = PyannoteDiarizer(expected_speakers=3, hf_token="token")
    diarizer._pipeline = _FakePipeline(received)

    diarizer.assign_speakers([Segment(0.0, 1.0, "A", [Word(0.0, 0.5, "A")])], Path("audio.wav"))

    assert received["num_speakers"] == 3
    assert "min_speakers" not in received
    assert "max_speakers" not in received


def test_pyannote_diarizer_passes_speaker_bounds() -> None:
    received: dict[str, int | str] = {}
    diarizer = PyannoteDiarizer(min_speakers=2, max_speakers=5, hf_token="token")
    diarizer._pipeline = _FakePipeline(received)

    diarizer.assign_speakers([Segment(0.0, 1.0, "A", [Word(0.0, 0.5, "A")])], Path("audio.wav"))

    assert received["min_speakers"] == 2
    assert received["max_speakers"] == 5
    assert "num_speakers" not in received


class _FakeTurn:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class _FakeAnnotation:
    def itertracks(self, yield_label: bool = False):
        assert yield_label is True
        yield _FakeTurn(0.0, 1.0), "_", "SPEAKER_00"


class _FakePipeline:
    def __init__(self, received: dict[str, int | str]) -> None:
        self._received = received

    def __call__(self, audio_path: str, **kwargs):
        self._received["audio_path"] = audio_path
        self._received.update(kwargs)
        return _FakeAnnotation()
