import numpy as np

from whisper_transcriber.diarizer import SpeakerDiarizer
from whisper_transcriber.pipeline import Segment


def test_normalize_labels_does_not_cycle_when_too_many_speakers() -> None:
    diarizer = SpeakerDiarizer(max_speakers=3)

    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    normalized = diarizer._normalize_labels(labels, n_segments=8)

    # Must be capped to <= max_speakers without round-robin oscillation.
    assert len(set(normalized.tolist())) <= 3
    assert normalized.tolist() != [0, 1, 2, 0, 1, 2, 0, 1]


def test_cluster_embeddings_respects_expected_speakers() -> None:
    diarizer = SpeakerDiarizer(expected_speakers=4, max_speakers=8)
    matrix = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [-1.0, 0.0],
            [-0.9, -0.1],
            [0.0, -1.0],
            [0.1, -0.9],
        ]
    )

    labels = diarizer._cluster_embeddings(matrix)

    assert len(set(labels.tolist())) == 4


def test_smooth_short_turns_removes_isolated_false_switch() -> None:
    diarizer = SpeakerDiarizer(min_turn_duration_seconds=1.0)
    labels = np.array([0, 1, 0], dtype=int)
    segments = [
        Segment(start=0.0, end=3.0, text="First"),
        Segment(start=3.0, end=3.4, text="False switch"),
        Segment(start=3.4, end=6.0, text="Second"),
    ]

    smoothed = diarizer._smooth_short_turns(labels, segments)

    assert smoothed.tolist() == [0, 0, 0]


def test_smooth_short_turns_merges_short_run_into_longer_neighbor() -> None:
    diarizer = SpeakerDiarizer(min_turn_duration_seconds=3.0)
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=int)
    segments = [
        Segment(start=0.0, end=2.0, text="A"),
        Segment(start=2.0, end=4.0, text="B"),
        Segment(start=4.0, end=4.5, text="C"),
        Segment(start=4.5, end=5.0, text="D"),
        Segment(start=5.0, end=8.0, text="E"),
        Segment(start=8.0, end=11.0, text="F"),
    ]

    smoothed = diarizer._smooth_short_turns(labels, segments)

    assert smoothed.tolist() == [0, 0, 1, 1, 2, 2]


def test_relabel_by_first_appearance_keeps_output_labels_compact() -> None:
    diarizer = SpeakerDiarizer()

    relabeled = diarizer._relabel_by_first_appearance(np.array([3, 3, 8, 3, 5]))

    assert relabeled.tolist() == [0, 0, 1, 0, 2]
