from __future__ import annotations

import logging
from pathlib import Path

from whisper_transcriber.pipeline import Segment, Word

logger = logging.getLogger("chatter_split.pyannote")
UNKNOWN_SPEAKER = "UNKNOWN"


def _overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers_by_overlap(
    segments: list[Segment],
    speaker_turns: list[tuple[float, float, str]],
    *,
    max_gap_seconds: float = 1.0,
) -> list[tuple[str, Segment]]:
    if not segments:
        return []

    label_map: dict[str, str] = {}
    next_label = 1
    word_speakers: list[tuple[str, Word]] = []

    for segment in segments:
        words = segment.words or [Word(start=segment.start, end=segment.end, text=segment.text)]
        for word in words:
            best_score = 0.0
            best_raw_label: str | None = None
            for start, end, raw_label in speaker_turns:
                score = _overlap_seconds(word.start, word.end, start, end)
                if score > best_score:
                    best_score = score
                    best_raw_label = raw_label

            if best_raw_label is None:
                speaker = UNKNOWN_SPEAKER
            else:
                if best_raw_label not in label_map:
                    label_map[best_raw_label] = f"Speaker {next_label}"
                    next_label += 1
                speaker = label_map[best_raw_label]

            word_speakers.append((speaker, word))

    return _group_words_by_speaker(word_speakers, max_gap_seconds=max_gap_seconds)


def _group_words_by_speaker(
    word_speakers: list[tuple[str, Word]],
    *,
    max_gap_seconds: float,
) -> list[tuple[str, Segment]]:
    result: list[tuple[str, Segment]] = []
    current_speaker: str | None = None
    current_words: list[Word] = []

    def flush() -> None:
        nonlocal current_speaker, current_words
        if current_speaker is None or not current_words:
            return
        text = " ".join(word.text.strip() for word in current_words if word.text.strip())
        text = " ".join(text.split())
        if text:
            result.append(
                (
                    current_speaker,
                    Segment(
                        start=current_words[0].start,
                        end=current_words[-1].end,
                        text=text,
                        words=list(current_words),
                    ),
                )
            )
        current_speaker = None
        current_words = []

    for speaker, word in word_speakers:
        gap = 0.0 if not current_words else max(0.0, word.start - current_words[-1].end)
        starts_new_turn = (
            current_speaker is None
            or speaker != current_speaker
            or gap > max_gap_seconds
            or speaker == UNKNOWN_SPEAKER
            or current_speaker == UNKNOWN_SPEAKER
        )

        if starts_new_turn:
            flush()
            current_speaker = speaker
            current_words = [word]
        else:
            current_words.append(word)

    flush()
    return result


def _speaker_turns_from_pyannote(annotation) -> list[tuple[float, float, str]]:
    if hasattr(annotation, "itertracks"):
        return [
            (float(turn.start), float(turn.end), str(speaker))
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
    return [(float(turn.start), float(turn.end), str(speaker)) for turn, speaker in annotation]


class PyannoteDiarizer:
    def __init__(
        self,
        model_id: str = "pyannote/speaker-diarization-community-1",
        expected_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        hf_token: str | None = None,
        max_gap_seconds: float = 1.0,
    ) -> None:
        self._model_id = model_id
        self._expected_speakers = expected_speakers
        self._min_speakers = min_speakers
        self._max_speakers = max_speakers
        self._hf_token = hf_token
        self._max_gap_seconds = max_gap_seconds
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            try:
                from pyannote.audio import Pipeline
            except Exception as exc:
                raise RuntimeError(
                    "pyannote.audio is required for pyannote diarization backend. "
                    "Install dependencies and set HF_TOKEN."
                ) from exc
            logger.info("Loading pyannote diarization pipeline: %s", self._model_id)
            self._pipeline = Pipeline.from_pretrained(self._model_id, token=self._hf_token)
        return self._pipeline

    def assign_speakers(self, segments: list[Segment], audio_path: Path) -> list[tuple[str, Segment]]:
        if not segments:
            return []

        kwargs = {}
        if self._expected_speakers is not None:
            kwargs["num_speakers"] = self._expected_speakers
        else:
            if self._min_speakers is not None:
                kwargs["min_speakers"] = self._min_speakers
            if self._max_speakers is not None:
                kwargs["max_speakers"] = self._max_speakers

        diarization = self.pipeline(str(audio_path), **kwargs)
        turns = getattr(diarization, "exclusive_speaker_diarization", diarization)
        speaker_turns = _speaker_turns_from_pyannote(turns)
        return assign_speakers_by_overlap(segments, speaker_turns, max_gap_seconds=self._max_gap_seconds)
