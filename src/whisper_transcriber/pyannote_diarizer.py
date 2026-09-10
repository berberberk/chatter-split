from __future__ import annotations

import logging
from pathlib import Path
import warnings

import torch
from faster_whisper.audio import decode_audio

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
    nearest_speaker_tolerance_seconds: float = 0.5,
    max_unknown_merge_duration_seconds: float = 0.5,
    max_unknown_merge_words: int = 2,
    max_unknown_bridge_duration_seconds: float = 30.0,
    max_unknown_adoption_gap_seconds: float = 1.0,
) -> list[tuple[str, Segment]]:
    if not segments:
        return []

    label_map: dict[str, str] = {}
    next_label = 1
    word_speakers: list[tuple[str, Word]] = []

    for segment in segments:
        words = _normalized_words_for_segment(segment)
        for word in words:
            best_raw_label = _best_speaker_for_word(
                word,
                speaker_turns,
                nearest_speaker_tolerance_seconds=nearest_speaker_tolerance_seconds,
            )

            if best_raw_label is None:
                speaker = UNKNOWN_SPEAKER
            else:
                if best_raw_label not in label_map:
                    label_map[best_raw_label] = f"Speaker {next_label}"
                    next_label += 1
                speaker = label_map[best_raw_label]

            word_speakers.append((speaker, word))

    grouped = _group_words_by_speaker(word_speakers, max_gap_seconds=max_gap_seconds)
    return _merge_short_unknown_turns(
        grouped,
        max_unknown_merge_duration_seconds=max_unknown_merge_duration_seconds,
        max_unknown_merge_words=max_unknown_merge_words,
        max_unknown_bridge_duration_seconds=max_unknown_bridge_duration_seconds,
        max_unknown_adoption_gap_seconds=max_unknown_adoption_gap_seconds,
    )


def _normalized_words_for_segment(segment: Segment, *, min_duration_seconds: float = 0.05) -> list[Word]:
    words = segment.words or [Word(start=segment.start, end=segment.end, text=segment.text)]
    normalized: list[Word] = []
    for word in words:
        start = float(word.start)
        end = float(word.end)
        if end <= start:
            end = start + min_duration_seconds
        normalized.append(Word(start=start, end=end, text=word.text))
    return normalized


def _distance_to_turn(word: Word, turn_start: float, turn_end: float) -> float:
    if word.end < turn_start:
        return turn_start - word.end
    if word.start > turn_end:
        return word.start - turn_end
    return 0.0


def _best_speaker_for_word(
    word: Word,
    speaker_turns: list[tuple[float, float, str]],
    *,
    nearest_speaker_tolerance_seconds: float,
) -> str | None:
    best_score = 0.0
    best_raw_label: str | None = None
    for start, end, raw_label in speaker_turns:
        score = _overlap_seconds(word.start, word.end, start, end)
        if score > best_score:
            best_score = score
            best_raw_label = raw_label

    if best_raw_label is not None:
        return best_raw_label

    nearest_distance = nearest_speaker_tolerance_seconds
    nearest_raw_label: str | None = None
    for start, end, raw_label in speaker_turns:
        distance = _distance_to_turn(word, start, end)
        if distance <= nearest_distance:
            nearest_distance = distance
            nearest_raw_label = raw_label
    return nearest_raw_label


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
        )

        if starts_new_turn:
            flush()
            current_speaker = speaker
            current_words = [word]
        else:
            current_words.append(word)

    flush()
    return result


def _merge_short_unknown_turns(
    turns: list[tuple[str, Segment]],
    *,
    max_unknown_merge_duration_seconds: float,
    max_unknown_merge_words: int,
    max_unknown_bridge_duration_seconds: float,
    max_unknown_adoption_gap_seconds: float,
) -> list[tuple[str, Segment]]:
    if not turns:
        return []

    merged: list[tuple[str, Segment]] = []
    index = 0
    while index < len(turns):
        speaker, segment = turns[index]
        previous_turn = merged[-1] if merged else None
        next_turn = turns[index + 1] if index + 1 < len(turns) else None
        if speaker == UNKNOWN_SPEAKER:
            if (
                previous_turn is not None
                and next_turn is not None
                and previous_turn[0] == next_turn[0]
                and _turn_duration(segment) <= max_unknown_bridge_duration_seconds
            ):
                merged[-1] = (previous_turn[0], _merge_segments(previous_turn[1], segment, next_turn[1]))
                index += 2
                continue

            if not _is_short_unknown_turn(
                segment,
                max_unknown_merge_duration_seconds=max_unknown_merge_duration_seconds,
                max_unknown_merge_words=max_unknown_merge_words,
            ):
                adopted = _adopt_long_unknown_turn(
                    segment,
                    previous_turn=previous_turn,
                    next_turn=next_turn,
                    max_unknown_adoption_gap_seconds=max_unknown_adoption_gap_seconds,
                    max_unknown_bridge_duration_seconds=max_unknown_bridge_duration_seconds,
                )
                if adopted is not None:
                    adopted_side, adopted_speaker, adopted_segment = adopted
                    if adopted_side == "previous":
                        merged[-1] = (adopted_speaker, adopted_segment)
                        index += 1
                    else:
                        merged.append((adopted_speaker, adopted_segment))
                        index += 2
                    continue

        merged.append((speaker, segment))
        index += 1
    return merged


def _adopt_long_unknown_turn(
    segment: Segment,
    *,
    previous_turn: tuple[str, Segment] | None,
    next_turn: tuple[str, Segment] | None,
    max_unknown_adoption_gap_seconds: float,
    max_unknown_bridge_duration_seconds: float,
) -> tuple[str, str, Segment] | None:
    if _turn_duration(segment) > max_unknown_bridge_duration_seconds:
        return None

    previous_gap = None
    if previous_turn is not None:
        previous_gap = max(0.0, segment.start - previous_turn[1].end)

    next_gap = None
    if next_turn is not None:
        next_gap = max(0.0, next_turn[1].start - segment.end)

    if previous_gap is not None and previous_gap <= max_unknown_adoption_gap_seconds:
        if next_gap is None or previous_gap <= next_gap:
            return "previous", previous_turn[0], _merge_segments(previous_turn[1], segment)

    if next_gap is not None and next_gap <= max_unknown_adoption_gap_seconds:
        return "next", next_turn[0], _merge_segments(segment, next_turn[1])

    return None


def _turn_duration(segment: Segment) -> float:
    return max(0.0, segment.end - segment.start)


def _is_short_unknown_turn(
    segment: Segment,
    *,
    max_unknown_merge_duration_seconds: float,
    max_unknown_merge_words: int,
) -> bool:
    word_count = len(segment.words) if segment.words else len(segment.text.split())
    return word_count <= max_unknown_merge_words or _turn_duration(segment) <= max_unknown_merge_duration_seconds


def _merge_segments(*segments: Segment) -> Segment:
    words = [word for segment in segments for word in segment.words]
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    text = " ".join(text.split())
    return Segment(start=segments[0].start, end=segments[-1].end, text=text, words=words)


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
        nearest_speaker_tolerance_seconds: float = 0.5,
        max_unknown_merge_duration_seconds: float = 0.5,
        max_unknown_merge_words: int = 2,
        max_unknown_bridge_duration_seconds: float = 30.0,
        max_unknown_adoption_gap_seconds: float = 1.0,
    ) -> None:
        self._model_id = model_id
        self._expected_speakers = expected_speakers
        self._min_speakers = min_speakers
        self._max_speakers = max_speakers
        self._hf_token = hf_token
        self._max_gap_seconds = max_gap_seconds
        self._nearest_speaker_tolerance_seconds = nearest_speaker_tolerance_seconds
        self._max_unknown_merge_duration_seconds = max_unknown_merge_duration_seconds
        self._max_unknown_merge_words = max_unknown_merge_words
        self._max_unknown_bridge_duration_seconds = max_unknown_bridge_duration_seconds
        self._max_unknown_adoption_gap_seconds = max_unknown_adoption_gap_seconds
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    from pyannote.audio import Pipeline
            except Exception as exc:
                raise RuntimeError(
                    "pyannote.audio is required for pyannote diarization backend. "
                    "Install dependencies and set HF_TOKEN."
                ) from exc
            logger.info("Loading pyannote diarization pipeline: %s", self._model_id)
            try:
                self._pipeline = Pipeline.from_pretrained(self._model_id, token=self._hf_token)
            except Exception as exc:
                raise RuntimeError(
                    "Cannot load pyannote diarization model. Make sure the HF token has access to "
                    f"{self._model_id} and that you accepted the model terms on Hugging Face."
                ) from exc
            if self._pipeline is None:
                raise RuntimeError(
                    "Cannot load pyannote diarization model. Make sure the HF token has access to "
                    f"{self._model_id} and that you accepted the model terms on Hugging Face."
                )
        return self._pipeline

    def prepare(self) -> None:
        self.pipeline

    def _load_audio(self, audio_path: Path) -> dict[str, object]:
        audio = decode_audio(str(audio_path), sampling_rate=16000)
        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        return {"waveform": waveform, "sample_rate": 16000}

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

        diarization = self.pipeline(self._load_audio(audio_path), **kwargs)
        turns = getattr(diarization, "exclusive_speaker_diarization", diarization)
        speaker_turns = _speaker_turns_from_pyannote(turns)
        return assign_speakers_by_overlap(
            segments,
            speaker_turns,
            max_gap_seconds=self._max_gap_seconds,
            nearest_speaker_tolerance_seconds=self._nearest_speaker_tolerance_seconds,
            max_unknown_merge_duration_seconds=self._max_unknown_merge_duration_seconds,
            max_unknown_merge_words=self._max_unknown_merge_words,
            max_unknown_bridge_duration_seconds=self._max_unknown_bridge_duration_seconds,
            max_unknown_adoption_gap_seconds=self._max_unknown_adoption_gap_seconds,
        )
