from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path

import numpy as np
import torch
from faster_whisper.audio import decode_audio
from sklearn.cluster import AgglomerativeClustering
from speechbrain.inference.speaker import EncoderClassifier

from whisper_transcriber.pipeline import Segment

logger = logging.getLogger("chatter_split.diarizer")


@dataclass(frozen=True)
class DiarizationConfig:
    expected_speakers: int | None = 4
    threshold: float = 0.65
    min_speakers: int = 1
    max_speakers: int = 8
    min_turn_duration_seconds: float = 1.2


class SpeakerDiarizer:
    def __init__(
        self,
        threshold: float = 0.65,
        min_speakers: int = 1,
        max_speakers: int = 8,
        expected_speakers: int | None = 4,
        min_turn_duration_seconds: float = 1.2,
    ) -> None:
        env_expected = os.getenv("CHATTERSPLIT_EXPECTED_SPEAKERS")
        if env_expected and expected_speakers == 4:
            expected_speakers = int(env_expected)

        self._config = DiarizationConfig(
            expected_speakers=expected_speakers,
            threshold=threshold,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            min_turn_duration_seconds=min_turn_duration_seconds,
        )
        self._encoder: EncoderClassifier | None = None

    @property
    def encoder(self) -> EncoderClassifier:
        if self._encoder is None:
            logger.info("Loading speaker model")
            self._encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        return self._encoder

    def assign_speakers(self, segments: list[Segment], audio_path: Path) -> list[tuple[str, Segment]]:
        if not segments:
            logger.warning("No speech segments found for diarization")
            return []

        logger.info("Starting speaker separation")
        sample_rate = 16000
        audio = decode_audio(str(audio_path), sampling_rate=sample_rate)
        waveform = torch.from_numpy(audio).float().unsqueeze(0)

        embeddings: list[np.ndarray] = []
        for segment in segments:
            start = int(max(segment.start, 0.0) * sample_rate)
            end = int(max(segment.end, segment.start + 0.2) * sample_rate)
            chunk = waveform[:, start:end]
            if chunk.numel() == 0:
                chunk = waveform[:, start : min(start + int(0.2 * sample_rate), waveform.shape[1])]
            if chunk.numel() == 0:
                chunk = torch.zeros((1, int(0.2 * sample_rate)))

            emb = self.encoder.encode_batch(chunk).squeeze().detach().cpu().numpy()
            embeddings.append(emb)

        matrix = np.vstack(embeddings)
        labels = self._cluster_embeddings(matrix)
        labels = self._normalize_labels(labels, len(segments))
        labels = self._smooth_short_turns(labels, segments)
        labels = self._relabel_by_first_appearance(labels)
        result = [(f"Speaker {label + 1}", segment) for label, segment in zip(labels, segments)]
        speaker_count = len({speaker for speaker, _ in result})
        logger.info("Speaker separation complete: %s speakers detected", speaker_count)
        logger.debug("Speaker labels: %s", [speaker for speaker, _ in result])
        return result

    def _cluster_embeddings(self, matrix: np.ndarray) -> np.ndarray:
        n_segments = len(matrix)
        if n_segments == 1:
            return np.array([0], dtype=int)

        expected = self._config.expected_speakers
        if expected is not None:
            n_clusters = max(self._config.min_speakers, min(expected, self._config.max_speakers, n_segments))
            return AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="cosine",
                linkage="average",
            ).fit_predict(matrix)

        return AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=self._config.threshold,
        ).fit_predict(matrix)

    def _normalize_labels(self, labels: np.ndarray, n_segments: int) -> np.ndarray:
        unique = sorted(set(int(v) for v in labels.tolist()))
        mapping = {old: new for new, old in enumerate(unique)}
        normalized = np.array([mapping[int(v)] for v in labels], dtype=int)

        speakers = len(set(normalized.tolist()))
        if speakers < self._config.min_speakers:
            return np.array([i % self._config.min_speakers for i in range(n_segments)], dtype=int)
        if speakers > self._config.max_speakers:
            counts = np.bincount(normalized)
            keep_list = np.argsort(counts)[-self._config.max_speakers :].tolist()
            keep = set(keep_list)
            fallback = int(keep_list[0])
            capped = np.array([label if label in keep else fallback for label in normalized], dtype=int)

            remap_unique = sorted(set(capped.tolist()))
            remap = {old: new for new, old in enumerate(remap_unique)}
            return np.array([remap[int(v)] for v in capped], dtype=int)
        return normalized

    def _smooth_short_turns(self, labels: np.ndarray, segments: list[Segment]) -> np.ndarray:
        smoothed = labels.copy()
        changed = True
        while changed:
            changed = False
            runs = self._label_runs(smoothed, segments)
            for run_index, (start, end, label, duration) in enumerate(runs):
                if duration >= self._config.min_turn_duration_seconds or len(runs) == 1:
                    continue

                previous_run = runs[run_index - 1] if run_index > 0 else None
                next_run = runs[run_index + 1] if run_index < len(runs) - 1 else None
                if previous_run and next_run and previous_run[2] == next_run[2] and previous_run[2] != label:
                    replacement = previous_run[2]
                    smoothed[start:end] = replacement
                    changed = True
                    break
        return smoothed

    def _label_runs(self, labels: np.ndarray, segments: list[Segment]) -> list[tuple[int, int, int, float]]:
        runs: list[tuple[int, int, int, float]] = []
        start = 0
        for index in range(1, len(labels) + 1):
            if index < len(labels) and labels[index] == labels[start]:
                continue
            duration = max(0.0, segments[index - 1].end - segments[start].start)
            runs.append((start, index, int(labels[start]), duration))
            start = index
        return runs

    def _relabel_by_first_appearance(self, labels: np.ndarray) -> np.ndarray:
        mapping: dict[int, int] = {}
        next_label = 0
        relabeled: list[int] = []
        for label in labels.tolist():
            label = int(label)
            if label not in mapping:
                mapping[label] = next_label
                next_label += 1
            relabeled.append(mapping[label])
        return np.array(relabeled, dtype=int)
