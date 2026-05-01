from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchaudio
from sklearn.cluster import AgglomerativeClustering
from speechbrain.inference.speaker import EncoderClassifier

from whisper_transcriber.pipeline import Segment


class SpeakerDiarizer:
    def __init__(self, threshold: float = 0.65, min_speakers: int = 1, max_speakers: int = 8) -> None:
        self._threshold = threshold
        self._min_speakers = min_speakers
        self._max_speakers = max_speakers
        self._encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    def assign_speakers(self, segments: list[Segment], audio_path: Path) -> list[tuple[str, Segment]]:
        if not segments:
            return []

        waveform, sample_rate = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        embeddings: list[np.ndarray] = []
        for seg in segments:
            start = int(max(seg.start, 0.0) * sample_rate)
            end = int(max(seg.end, seg.start + 0.05) * sample_rate)
            chunk = waveform[:, start:end]
            if chunk.numel() == 0:
                chunk = waveform[:, start : min(start + int(0.2 * sample_rate), waveform.shape[1])]
            if chunk.numel() == 0:
                chunk = torch.zeros((1, int(0.2 * sample_rate)))

            emb = self._encoder.encode_batch(chunk).squeeze().detach().cpu().numpy()
            embeddings.append(emb)

        matrix = np.vstack(embeddings)
        n_segments = len(segments)
        if n_segments == 1:
            labels = np.array([0])
        else:
            labels = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=self._threshold,
            ).fit_predict(matrix)

        labels = self._normalize_labels(labels, n_segments)
        return [(f"Speaker {label + 1}", segment) for label, segment in zip(labels, segments)]

    def _normalize_labels(self, labels: np.ndarray, n_segments: int) -> np.ndarray:
        unique = sorted(set(int(v) for v in labels.tolist()))
        mapping = {old: new for new, old in enumerate(unique)}
        normalized = np.array([mapping[int(v)] for v in labels], dtype=int)

        speakers = len(set(normalized.tolist()))
        if speakers < self._min_speakers:
            return np.array([i % self._min_speakers for i in range(n_segments)], dtype=int)
        if speakers > self._max_speakers:
            return np.array([i % self._max_speakers for i in range(n_segments)], dtype=int)
        return normalized
