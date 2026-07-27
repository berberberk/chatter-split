from __future__ import annotations

from whisper_transcriber.diarizer import SpeakerDiarizer
from whisper_transcriber.pyannote_diarizer import PyannoteDiarizer


def build_diarizer(
    backend: str = "pyannote",
    expected_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    hf_token: str | None = None,
):
    if backend == "pyannote":
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN or HUGGINGFACE_TOKEN is required for the pyannote diarization backend. "
                "Set one of them in .env, or run with --diarizer-backend speechbrain for the experimental fallback."
            )
        return PyannoteDiarizer(
            expected_speakers=expected_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            hf_token=hf_token,
        )
    if backend == "speechbrain":
        return SpeakerDiarizer(expected_speakers=expected_speakers)
    raise ValueError("Unsupported diarizer backend. Use 'speechbrain' or 'pyannote'.")
