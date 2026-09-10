from whisper_transcriber.diarizer_factory import build_diarizer
from whisper_transcriber.diarizer import SpeakerDiarizer


def test_build_diarizer_supports_speechbrain_backend() -> None:
    diarizer = build_diarizer(backend="speechbrain", expected_speakers=4)
    assert isinstance(diarizer, SpeakerDiarizer)


def test_build_diarizer_supports_pyannote_backend() -> None:
    diarizer = build_diarizer(backend="pyannote", expected_speakers=4, hf_token="token")
    assert diarizer.__class__.__name__ == "PyannoteDiarizer"


def test_build_diarizer_requires_token_for_default_pyannote_backend() -> None:
    try:
        build_diarizer()
    except RuntimeError as exc:
        assert "HF_TOKEN" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")
