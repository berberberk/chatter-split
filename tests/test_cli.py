from pathlib import Path

from typer.testing import CliRunner

from whisper_transcriber.cli import app, build_pipeline
from whisper_transcriber.pipeline import Segment, TranscriptionResult


class FakePipeline:
    def run(self, _input_path: Path) -> str:
        return self.run_detailed(_input_path).markdown

    def run_detailed(self, _input_path: Path) -> TranscriptionResult:
        segment = Segment(start=0.0, end=1.0, text="Test")
        return TranscriptionResult(
            markdown="Speaker 1:\n- [00:00.00-00:01.00] Test\n",
            source_segments=[segment],
            speaker_segments=[("Speaker 1", segment)],
        )


class FailingPipeline:
    def run(self, _input_path: Path) -> str:
        raise RuntimeError("Cannot load pyannote diarization model.")

    def run_detailed(self, _input_path: Path) -> TranscriptionResult:
        raise RuntimeError("Cannot load pyannote diarization model.")


runner = CliRunner()


def test_cli_run_writes_output_file_for_mp3(monkeypatch, tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    output = tmp_path / "output"
    inbox.mkdir()
    output.mkdir()
    (inbox / "input.mp3").write_bytes(b"fake")

    monkeypatch.setattr("whisper_transcriber.cli.INBOX_DIR", inbox)
    monkeypatch.setattr("whisper_transcriber.cli.OUTPUT_DIR", output)
    monkeypatch.setattr("whisper_transcriber.cli.build_pipeline", lambda **kwargs: FakePipeline())

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 0
    content = (output / "transcript.md").read_text(encoding="utf-8")
    assert "Speaker 1:" in content
    assert (output / "transcript.metrics.json").exists()


def test_cli_run_writes_output_file_for_m4a(monkeypatch, tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    output = tmp_path / "output"
    inbox.mkdir()
    output.mkdir()
    (inbox / "input.m4a").write_bytes(b"fake")

    monkeypatch.setattr("whisper_transcriber.cli.INBOX_DIR", inbox)
    monkeypatch.setattr("whisper_transcriber.cli.OUTPUT_DIR", output)
    monkeypatch.setattr("whisper_transcriber.cli.build_pipeline", lambda **kwargs: FakePipeline())

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 0
    content = (output / "transcript.md").read_text(encoding="utf-8")
    assert "Speaker 1:" in content
    assert (output / "transcript.metrics.json").exists()


def test_cli_run_passes_expected_speakers(monkeypatch, tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    output = tmp_path / "output"
    inbox.mkdir()
    output.mkdir()
    (inbox / "input.m4a").write_bytes(b"fake")
    received: dict[str, int | None] = {}

    def fake_build_pipeline(
        expected_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        diarizer_backend: str | None = None,
    ) -> FakePipeline:
        received["expected_speakers"] = expected_speakers
        received["min_speakers"] = min_speakers
        received["max_speakers"] = max_speakers
        received["diarizer_backend"] = diarizer_backend
        return FakePipeline()

    monkeypatch.setattr("whisper_transcriber.cli.INBOX_DIR", inbox)
    monkeypatch.setattr("whisper_transcriber.cli.OUTPUT_DIR", output)
    monkeypatch.setattr("whisper_transcriber.cli.build_pipeline", fake_build_pipeline)

    result = runner.invoke(app, ["run", "--speakers", "4"])

    assert result.exit_code == 0
    assert received["expected_speakers"] == 4


def test_cli_run_passes_backend(monkeypatch, tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    output = tmp_path / "output"
    inbox.mkdir()
    output.mkdir()
    (inbox / "input.m4a").write_bytes(b"fake")
    received: dict[str, str | int | None] = {}

    def fake_build_pipeline(
        expected_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        diarizer_backend: str | None = None,
    ) -> FakePipeline:
        received["expected_speakers"] = expected_speakers
        received["min_speakers"] = min_speakers
        received["max_speakers"] = max_speakers
        received["diarizer_backend"] = diarizer_backend
        return FakePipeline()

    monkeypatch.setattr("whisper_transcriber.cli.INBOX_DIR", inbox)
    monkeypatch.setattr("whisper_transcriber.cli.OUTPUT_DIR", output)
    monkeypatch.setattr("whisper_transcriber.cli.build_pipeline", fake_build_pipeline)

    result = runner.invoke(app, ["run", "--diarizer-backend", "pyannote"])

    assert result.exit_code == 0
    assert received["diarizer_backend"] == "pyannote"


def test_cli_run_passes_speaker_bounds(monkeypatch, tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    output = tmp_path / "output"
    inbox.mkdir()
    output.mkdir()
    (inbox / "input.m4a").write_bytes(b"fake")
    received: dict[str, str | int | None] = {}

    def fake_build_pipeline(
        expected_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        diarizer_backend: str | None = None,
    ) -> FakePipeline:
        received["expected_speakers"] = expected_speakers
        received["min_speakers"] = min_speakers
        received["max_speakers"] = max_speakers
        received["diarizer_backend"] = diarizer_backend
        return FakePipeline()

    monkeypatch.setattr("whisper_transcriber.cli.INBOX_DIR", inbox)
    monkeypatch.setattr("whisper_transcriber.cli.OUTPUT_DIR", output)
    monkeypatch.setattr("whisper_transcriber.cli.build_pipeline", fake_build_pipeline)

    result = runner.invoke(app, ["run", "--min-speakers", "2", "--max-speakers", "5"])

    assert result.exit_code == 0
    assert received["expected_speakers"] is None
    assert received["min_speakers"] == 2
    assert received["max_speakers"] == 5


def test_cli_run_rejects_exact_and_bounds_together(monkeypatch, tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    output = tmp_path / "output"
    inbox.mkdir()
    output.mkdir()
    (inbox / "input.m4a").write_bytes(b"fake")

    monkeypatch.setattr("whisper_transcriber.cli.INBOX_DIR", inbox)
    monkeypatch.setattr("whisper_transcriber.cli.OUTPUT_DIR", output)

    result = runner.invoke(app, ["run", "--speakers", "2", "--min-speakers", "1"])

    assert result.exit_code != 0
    assert "exact count" in result.output or "Use either" in result.output


def test_cli_run_reports_pipeline_runtime_error(monkeypatch, tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    output = tmp_path / "output"
    inbox.mkdir()
    output.mkdir()
    (inbox / "input.m4a").write_bytes(b"fake")

    monkeypatch.setattr("whisper_transcriber.cli.INBOX_DIR", inbox)
    monkeypatch.setattr("whisper_transcriber.cli.OUTPUT_DIR", output)
    monkeypatch.setattr("whisper_transcriber.cli.build_pipeline", lambda **kwargs: FailingPipeline())

    result = runner.invoke(app, ["run"])

    assert result.exit_code != 0
    assert "Cannot load pyannote diarization model" in result.output
    assert not (output / "transcript.md").exists()


def test_build_pipeline_defaults_to_pyannote_without_speaker_count(monkeypatch) -> None:
    received: dict[str, str | int | None] = {}

    class FakeTranscriber:
        pass

    def fake_build_diarizer(
        backend: str,
        expected_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        hf_token: str | None = None,
    ):
        received["backend"] = backend
        received["expected_speakers"] = expected_speakers
        received["min_speakers"] = min_speakers
        received["max_speakers"] = max_speakers
        received["hf_token"] = hf_token
        return object()

    monkeypatch.delenv("CHATTERSPLIT_DIARIZER_BACKEND", raising=False)
    monkeypatch.delenv("CHATTERSPLIT_EXPECTED_SPEAKERS", raising=False)
    monkeypatch.delenv("CHATTERSPLIT_MIN_SPEAKERS", raising=False)
    monkeypatch.delenv("CHATTERSPLIT_MAX_SPEAKERS", raising=False)
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setattr("whisper_transcriber.cli.WhisperTranscriber", lambda model_name: FakeTranscriber())
    monkeypatch.setattr("whisper_transcriber.cli.build_diarizer", fake_build_diarizer)

    build_pipeline()

    assert received == {
        "backend": "pyannote",
        "expected_speakers": None,
        "min_speakers": None,
        "max_speakers": None,
        "hf_token": "token",
    }


def test_build_pipeline_fails_without_token_before_loading_whisper(monkeypatch) -> None:
    loaded_whisper = False

    def fake_transcriber(model_name: str):
        nonlocal loaded_whisper
        loaded_whisper = True
        return object()

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("CHATTERSPLIT_DIARIZER_BACKEND", raising=False)
    monkeypatch.setattr("whisper_transcriber.cli.WhisperTranscriber", fake_transcriber)

    try:
        build_pipeline()
    except RuntimeError as exc:
        assert "HF_TOKEN" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")

    assert loaded_whisper is False


def test_cli_run_fails_when_input_missing(monkeypatch, tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    output = tmp_path / "output"
    inbox.mkdir()
    output.mkdir()

    monkeypatch.setattr("whisper_transcriber.cli.INBOX_DIR", inbox)
    monkeypatch.setattr("whisper_transcriber.cli.OUTPUT_DIR", output)

    result = runner.invoke(app, ["run"])

    assert result.exit_code != 0
    assert not (output / "transcript.md").exists()
