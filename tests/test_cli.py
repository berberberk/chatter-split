from pathlib import Path

from typer.testing import CliRunner

from whisper_transcriber.cli import app


class FakePipeline:
    def run(self, _input_path: Path) -> str:
        return "Speaker 1:\n- Test\n"


runner = CliRunner()


def test_cli_run_writes_output_file_for_mp3(monkeypatch, tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    output = tmp_path / "output"
    inbox.mkdir()
    output.mkdir()
    (inbox / "input.mp3").write_bytes(b"fake")

    monkeypatch.setattr("whisper_transcriber.cli.INBOX_DIR", inbox)
    monkeypatch.setattr("whisper_transcriber.cli.OUTPUT_DIR", output)
    monkeypatch.setattr("whisper_transcriber.cli.build_pipeline", lambda: FakePipeline())

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 0
    content = (output / "transcript.md").read_text(encoding="utf-8")
    assert "Speaker 1:" in content


def test_cli_run_writes_output_file_for_m4a(monkeypatch, tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    output = tmp_path / "output"
    inbox.mkdir()
    output.mkdir()
    (inbox / "input.m4a").write_bytes(b"fake")

    monkeypatch.setattr("whisper_transcriber.cli.INBOX_DIR", inbox)
    monkeypatch.setattr("whisper_transcriber.cli.OUTPUT_DIR", output)
    monkeypatch.setattr("whisper_transcriber.cli.build_pipeline", lambda: FakePipeline())

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 0
    content = (output / "transcript.md").read_text(encoding="utf-8")
    assert "Speaker 1:" in content


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
