from pathlib import Path

from fastapi.testclient import TestClient

from whisper_transcriber.api import create_app
from whisper_transcriber.pipeline import Segment, TranscriptionResult


class FakePipeline:
    def run(self, _input_path: Path) -> str:
        return self.run_detailed(_input_path).markdown

    def run_detailed(self, _input_path: Path) -> TranscriptionResult:
        segment = Segment(start=0.0, end=1.0, text="Hello")
        return TranscriptionResult(
            markdown="Speaker 1:\n- [00:00.00-00:01.00] Hello\n",
            source_segments=[segment],
            speaker_segments=[("Speaker 1", segment)],
        )


def test_health_endpoint() -> None:
    app = create_app(lambda: FakePipeline())
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_transcribe_endpoint_writes_file_for_mp3(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    output = tmp_path / "output"
    inbox.mkdir()
    output.mkdir()
    (inbox / "input.mp3").write_bytes(b"fake")

    app = create_app(lambda: FakePipeline(), inbox_dir=inbox, output_dir=output)
    client = TestClient(app)

    response = client.post("/transcribe")

    assert response.status_code == 200
    body = response.json()
    assert body["output_file"].endswith("transcript.md")
    assert body["metrics_file"].endswith("transcript.metrics.json")
    assert "Speaker 1" in body["transcript"]
    assert (output / "transcript.md").exists()
    assert (output / "transcript.metrics.json").exists()


def test_transcribe_endpoint_writes_file_for_m4a(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    output = tmp_path / "output"
    inbox.mkdir()
    output.mkdir()
    (inbox / "input.m4a").write_bytes(b"fake")

    app = create_app(lambda: FakePipeline(), inbox_dir=inbox, output_dir=output)
    client = TestClient(app)

    response = client.post("/transcribe")

    assert response.status_code == 200
    body = response.json()
    assert "Speaker 1" in body["transcript"]
    assert (output / "transcript.md").exists()
    assert (output / "transcript.metrics.json").exists()


def test_transcribe_endpoint_missing_input_returns_400(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    output = tmp_path / "output"
    inbox.mkdir()
    output.mkdir()

    app = create_app(lambda: FakePipeline(), inbox_dir=inbox, output_dir=output)
    client = TestClient(app)

    response = client.post("/transcribe")

    assert response.status_code == 400
    assert "Input file does not exist" in response.json()["detail"]
