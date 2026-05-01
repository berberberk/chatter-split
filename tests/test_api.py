from pathlib import Path

from fastapi.testclient import TestClient

from whisper_transcriber.api import create_app


class FakePipeline:
    def run(self, _input_path: Path) -> str:
        return "Speaker 1:\n- Hello\n"


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
    assert "Speaker 1" in body["transcript"]
    assert (output / "transcript.md").exists()


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
