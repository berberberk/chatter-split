from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from whisper_transcriber.diarizer import SpeakerDiarizer
from whisper_transcriber.pipeline import TranscriptionPipeline
from whisper_transcriber.transcriber import WhisperTranscriber

app = typer.Typer(add_completion=False)
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INBOX_DIR = PROJECT_ROOT / "inbox"
OUTPUT_DIR = PROJECT_ROOT / "output"


def build_pipeline() -> TranscriptionPipeline:
    return TranscriptionPipeline(
        transcriber=WhisperTranscriber(model_name="small"),
        diarizer=SpeakerDiarizer(),
    )


def run_transcription(input_file: Path, output_file: Path) -> Path:
    if not input_file.exists():
        raise typer.BadParameter(f"Input file does not exist: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    pipeline = build_pipeline()
    markdown = pipeline.run(input_file)
    output_file.write_text(markdown, encoding="utf-8")
    return output_file


@app.command("run")
def run_command() -> None:
    input_file = INBOX_DIR / "input.mp3"
    output_file = OUTPUT_DIR / "transcript.md"
    saved = run_transcription(input_file, output_file)
    console.print(f"Saved transcript: {saved}")


@app.command("api")
def api_command(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run("whisper_transcriber.api:app", host=host, port=port)


if __name__ == "__main__":
    app()
