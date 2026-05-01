from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException

from whisper_transcriber.cli import INBOX_DIR, OUTPUT_DIR, build_pipeline
from whisper_transcriber.pipeline import TranscriptionPipeline


PipelineFactory = Callable[[], TranscriptionPipeline]


def create_app(
    pipeline_factory: PipelineFactory = build_pipeline,
    inbox_dir: Path = INBOX_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> FastAPI:
    app = FastAPI(title="Whisper Speech Transcriber", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/transcribe")
    def transcribe() -> dict[str, str]:
        input_file = inbox_dir / "input.mp3"
        output_file = output_dir / "transcript.md"

        if not input_file.exists():
            raise HTTPException(status_code=400, detail=f"Input file does not exist: {input_file}")

        output_dir.mkdir(parents=True, exist_ok=True)
        pipeline = pipeline_factory()
        transcript = pipeline.run(input_file)
        output_file.write_text(transcript, encoding="utf-8")

        return {
            "output_file": str(output_file),
            "transcript": transcript,
        }

    return app


app = create_app()
