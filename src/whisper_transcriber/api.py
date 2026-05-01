from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException

from whisper_transcriber.cli import INBOX_DIR, OUTPUT_DIR, build_pipeline
from whisper_transcriber.input_resolver import resolve_input_audio
from whisper_transcriber.pipeline import TranscriptionPipeline


PipelineFactory = Callable[[], TranscriptionPipeline]
logger = logging.getLogger("chatter_split.api")


def create_app(
    pipeline_factory: PipelineFactory = build_pipeline,
    inbox_dir: Path = INBOX_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> FastAPI:
    app = FastAPI(title="Whisper Speech Transcriber", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        logger.info("Health check request received")
        return {"status": "ok"}

    @app.post("/transcribe")
    def transcribe() -> dict[str, str]:
        input_file = resolve_input_audio(inbox_dir)
        output_file = output_dir / "transcript.md"

        if input_file is None:
            logger.error("Input audio file was not found in: %s", inbox_dir)
            raise HTTPException(status_code=400, detail=f"Input file does not exist: {inbox_dir / 'input.<ext>'}")

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("API transcription request: %s", input_file.name)
        pipeline = pipeline_factory()
        transcript = pipeline.run(input_file)
        output_file.write_text(transcript, encoding="utf-8")
        logger.info("API transcription complete: %s", output_file)

        return {
            "output_file": str(output_file),
            "transcript": transcript,
        }

    return app


app = create_app()
