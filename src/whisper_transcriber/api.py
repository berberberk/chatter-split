from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Callable

from fastapi import FastAPI, HTTPException

from whisper_transcriber.cli import INBOX_DIR, OUTPUT_DIR, build_pipeline, selected_diarizer_backend, speaker_count_mode
from whisper_transcriber.env_config import load_environment
from whisper_transcriber.input_resolver import resolve_input_audio
from whisper_transcriber.metrics import build_transcript_metrics, write_transcript_metrics
from whisper_transcriber.pipeline import TranscriptionPipeline


PipelineFactory = Callable[[], TranscriptionPipeline]
logger = logging.getLogger("chatter_split.api")
load_environment(Path(__file__).resolve().parents[2])


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
        started = time.perf_counter()
        result = pipeline.run_detailed(input_file)
        runtime_seconds = time.perf_counter() - started
        transcript = result.markdown
        output_file.write_text(transcript, encoding="utf-8")
        metrics_file = output_file.with_suffix(".metrics.json")
        metrics = build_transcript_metrics(
            result,
            runtime_seconds=runtime_seconds,
            diarizer_backend=selected_diarizer_backend(),
            speaker_count_mode=speaker_count_mode(),
        )
        write_transcript_metrics(metrics_file, metrics)
        logger.info("API transcription complete: %s", output_file)

        return {
            "output_file": str(output_file),
            "metrics_file": str(metrics_file),
            "transcript": transcript,
        }

    return app


app = create_app()
