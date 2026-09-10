from __future__ import annotations

import tomllib
from pathlib import Path
import logging
import os
import time

import typer
from rich.console import Console

from whisper_transcriber.diarizer_factory import build_diarizer
from whisper_transcriber.env_config import load_environment
from whisper_transcriber.input_resolver import resolve_input_audio
from whisper_transcriber.metrics import build_transcript_metrics, write_transcript_metrics
from whisper_transcriber.pipeline import TranscriptionPipeline, TranscriptionResult
from whisper_transcriber.transcriber import WhisperTranscriber

ASCII_LOGO = r"""
      _           _   _             __         _ _ _
  ___| |__   __ _| |_| |_ ___ _ __ / /__ _ __ | (_) |_
 / __| '_ \ / _` | __| __/ _ \ '__/ / __| '_ \| | | __|
| (__| | | | (_| | |_| ||  __/ | / /\__ \ |_) | | | |_
 \___|_| |_|\__,_|\__|\__\___|_|/_/ |___/ .__/|_|_|\__|
                                        |_|
"""

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    add_help_option=False,
    context_settings={"help_option_names": []},
    help="Chatter Split CLI",
)
console = Console()
logger = logging.getLogger("chatter_split.cli")
_LOGGING_CONFIGURED = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INBOX_DIR = PROJECT_ROOT / "inbox"
OUTPUT_DIR = PROJECT_ROOT / "output"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
load_environment(PROJECT_ROOT)


def build_pipeline(
    expected_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    diarizer_backend: str | None = None,
) -> TranscriptionPipeline:
    logger.info("Preparing transcription pipeline")
    backend = diarizer_backend or os.getenv("CHATTERSPLIT_DIARIZER_BACKEND", "pyannote")
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    expected_speakers = expected_speakers if expected_speakers is not None else _optional_env_int(
        "CHATTERSPLIT_EXPECTED_SPEAKERS"
    )
    min_speakers = min_speakers if min_speakers is not None else _optional_env_int("CHATTERSPLIT_MIN_SPEAKERS")
    max_speakers = max_speakers if max_speakers is not None else _optional_env_int("CHATTERSPLIT_MAX_SPEAKERS")
    _validate_speaker_options(expected_speakers, min_speakers, max_speakers)
    diarizer = build_diarizer(
        backend=backend,
        expected_speakers=expected_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        hf_token=hf_token,
    )
    return TranscriptionPipeline(
        transcriber=WhisperTranscriber(model_name="small"),
        diarizer=diarizer,
    )


def _optional_env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer.") from exc


def _validate_speaker_options(
    expected_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> None:
    if expected_speakers is not None and (min_speakers is not None or max_speakers is not None):
        raise RuntimeError("Use either --speakers for an exact count or --min-speakers/--max-speakers bounds, not both.")
    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise RuntimeError("--min-speakers cannot be greater than --max-speakers.")


def speaker_count_mode(
    expected_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> str:
    if expected_speakers is not None or _optional_env_int("CHATTERSPLIT_EXPECTED_SPEAKERS") is not None:
        return "exact"
    if (
        min_speakers is not None
        or max_speakers is not None
        or _optional_env_int("CHATTERSPLIT_MIN_SPEAKERS") is not None
        or _optional_env_int("CHATTERSPLIT_MAX_SPEAKERS") is not None
    ):
        return "bounds"
    return "auto"


def selected_diarizer_backend(diarizer_backend: str | None = None) -> str:
    return diarizer_backend or os.getenv("CHATTERSPLIT_DIARIZER_BACKEND", "pyannote")


def run_pipeline_with_metrics(
    pipeline: TranscriptionPipeline,
    input_file: Path,
    output_file: Path,
    *,
    diarizer_backend: str,
    speaker_mode: str,
) -> TranscriptionResult:
    started = time.perf_counter()
    result = pipeline.run_detailed(input_file)
    runtime_seconds = time.perf_counter() - started
    output_file.write_text(result.markdown, encoding="utf-8")
    metrics = build_transcript_metrics(
        result,
        runtime_seconds=runtime_seconds,
        diarizer_backend=diarizer_backend,
        speaker_count_mode=speaker_mode,
    )
    write_transcript_metrics(output_file.with_suffix(".metrics.json"), metrics)
    return result


def configure_logging() -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    level_name = os.getenv("CHATTERSPLIT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    class ColorLevelFormatter(logging.Formatter):
        COLORS = {
            logging.INFO: "\x1b[32m",
            logging.WARNING: "\x1b[33m",
            logging.ERROR: "\x1b[31m",
            logging.CRITICAL: "\x1b[31m",
        }
        RESET = "\x1b[0m"

        def format(self, record: logging.LogRecord) -> str:
            ts = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
            color = self.COLORS.get(record.levelno, "")
            level_label = f"{color}{record.levelname}{self.RESET}" if color else record.levelname
            return f"{ts} | {level_label} | {record.getMessage()}"

    handler = logging.StreamHandler()
    handler.setFormatter(ColorLevelFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)

    for noisy_logger in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "faster_whisper",
        "speechbrain",
        "urllib3",
        "asyncio",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _LOGGING_CONFIGURED = True


def project_version() -> str:
    if not PYPROJECT_PATH.exists():
        return "unknown"
    with PYPROJECT_PATH.open("rb") as f:
        data = tomllib.load(f)
    return str(data.get("project", {}).get("version", "unknown"))


def _custom_help() -> str:
    version = project_version()
    return (
        f"[yellow]{ASCII_LOGO}[/yellow]\n"
        "[bold]Chatter Split CLI[/bold]\n\n"
        f"[cyan]Version:[/cyan] {version}\n\n"
        "[bold]Usage:[/bold]\n"
        "  transcribe [OPTIONS] COMMAND [ARGS]...\n\n"
        "[bold]Options:[/bold]\n"
        "  --help, -h  Show this message and exit.\n\n"
        "[bold]Commands:[/bold]\n"
        "  run [--speakers N] [--min-speakers N] [--max-speakers N]\n"
        "  api\n"
    )


@app.callback(invoke_without_command=True)
def root_callback(
    ctx: typer.Context,
    help_flag: bool = typer.Option(False, "--help", "-h", is_eager=True),
) -> None:
    configure_logging()
    if help_flag or ctx.invoked_subcommand is None:
        console.print(_custom_help())
        raise typer.Exit()


def run_transcription(input_file: Path, output_file: Path, expected_speakers: int | None = None) -> Path:
    if not input_file.exists():
        logger.error("Input file does not exist: %s", input_file)
        raise typer.BadParameter(f"Input file does not exist: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Starting transcription: %s", input_file.name)
    backend = selected_diarizer_backend()
    pipeline = build_pipeline(expected_speakers=expected_speakers, diarizer_backend=backend)
    logger.info("Running speech recognition and speaker separation")
    run_pipeline_with_metrics(
        pipeline,
        input_file,
        output_file,
        diarizer_backend=backend,
        speaker_mode=speaker_count_mode(expected_speakers=expected_speakers),
    )
    logger.info("Done. Transcript saved: %s", output_file)
    return output_file


@app.command("run")
def run_command(
    speakers: int | None = typer.Option(
        None,
        "--speakers",
        "-s",
        min=1,
        max=12,
        help="Exact number of speakers. Use only when known.",
    ),
    min_speakers: int | None = typer.Option(
        None,
        "--min-speakers",
        min=1,
        max=12,
        help="Minimum number of speakers when the exact count is unknown.",
    ),
    max_speakers: int | None = typer.Option(
        None,
        "--max-speakers",
        min=1,
        max=12,
        help="Maximum number of speakers when the exact count is unknown.",
    ),
    diarizer_backend: str | None = typer.Option(
        None,
        "--diarizer-backend",
        help="Diarization backend: pyannote or speechbrain. Defaults to pyannote.",
    ),
) -> None:
    input_file = resolve_input_audio(INBOX_DIR)
    if input_file is None:
        logger.error("Input audio file was not found in: %s", INBOX_DIR)
        raise typer.BadParameter(f"Input file does not exist: {INBOX_DIR / 'input.<ext>'}")
    output_file = OUTPUT_DIR / "transcript.md"
    if diarizer_backend and diarizer_backend not in {"speechbrain", "pyannote"}:
        raise typer.BadParameter("Unsupported diarizer backend. Use 'speechbrain' or 'pyannote'.")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Starting transcription: %s", input_file.name)
    backend = selected_diarizer_backend(diarizer_backend)
    mode = speaker_count_mode(expected_speakers=speakers, min_speakers=min_speakers, max_speakers=max_speakers)
    try:
        pipeline = build_pipeline(
            expected_speakers=speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            diarizer_backend=backend,
        )
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    logger.info("Running speech recognition and speaker separation")
    try:
        run_pipeline_with_metrics(
            pipeline,
            input_file,
            output_file,
            diarizer_backend=backend,
            speaker_mode=mode,
        )
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    logger.info("Done. Transcript saved: %s", output_file)
    saved = output_file
    console.print(f"[green]Saved transcript:[/green] {saved}")


@app.command("api")
def api_command(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    logger.info("Starting API server on %s:%s", host, port)
    uvicorn.run("whisper_transcriber.api:app", host=host, port=port)


if __name__ == "__main__":
    configure_logging()
    app()
