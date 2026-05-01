from __future__ import annotations

import tomllib
from pathlib import Path
import logging
import os

import typer
from rich.console import Console

from whisper_transcriber.diarizer import SpeakerDiarizer
from whisper_transcriber.input_resolver import resolve_input_audio
from whisper_transcriber.pipeline import TranscriptionPipeline
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


def build_pipeline() -> TranscriptionPipeline:
    logger.info("Preparing transcription pipeline")
    return TranscriptionPipeline(
        transcriber=WhisperTranscriber(model_name="small"),
        diarizer=SpeakerDiarizer(),
    )


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
        "  run\n"
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


def run_transcription(input_file: Path, output_file: Path) -> Path:
    if not input_file.exists():
        logger.error("Input file does not exist: %s", input_file)
        raise typer.BadParameter(f"Input file does not exist: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Starting transcription: %s", input_file.name)
    pipeline = build_pipeline()
    logger.info("Running speech recognition and speaker separation")
    markdown = pipeline.run(input_file)
    output_file.write_text(markdown, encoding="utf-8")
    logger.info("Done. Transcript saved: %s", output_file)
    return output_file


@app.command("run")
def run_command() -> None:
    input_file = resolve_input_audio(INBOX_DIR)
    if input_file is None:
        logger.error("Input audio file was not found in: %s", INBOX_DIR)
        raise typer.BadParameter(f"Input file does not exist: {INBOX_DIR / 'input.<ext>'}")
    output_file = OUTPUT_DIR / "transcript.md"
    saved = run_transcription(input_file, output_file)
    console.print(f"[green]Saved transcript:[/green] {saved}")


@app.command("api")
def api_command(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    logger.info("Starting API server on %s:%s", host, port)
    uvicorn.run("whisper_transcriber.api:app", host=host, port=port)


if __name__ == "__main__":
    configure_logging()
    app()
