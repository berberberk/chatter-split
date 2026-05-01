# ChatterSplit

A local, free speech transcription tool for MP3 files using Whisper, with speaker separation and clean dialogue output.

## Output format

```md
Speaker 1:
- ...

Speaker 2:
- ...
```

## Project structure

- `inbox/input.<ext>` - input audio file (`mp3`, `m4a`, `wav`, `flac`, `ogg`, `aac`, `mp4`, `webm`).
- `output/transcript.md` - generated transcript.
- `src/whisper_transcriber/transcriber.py` - speech-to-text via Whisper (`faster-whisper`).
- `src/whisper_transcriber/diarizer.py` - speaker assignment using voice embeddings and clustering.
- `src/whisper_transcriber/pipeline.py` - orchestration layer.
- `src/whisper_transcriber/formatter.py` - Markdown dialogue renderer.
- `src/whisper_transcriber/cli.py` - CLI commands.
- `src/whisper_transcriber/api.py` - FastAPI endpoints.

## Installation

```bash
uv sync
```

## CLI usage

Run transcription:

```bash
uv run transcribe run
```

Or with Make:

```bash
  make run
```

The transcript will be saved to `output/transcript.md`.

## API usage

Start API server:

```bash
uv run transcribe api
```

Or with Make:

```bash
make api
```

### Endpoints

- `GET /health` -> `{"status":"ok"}`
- `POST /transcribe` -> transcribes `inbox/input.<ext>`, writes `output/transcript.md`, returns JSON with `output_file` and `transcript`.

## Make commands

```bash
make help
make test
make run
make api
make lint
```

## Testing

```bash
make test
```
