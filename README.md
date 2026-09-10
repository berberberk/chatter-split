# ChatterSplit

A speech transcription tool using Whisper plus speaker diarization, with clean dialogue output.

## Output format

```md
Speaker 1:
- [00:00.00-00:05.20] ...

Speaker 2:
- [00:05.20-00:08.90] ...
```

## Project structure

- `inbox/input.<ext>` - input audio file (`mp3`, `m4a`, `wav`, `flac`, `ogg`, `aac`, `mp4`, `webm`).
- `output/transcript.md` - generated transcript.
- `src/whisper_transcriber/transcriber.py` - speech-to-text via Whisper (`faster-whisper`).
- `src/whisper_transcriber/diarizer.py` - experimental SpeechBrain fallback backend.
- `src/whisper_transcriber/pyannote_diarizer.py` - pyannote diarization backend with word-level speaker assignment.
- `src/whisper_transcriber/diarizer_factory.py` - diarization backend factory.
- `src/whisper_transcriber/pipeline.py` - orchestration layer.
- `src/whisper_transcriber/formatter.py` - Markdown dialogue renderer.
- `src/whisper_transcriber/cli.py` - CLI commands.
- `src/whisper_transcriber/api.py` - FastAPI endpoints.

## Installation

```bash
uv sync
cp .env.example .env
# set HF_TOKEN in .env for the default pyannote backend
# optionally set CHATTERSPLIT_EXPECTED_SPEAKERS only when the exact count is known
# optionally set CHATTERSPLIT_MIN_SPEAKERS / CHATTERSPLIT_MAX_SPEAKERS when only bounds are known
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

If you know the number of speakers, pass it explicitly:

```bash
uv run transcribe run --speakers 4
```

If you only know bounds, pass them instead of `--speakers`:

```bash
uv run transcribe run --min-speakers 2 --max-speakers 6
```

Use the local SpeechBrain fallback only when you cannot use a Hugging Face token:

```bash
uv run transcribe run --diarizer-backend speechbrain
```

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
