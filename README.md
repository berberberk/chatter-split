# Whisper Speech Transcriber

Локальный бесплатный транскрибатор MP3 в Markdown c разделением по спикерам.

## Формат результата

```md
Speaker 1:
- ...

Speaker 2:
- ...
```

## Структура

- `inbox/input.mp3` - входной файл.
- `output/transcript.md` - результат.
- `src/whisper_transcriber/transcriber.py` - распознавание речи через Whisper.
- `src/whisper_transcriber/diarizer.py` - назначение спикеров по эмбеддингам.
- `src/whisper_transcriber/pipeline.py` - оркестрация шагов.
- `src/whisper_transcriber/formatter.py` - рендер Markdown.
- `src/whisper_transcriber/cli.py` - CLI команды.
- `src/whisper_transcriber/api.py` - FastAPI эндпоинты.

## Установка

```bash
uv sync
```

## CLI

```bash
uv run transcribe run
```

или через make:

```bash
make run
```

Результат сохраняется в `output/transcript.md`.

## FastAPI

Запуск API:

```bash
uv run transcribe api
```

или через make:

```bash
make api
```

Эндпоинты:

- `GET /health` -> `{"status":"ok"}`
- `POST /transcribe` -> запускает транскрибацию `inbox/input.mp3`, пишет `output/transcript.md` и возвращает JSON с путём и текстом.

## Make команды

```bash
make test
make run
make api
make lint
```

## Тесты

```bash
make test
```
