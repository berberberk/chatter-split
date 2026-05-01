.DEFAULT_GOAL := help
.PHONY: help test run api lint

help:
	uv run python -m whisper_transcriber.help_view

test:
	uv run --with pytest --with pytest-mock pytest -q

run:
	uv run transcribe run

api:
	uv run uvicorn whisper_transcriber.api:app --host 0.0.0.0 --port 8000

lint:
	uv run python -m compileall src tests
