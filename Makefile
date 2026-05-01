.PHONY: test run api lint

test:
	uv run --with pytest --with pytest-mock pytest -q

run:
	uv run transcribe run

api:
	uv run uvicorn whisper_transcriber.api:app --host 0.0.0.0 --port 8000

lint:
	uv run python -m compileall src tests
