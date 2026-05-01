from pathlib import Path


def test_makefile_targets_exist() -> None:
    content = Path("Makefile").read_text(encoding="utf-8")

    assert "test:" in content
    assert "run:" in content
    assert "api:" in content
    assert "pytest -q" in content
