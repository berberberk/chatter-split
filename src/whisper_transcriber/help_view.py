from __future__ import annotations

import tomllib
from pathlib import Path

from rich.console import Console

ASCII_LOGO = r"""
      _           _   _             __         _ _ _
  ___| |__   __ _| |_| |_ ___ _ __ / /__ _ __ | (_) |_
 / __| '_ \ / _` | __| __/ _ \ '__/ / __| '_ \| | | __|
| (__| | | | (_| | |_| ||  __/ | / /\__ \ |_) | | | |_
 \___|_| |_|\__,_|\__|\__\___|_|/_/ |___/ .__/|_|_|\__|
                                        |_|
"""


def project_version(pyproject_path: Path) -> str:
    if not pyproject_path.exists():
        return "unknown"
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    return str(data.get("project", {}).get("version", "unknown"))


def print_make_help(console: Console | None = None) -> None:
    console = console or Console()
    root = Path(__file__).resolve().parents[2]
    version = project_version(root / "pyproject.toml")

    console.print(f"[yellow]{ASCII_LOGO}[/yellow]")
    console.print(f"[cyan]Version:[/cyan] {version}")
    console.print("\n[bold]Commands:[/bold]")
    console.print("  [green]make test[/green]  - run unit tests")
    console.print("  [green]make run[/green]   - transcribe inbox/input.mp3 to output/transcript.md")
    console.print("  [green]make api[/green]   - start FastAPI server on :8000")
    console.print("  [green]make lint[/green]  - compile-check sources")


if __name__ == "__main__":
    print_make_help()
