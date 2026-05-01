from __future__ import annotations

from typing import Iterable


def render_markdown_dialogue(turns: Iterable[tuple[str, str]]) -> str:
    blocks: list[tuple[str, list[str]]] = []

    for speaker, text in turns:
        clean = text.strip()
        if not clean:
            continue

        if blocks and blocks[-1][0] == speaker:
            blocks[-1][1].append(clean)
        else:
            blocks.append((speaker, [clean]))

    lines: list[str] = []
    for i, (speaker, messages) in enumerate(blocks):
        lines.append(f"{speaker}:")
        for message in messages:
            lines.append(f"- {message}")
        if i != len(blocks) - 1:
            lines.append("")

    return "\n".join(lines) + ("\n" if lines else "")
