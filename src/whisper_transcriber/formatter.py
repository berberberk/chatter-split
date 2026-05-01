from __future__ import annotations

from typing import Iterable


def _join_turn_messages(messages: list[str]) -> str:
    text = " ".join(message.strip() for message in messages if message.strip())
    return " ".join(text.split())


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
        lines.append(f"- {_join_turn_messages(messages)}")
        if i != len(blocks) - 1:
            lines.append("")

    return "\n".join(lines) + ("\n" if lines else "")
