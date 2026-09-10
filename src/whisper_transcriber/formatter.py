from __future__ import annotations

from typing import Any, Iterable


def _join_turn_messages(messages: list[str]) -> str:
    text = " ".join(message.strip() for message in messages if message.strip())
    return " ".join(text.split())


def _format_timestamp(seconds: float) -> str:
    minutes, remainder = divmod(max(seconds, 0.0), 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remainder:05.2f}"
    return f"{minutes:02d}:{remainder:05.2f}"


def render_markdown_dialogue(turns: Iterable[tuple[str, str | Any]]) -> str:
    blocks: list[tuple[str, list[str], float | None, float | None]] = []

    for speaker, payload in turns:
        if hasattr(payload, "text") and hasattr(payload, "start") and hasattr(payload, "end"):
            text = str(payload.text)
            start = float(payload.start)
            end = float(payload.end)
        else:
            text = payload
            start = None
            end = None
        clean = text.strip()
        if not clean:
            continue

        if blocks and blocks[-1][0] == speaker:
            block_speaker, messages, block_start, block_end = blocks[-1]
            messages.append(clean)
            if start is not None:
                block_start = start if block_start is None else min(block_start, start)
            if end is not None:
                block_end = end if block_end is None else max(block_end, end)
            blocks[-1] = (block_speaker, messages, block_start, block_end)
        else:
            blocks.append((speaker, [clean], start, end))

    lines: list[str] = []
    for i, (speaker, messages, start, end) in enumerate(blocks):
        time_prefix = ""
        if start is not None and end is not None:
            time_prefix = f"[{_format_timestamp(start)}-{_format_timestamp(end)}] "
        lines.append(f"{speaker}:")
        lines.append(f"- {time_prefix}{_join_turn_messages(messages)}")
        if i != len(blocks) - 1:
            lines.append("")

    return "\n".join(lines) + ("\n" if lines else "")
