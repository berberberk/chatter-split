from whisper_transcriber.formatter import render_markdown_dialogue


def test_render_markdown_dialogue_groups_consecutive_same_speaker() -> None:
    turns = [
        ("Speaker 1", "Привет"),
        ("Speaker 1", "Как дела?"),
        ("Speaker 2", "Нормально"),
        ("Speaker 1", "Отлично"),
    ]

    output = render_markdown_dialogue(turns)

    expected = (
        "Speaker 1:\n"
        "- Привет\n"
        "- Как дела?\n\n"
        "Speaker 2:\n"
        "- Нормально\n\n"
        "Speaker 1:\n"
        "- Отлично\n"
    )
    assert output == expected


def test_render_markdown_dialogue_skips_empty_lines() -> None:
    turns = [
        ("Speaker 1", "  "),
        ("Speaker 2", "Ответ"),
    ]

    output = render_markdown_dialogue(turns)

    assert output == "Speaker 2:\n- Ответ\n"
