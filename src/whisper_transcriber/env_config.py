from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_environment(project_root: Path) -> None:
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)

    hf_token = os.getenv("HF_TOKEN", "").strip()
    if hf_token:
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
        os.environ.setdefault("HF_HUB_TOKEN", hf_token)
