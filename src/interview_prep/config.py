"""Runtime configuration loaded from the project's optional ``.env`` file."""

from __future__ import annotations

from functools import cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"


@cache
def load_environment() -> bool:
    """Load project-local environment values without replacing shell settings.

    Exported environment variables take precedence, which keeps CI and deployed
    environments predictable while allowing local development through ``.env``.
    """

    return load_dotenv(DOTENV_PATH, override=False)
