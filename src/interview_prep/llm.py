"""Gemini client configuration kept outside the graph node."""

from __future__ import annotations

import os
from functools import cache

from google import genai
from langsmith import wrappers

DEFAULT_MODEL = "gemini-3.5-flash-lite"


def get_model_name() -> str:
    """Return the model pinned for class, with an environment override."""

    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL)


@cache
def get_gemini_client() -> genai.Client:
    """Return the shared traceable Gemini client."""

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY before calling Gemini."
        )

    raw_client = genai.Client(api_key=api_key)
    return wrappers.wrap_gemini(
        raw_client,
        tracing_extra={
            "tags": ["interview-prep", "requirement-extraction"],
            "metadata": {
                "provider": "google",
                "model": get_model_name(),
                "teaching_slice": "extract_requirements",
            },
        },
    )
