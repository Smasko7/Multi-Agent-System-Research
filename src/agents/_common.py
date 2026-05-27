"""Shared helpers for agent nodes."""

import re

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def extract_text(content) -> str:
    """Normalize LLM `response.content` to a plain string.

    Handles two quirks:
    - Gemini Flash-Lite returns a list of content-block dicts like
      `[{"type": "text", "text": ...}]` instead of a plain string; we flatten it.
    - Qwen3 reasoning models (Groq) emit `<think>...</think>` blocks before the
      actual answer; we strip them so downstream JSON parsing isn't fooled by
      braces inside the reasoning trace.
    """
    if isinstance(content, list):
        text = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    else:
        text = content or ""
    return _THINK_BLOCK.sub("", text).strip()
