"""Shared helpers for agent nodes."""


def extract_text(content) -> str:
    """Normalize LLM `response.content` to a plain string.

    Some Gemini models (notably Flash-Lite via langchain-google-genai) return
    content as a list of content-block dicts like `[{"type": "text", "text": ...}]`
    instead of a plain string. Stringifying the list directly produces ugly output
    in the UI, so we flatten it here.
    """
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return content or ""
