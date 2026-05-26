"""Shared state definition for the LangGraph agent pipeline."""

from typing import Annotated, TypedDict


def _take_max(a: int, b: int) -> int:
    """Reducer for parallel scalar writes — keep the larger value.

    Both parallel researchers write `iteration + 1`, so this returns the same
    value either way; it just lets LangGraph merge the concurrent updates.
    Wrapped (instead of using builtin `max`) because LangGraph introspects the
    signature, and builtins have no inspectable signature.
    """
    return a if a > b else b


class AgentState(TypedDict):
    query: str

    # Fan-out: two researchers with different angles
    research_findings_pro: str         # supporting / corroborating evidence
    research_findings_skeptical: str   # counter-evidence, limitations, criticism
    research_sources_pro: list[str]    # tool names used by pro researcher
    research_sources_skeptical: list[str]  # tool names used by skeptical researcher

    critique: dict           # {"rubric": {...}, "verdict": "accept"|"revise", "feedback": str}
    synthesis: str
    fact_check: dict         # {"verdict": "pass"|"revise", "claims": [...], "issues": [...]}
    devils_advocate: dict    # {"weaknesses": [...], "verdict": "pass"|"revise"}

    # Both parallel researchers write iteration+1; reducer merges the concurrent updates.
    research_iteration: Annotated[int, _take_max]
    synthesis_iteration: int  # Synthesizer-only writes — no reducer needed
    final_answer: str
