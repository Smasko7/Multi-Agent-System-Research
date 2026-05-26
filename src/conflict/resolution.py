"""Conditional edge functions for the feedback loops (conflict gates)."""

from src.config import MAX_ITERATIONS
from src.state import AgentState


def critic_router(state: AgentState) -> list[str] | str:
    """Route after Critic.

    If Critic says revise AND we have iteration budget, fan-out back to BOTH
    researchers in parallel (returning a list triggers parallel branches in
    LangGraph). Otherwise advance to the Synthesizer.

    Maps to Contract Net Protocol: the proposal (combined PRO+SKEPTICAL research)
    is evaluated by the Critic. Rejection re-announces the task to the entire
    researcher cohort with structured feedback.
    """
    critique = state.get("critique", {})
    if critique.get("verdict") == "revise" and state.get("research_iteration", 0) < MAX_ITERATIONS:
        return ["researcher_pro", "researcher_skeptical"]
    return "synthesizer"


def post_synthesis_router(state: AgentState) -> str:
    """Route after Devil's Advocate (which runs last in the synthesis loop).

    Re-runs the Synthesizer if EITHER the Fact-Checker OR the Devil's Advocate
    flagged issues, subject to the iteration cap. This is FIPA-style aggregation
    of multiple `reject-proposal` messages from independent reviewers.
    """
    fact_check = state.get("fact_check", {})
    devils = state.get("devils_advocate", {})

    needs_fc_revise = fact_check.get("verdict") == "revise"
    needs_da_revise = devils.get("verdict") == "revise"

    if (needs_fc_revise or needs_da_revise) and state.get("synthesis_iteration", 0) < MAX_ITERATIONS:
        return "synthesizer"
    return "end"
