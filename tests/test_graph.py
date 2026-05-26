"""Tests for graph structure and routing logic."""

from src.graph import build_graph
from src.conflict.resolution import critic_router, post_synthesis_router


def test_graph_compiles():
    """The graph should compile without errors."""
    graph = build_graph()
    assert graph is not None


# --- critic_router ---

def test_critic_router_accept():
    state = {"critique": {"verdict": "accept", "feedback": ""}, "research_iteration": 0}
    assert critic_router(state) == "synthesizer"


def test_critic_router_revise_fans_out():
    """Revise should fan out to BOTH researchers in parallel."""
    state = {"critique": {"verdict": "revise", "feedback": "Need depth"}, "research_iteration": 0}
    result = critic_router(state)
    assert isinstance(result, list)
    assert set(result) == {"researcher_pro", "researcher_skeptical"}


def test_critic_router_max_iterations():
    """At the iteration cap, advance to synthesizer regardless of verdict."""
    state = {"critique": {"verdict": "revise", "feedback": "Still weak"}, "research_iteration": 2}
    assert critic_router(state) == "synthesizer"


# --- post_synthesis_router ---

def test_post_synthesis_router_both_pass():
    state = {
        "fact_check": {"verdict": "pass", "issues": []},
        "devils_advocate": {"verdict": "pass", "weaknesses": []},
        "synthesis_iteration": 0,
    }
    assert post_synthesis_router(state) == "end"


def test_post_synthesis_router_fc_revise():
    state = {
        "fact_check": {"verdict": "revise", "issues": ["Bad claim"]},
        "devils_advocate": {"verdict": "pass", "weaknesses": []},
        "synthesis_iteration": 0,
    }
    assert post_synthesis_router(state) == "synthesizer"


def test_post_synthesis_router_da_revise():
    """Devil's Advocate alone can trigger revision."""
    state = {
        "fact_check": {"verdict": "pass", "issues": []},
        "devils_advocate": {"verdict": "revise", "weaknesses": ["Missing angle X", "Hidden assumption Y"]},
        "synthesis_iteration": 0,
    }
    assert post_synthesis_router(state) == "synthesizer"


def test_post_synthesis_router_max_iterations():
    """Iteration cap ends the loop even if both reviewers want revision."""
    state = {
        "fact_check": {"verdict": "revise", "issues": ["X"]},
        "devils_advocate": {"verdict": "revise", "weaknesses": ["Y", "Z"]},
        "synthesis_iteration": 2,
    }
    assert post_synthesis_router(state) == "end"
