"""LangGraph state graph construction.

Structure:

    START
      ├─→ researcher_pro       ─┐
      └─→ researcher_skeptical ─┴─→ critic ─┬─→ (revise) BOTH researchers (parallel)
                                            └─→ (accept) synthesizer
    synthesizer → fact_checker → devils_advocate ─┬─→ (revise) synthesizer
                                                   └─→ (pass)   END

LangGraph waits for BOTH parallel branches before invoking the downstream node.
A conditional edge that returns a list of targets dispatches to all of them in
parallel (used by `critic_router` when revising).
"""

from langgraph.graph import StateGraph, START, END
from src.state import AgentState
from src.agents.researcher import researcher_pro_node, researcher_skeptical_node
from src.agents.critic import critic_node
from src.agents.synthesizer import synthesizer_node
from src.agents.fact_checker import fact_checker_node
from src.agents.devils_advocate import devils_advocate_node
from src.conflict.resolution import critic_router, post_synthesis_router


def build_graph() -> StateGraph:
    """Build and compile the multi-agent research graph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("researcher_pro", researcher_pro_node)
    workflow.add_node("researcher_skeptical", researcher_skeptical_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("fact_checker", fact_checker_node)
    workflow.add_node("devils_advocate", devils_advocate_node)

    # Parallel fan-out at START: both researchers fire independently
    workflow.add_edge(START, "researcher_pro")
    workflow.add_edge(START, "researcher_skeptical")

    # Both researchers feed into the critic (LangGraph joins them automatically)
    workflow.add_edge("researcher_pro", "critic")
    workflow.add_edge("researcher_skeptical", "critic")

    # Critic: revise → re-fan-out to BOTH researchers in parallel; accept → synthesizer
    workflow.add_conditional_edges("critic", critic_router, {
        "researcher_pro": "researcher_pro",
        "researcher_skeptical": "researcher_skeptical",
        "synthesizer": "synthesizer",
    })

    workflow.add_edge("synthesizer", "fact_checker")
    workflow.add_edge("fact_checker", "devils_advocate")
    workflow.add_conditional_edges("devils_advocate", post_synthesis_router, {
        "synthesizer": "synthesizer",
        "end": END,
    })

    return workflow.compile()
