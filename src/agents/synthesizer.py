"""Synthesizer agent — mediates between PRO and SKEPTICAL research outputs.

This is no longer a re-stater of a single research output; it has to actually
reconcile two viewpoints into a balanced answer.
"""

import time
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from src import config
from src.state import AgentState
from src.agents._common import extract_text

logger = logging.getLogger(__name__)

SYNTHESIZER_SYSTEM_PROMPT = """You are the Synthesizer agent in a multi-agent research system.

You receive TWO research outputs that were produced independently with different
angles:
- PRO findings: evidence supporting / explaining the topic
- SKEPTICAL findings: limitations, counter-evidence, controversies

Your task: produce a single, balanced answer that integrates BOTH perspectives.
You are a mediator, not a re-stater.

Requirements:
- Open with a clear, direct answer to the query.
- Cover the substantive supporting evidence (from PRO).
- Cover the substantive limitations and counter-evidence (from SKEPTICAL).
- Where the two perspectives conflict, explicitly acknowledge the tension rather
  than picking one side silently.
- Preserve source attribution from both researchers where present.
- Address any fact-check issues or devil's-advocate weaknesses if this is a revision."""


def synthesizer_node(state: AgentState) -> dict:
    """Mediate between pro/skeptical research and produce a balanced answer."""
    logger.info("Synthesizer running (iteration %d)", state.get("synthesis_iteration", 0))

    llm = ChatGoogleGenerativeAI(
        model=config.GENERATION_MODEL,
        temperature=config.MODEL_TEMPERATURE,
    )

    fact_check = state.get("fact_check", {})
    fc_issues = fact_check.get("issues", []) if fact_check.get("verdict") == "revise" else []

    devils = state.get("devils_advocate", {})
    da_weaknesses = devils.get("weaknesses", []) if devils.get("verdict") == "revise" else []

    human_parts = [
        f"Original query: {state['query']}",
        f"\n=== PRO research findings ===\n{state.get('research_findings_pro', '')}",
        f"\n=== SKEPTICAL research findings ===\n{state.get('research_findings_skeptical', '')}",
    ]
    if fc_issues:
        items = "\n".join(f"- {i}" for i in fc_issues)
        human_parts.append(f"\nFact-check issues to fix:\n{items}")
    if da_weaknesses:
        items = "\n".join(f"- {w}" for w in da_weaknesses)
        human_parts.append(f"\nDevil's-advocate weaknesses to address:\n{items}")
    if state.get("synthesis") and (fc_issues or da_weaknesses):
        human_parts.append(f"\nPrevious synthesis (to revise):\n{state['synthesis']}")

    messages = [
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
        HumanMessage(content="\n".join(human_parts)),
    ]

    response = llm.invoke(messages)
    synthesis = extract_text(response.content) or "No synthesis produced."

    logger.info("Synthesizer produced answer (%d chars)", len(synthesis))
    time.sleep(config.RATE_LIMIT_SLEEP)

    return {
        "synthesis": synthesis,
        "final_answer": synthesis,
        "synthesis_iteration": state.get("synthesis_iteration", 0) + 1,
    }
