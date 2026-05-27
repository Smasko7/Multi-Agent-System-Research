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

You receive two research briefs that were produced independently — one focused on
supporting evidence and applications, one focused on limitations and
counter-evidence. Internally they were called the "PRO" and "SKEPTICAL" briefs,
but **the reader of your output must NOT see those labels or any other reference
to the internal architecture**. You are producing a finished research answer,
not a stitched-together transcript of two debaters.

Style and structure guidelines:
- Open with a 2-3 sentence direct answer that captures the core idea and signals
  the main tension if one exists.
- Use NATURAL section headers that fit the topic. Good examples: "How it works",
  "Why it matters", "Where it falls short", "Open questions", "Bottom line",
  "Trade-offs", "When to use it", etc. **Never use headers like "PRO Perspective",
  "SKEPTICAL Perspective", "The Argument For/Against", or anything that exposes
  the dual-researcher design.**
- Integrate supporting and skeptical material naturally within the same sections
  where the topic calls for it. Acknowledge tension in prose ("evidence is
  mixed", "critics argue that…") rather than dedicating a separate "tension"
  section.
- Preserve source attribution where the research briefs provided it (e.g.
  "the AutoresearchClaw paper reports…", "according to recent benchmarks…").
- If you are revising, address the specific fact-check issues and
  devil's-advocate weaknesses you receive — but keep the natural structure.
- Close with a takeaway, recommendation, or open question if the topic warrants
  it. No need to force a conclusion section if a clean direct paragraph wraps
  things up."""


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
