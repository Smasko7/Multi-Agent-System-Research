"""Fact-Checker agent — Chain-of-Verification (CoV) over the synthesized answer.

Process:
1. Extract atomic claims from the synthesis (3-7 of them).
2. For each claim, check it against the research findings AND against web search.
3. Flag claims that are unsupported, contradicted, or hallucinated.

Runs on Groq Qwen3-32B (different model family from generation agents) and is
REQUIRED to invoke web search at least once before issuing a verdict.
"""

import json
import re
import time
import logging

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src import config
from src.state import AgentState
from src.agents._common import extract_text
from src.tools.web_search import get_search_tool

logger = logging.getLogger(__name__)

FACT_CHECKER_SYSTEM_PROMPT = """You are the Fact-Checker agent in a multi-agent research system.

You receive a synthesized answer plus the two research outputs (PRO and SKEPTICAL)
that produced it. Your job: rigorously verify the answer using Chain-of-Verification.

Process you MUST follow:
1. Extract 3-7 ATOMIC, VERIFIABLE claims from the synthesis (specific facts, numbers,
   relationships, named entities — not vague generalizations).
2. For each claim, determine if it is supported by the PRO/SKEPTICAL findings.
3. For at least 2 critical or surprising claims, INVOKE `web_search` to corroborate
   them independently. Skipping this step is a failure of your role.
4. Mark each claim as "supported", "unsupported", "contradicted", or "verified" (web).

You MUST respond with EXACTLY this JSON format and nothing else:
{
    "claims": [
        {"claim": "...", "status": "supported|unsupported|contradicted|verified", "note": "..."}
    ],
    "verdict": "pass" or "revise",
    "issues": ["specific issue 1", "specific issue 2"]
}

Verdict rule: "revise" ONLY if at least one claim is "contradicted" (actively
disagreed with by research or web). Claims that are merely "unsupported" (no
source found but not contradicted) should be listed in issues for the Synthesizer's
awareness, but do NOT by themselves require revision — missing attribution is too
common to block on. "pass" when no claim is contradicted, even if some are unsupported.
Issues list must give the Synthesizer concrete fixes (e.g. "Drop the claim that X
is N% accurate — no source in research or web supports this")."""


def _parse_json(content, default: dict) -> dict:
    content = extract_text(content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning("Fact-Checker JSON parse failed; using default. Raw: %s", content[:200])
    return default


def fact_checker_node(state: AgentState) -> dict:
    """Verify synthesis against research findings + web. Chain-of-verification."""
    logger.info("Fact-Checker running CoV on synthesis")

    search_tool = get_search_tool()
    tools = [search_tool]
    tools_by_name = {t.name: t for t in tools}

    llm = ChatGroq(
        model=config.JUDGMENT_MODEL,
        temperature=config.MODEL_TEMPERATURE,
    )
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=FACT_CHECKER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Original query: {state['query']}\n\n"
                f"=== PRO research ===\n{state.get('research_findings_pro', '')}\n\n"
                f"=== SKEPTICAL research ===\n{state.get('research_findings_skeptical', '')}\n\n"
                f"=== Synthesis to verify ===\n{state.get('synthesis', '')}"
            )
        ),
    ]

    response = None
    for _ in range(4):
        response = llm_with_tools.invoke(messages)
        if not getattr(response, "tool_calls", None):
            break
        messages.append(response)
        for tc in response.tool_calls:
            tool = tools_by_name.get(tc["name"])
            if tool:
                try:
                    result = tool.invoke(tc["args"])
                except Exception as exc:
                    result = f"Tool error: {exc}"
            else:
                result = f"Unknown tool: {tc['name']}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    content = extract_text(response.content if response else "")
    default = {"claims": [], "verdict": "pass", "issues": []}
    fact_check = _parse_json(content, default)

    fact_check.setdefault("verdict", "pass")
    fact_check.setdefault("issues", [])
    fact_check.setdefault("claims", [])

    # Enforce: revise ONLY on active contradictions (factual errors).
    # `unsupported` claims are noted as issues but don't auto-trigger revision —
    # missing attribution alone is too noisy a trigger. The LLM's own verdict still
    # stands for borderline cases.
    if any(c.get("status") == "contradicted" for c in fact_check["claims"] if isinstance(c, dict)):
        fact_check["verdict"] = "revise"

    logger.info(
        "Fact-Checker verdict: %s (%d claims checked, %d issues)",
        fact_check["verdict"], len(fact_check["claims"]), len(fact_check["issues"]),
    )
    time.sleep(config.RATE_LIMIT_SLEEP)

    return {"fact_check": fact_check}
