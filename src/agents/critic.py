"""Critic agent — evaluates BOTH researchers' findings using a structured rubric.

Runs on a different model family (Groq Qwen3-32B) than the researchers (Gemini),
so it has genuinely independent training biases. Has its own RAG tool so it can
spot-check claims against the local corpus instead of just trusting the researchers.
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
from src.tools.rag import get_retriever_tool

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = """You are the Critic agent in a multi-agent research system.

You receive TWO research outputs:
- PRO findings: evidence supporting / explaining the query topic
- SKEPTICAL findings: limitations, counter-evidence, controversies

Your task: Evaluate the COMBINED research using a structured rubric. Score each
dimension from 1 (poor) to 5 (excellent):

- **relevance**: Do the findings address the original query directly?
- **completeness**: Are both perspectives substantively developed? Are there obvious gaps?
- **source_quality**: Are claims attributed to identifiable sources (papers, web, corpus)?
- **balance**: Is the pro/skeptical split substantive, or is one side thin/missing?

If you have access to `document_retriever`, USE IT to independently verify at least
one specific claim from each researcher against the local corpus. Do not skip this
step when the tool is available.

You MUST respond with EXACTLY this JSON format and nothing else:
{
    "rubric": {
        "relevance": <1-5>,
        "completeness": <1-5>,
        "source_quality": <1-5>,
        "balance": <1-5>
    },
    "verdict": "accept" or "revise",
    "feedback": "Specific, actionable feedback referencing exact claims or gaps. \
If revise: tell the researchers EXACTLY what to fix."
}

Verdict rule (NON-NEGOTIABLE): set verdict to "revise" if ANY rubric score is
strictly less than 3. Otherwise "accept". Do not soften this — your value to the
system comes from catching weak research, not from agreeing."""


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

    logger.warning("Critic JSON parse failed; using default. Raw: %s", content[:200])
    return default


def critic_node(state: AgentState) -> dict:
    """Evaluate both researchers' findings using a structured rubric."""
    logger.info("Critic evaluating PRO + SKEPTICAL research findings")

    retriever_tool = get_retriever_tool()
    tools = [retriever_tool] if retriever_tool is not None else []
    tools_by_name = {t.name: t for t in tools}

    llm = ChatGroq(
        model=config.JUDGMENT_MODEL,
        temperature=config.MODEL_TEMPERATURE,
    )
    if tools:
        llm = llm.bind_tools(tools)

    pro = state.get("research_findings_pro", "")
    skeptical = state.get("research_findings_skeptical", "")

    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Original query: {state['query']}\n\n"
                f"=== PRO findings ===\n{pro}\n\n"
                f"=== SKEPTICAL findings ===\n{skeptical}"
            )
        ),
    ]

    response = None
    for _ in range(3):
        response = llm.invoke(messages)
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

    default = {
        "rubric": {"relevance": 3, "completeness": 3, "source_quality": 3, "balance": 3},
        "verdict": "accept",
        "feedback": "Parse error — accepting by default.",
    }
    critique = _parse_json(response.content if response else "", default)

    critique.setdefault("verdict", "accept")
    critique.setdefault("feedback", "")
    rubric = critique.setdefault("rubric", default["rubric"])

    # Enforce the verdict rule even if the LLM ignored it
    if any(int(v) < 3 for v in rubric.values() if isinstance(v, (int, float, str)) and str(v).isdigit()):
        critique["verdict"] = "revise"

    logger.info("Critic rubric: %s | verdict: %s", critique.get("rubric"), critique["verdict"])
    time.sleep(config.RATE_LIMIT_SLEEP)

    return {"critique": critique}
