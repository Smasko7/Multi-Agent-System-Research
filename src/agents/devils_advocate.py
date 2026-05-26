"""Devil's Advocate agent — adversarial critique of the synthesized answer.

While the Fact-Checker verifies factual claims, the Devil's Advocate looks at the
synthesis from a contrarian / red-team angle: what's MISSING, what's FRAMED in a
self-serving way, what assumption is unstated, what audience or use case is being
ignored. It runs AFTER the Fact-Checker so it sees a synthesis that has already
passed factual verification — its job is to find structural / framing / coverage
weaknesses, not factual errors.

Runs on Groq Llama-3.3-70B for model-family independence.
"""

import json
import re
import time
import logging

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src import config
from src.state import AgentState
from src.agents._common import extract_text

logger = logging.getLogger(__name__)

DEVILS_ADVOCATE_SYSTEM_PROMPT = """You are the Devil's Advocate agent in a multi-agent research system.

The synthesis you are reviewing has ALREADY been fact-checked for claim-level
correctness. Do not re-do that work. Your job is structural and adversarial:
find what a sharp critic would attack about this answer if they wanted to weaken
its persuasiveness or expose what it's missing.

Look for, at minimum:
- Hidden assumptions presented as fact
- Important counterexamples or edge cases the synthesis ignores
- Audiences or use cases where this answer would mislead
- Trade-offs that are stated as benefits (or vice versa)
- Framing choices that subtly favor one perspective
- Topics adjacent to the query that a reader would want addressed and aren't

You MUST respond with EXACTLY this JSON format and nothing else:
{
    "weaknesses": [
        "Specific, concrete weakness 1 (what's missing or misleading, and why it matters)",
        "Specific, concrete weakness 2",
        "..."
    ],
    "verdict": "pass" or "revise"
}

Verdict rule: "revise" if you identify ≥2 SUBSTANTIVE weaknesses (not nitpicks).
"pass" only if the synthesis is genuinely well-rounded. Be hard-nosed — the system
relies on you to surface what the other agents would prefer to ignore. Each
weakness must be specific enough that the Synthesizer knows exactly what to add
or rephrase."""


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

    logger.warning("Devil's Advocate JSON parse failed; using default. Raw: %s", content[:200])
    return default


def devils_advocate_node(state: AgentState) -> dict:
    """Red-team the synthesis for structural / framing / coverage weaknesses."""
    logger.info("Devil's Advocate red-teaming synthesis")

    llm = ChatGroq(
        model=config.JUDGMENT_MODEL,
        temperature=config.MODEL_TEMPERATURE,
    )

    messages = [
        SystemMessage(content=DEVILS_ADVOCATE_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Original query: {state['query']}\n\n"
                f"Synthesized answer to red-team:\n{state.get('synthesis', '')}"
            )
        ),
    ]

    response = llm.invoke(messages)
    content = extract_text(response.content)

    default = {"weaknesses": [], "verdict": "pass"}
    da = _parse_json(content, default)

    da.setdefault("weaknesses", [])
    da.setdefault("verdict", "pass")

    # Enforce: revise if 2+ substantive weaknesses
    if len(da["weaknesses"]) >= 2:
        da["verdict"] = "revise"
    else:
        da["verdict"] = "pass"

    logger.info("Devil's Advocate verdict: %s (%d weaknesses)", da["verdict"], len(da["weaknesses"]))
    time.sleep(config.RATE_LIMIT_SLEEP)

    return {"devils_advocate": da}
