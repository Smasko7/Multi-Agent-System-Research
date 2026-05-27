"""Devil's Advocate agent — adversarial critique of the synthesized answer.

While the Fact-Checker verifies factual claims, the Devil's Advocate looks at the
synthesis from a contrarian / red-team angle: what's MISSING, what's FRAMED in a
self-serving way, what assumption is unstated, what audience or use case is being
ignored. It runs AFTER the Fact-Checker so it sees a synthesis that has already
passed factual verification — its job is to find structural / framing / coverage
weaknesses, not factual errors.

Runs on Groq Qwen3-32B for model-family independence.
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

DEVILS_ADVOCATE_SYSTEM_PROMPT = """You are the Devil's Advocate agent in a multi-agent research system. Your
role is adversarial: you exist to find what's wrong with the synthesis, not to
agree with it.

The synthesis has already been fact-checked for claim-level correctness — do not
re-do that work. Your job is STRUCTURAL critique: what's the synthesis missing,
mis-framing, or assuming that a sharp, skeptical reader would push back on?

**You should expect to find at least 2-3 meaningful weaknesses in almost any
synthesis.** Even a strong answer has real gaps, hidden assumptions, audiences
it underserves, or counterexamples it doesn't engage with. If you find fewer
than 2, you are probably being too easy on the synthesis — look harder.

Categories to scrutinize (find at least one in most categories):
- **Hidden assumptions** presented as settled fact. What does the answer take
  for granted that a critic might dispute?
- **Counterexamples** or edge cases the synthesis doesn't address.
- **Audiences or contexts** where the answer is actively misleading or wrong.
- **Framing bias**: trade-offs presented as benefits, contested claims presented
  as consensus, or one perspective implicitly favored.
- **Missing scope**: important adjacent topics or follow-up questions left
  unaddressed.
- **Overgeneralization**: claims stated more broadly than the evidence supports.

You MUST respond with EXACTLY this JSON format and nothing else:
{
    "weaknesses": [
        "Specific, concrete weakness 1 (what's missing/misleading and why it matters)",
        "Specific, concrete weakness 2",
        "..."
    ],
    "verdict": "pass" or "revise"
}

Verdict rule: "revise" if you identify ≥2 substantive weaknesses. "pass" only
if the synthesis is genuinely close to airtight (which is rare). Lean toward
revise when in doubt — the system needs your adversarial pushback to produce
better answers."""


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

    # Count-based override: ≥2 substantive weaknesses trips revision regardless
    # of what the LLM chose for verdict. Combined with the harsher prompt that
    # tells the model to *expect* finding 2-3 weaknesses, this should produce
    # revisions on the majority of runs while still allowing genuinely strong
    # syntheses to pass cleanly.
    if len(da["weaknesses"]) >= 2:
        da["verdict"] = "revise"
    else:
        da["verdict"] = "pass"

    logger.info("Devil's Advocate verdict: %s (%d weaknesses)", da["verdict"], len(da["weaknesses"]))
    time.sleep(config.RATE_LIMIT_SLEEP)

    return {"devils_advocate": da}
