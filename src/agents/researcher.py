"""Researcher agents — parallel fan-out with PRO and SKEPTICAL angles.

Both agents share the same tool stack and code path; only the angle in the system
prompt differs. Each writes to its own state fields so the parallel branches don't
clobber each other.
"""

import time
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src import config
from src.state import AgentState
from src.agents._common import extract_text
from src.tools.web_search import get_search_tool
from src.tools.rag import get_retriever_tool
from src.tools.code_exec import get_code_exec_tool

logger = logging.getLogger(__name__)

BASE_RESEARCHER_PROMPT = """You are a Researcher agent in a multi-agent research system.

Your task: Given a query, gather comprehensive, relevant information using your available tools.

Tool selection priority:
1. If a `document_retriever` tool is available, read its description for the list of
documents in the local corpus. When the query references any of those documents or their
topics, CALL `document_retriever` FIRST — it is your primary source for corpus-covered topics.
2. Use web search to corroborate, fill gaps, or for topics not covered by the corpus.
3. Use `python_repl` for calculations or data analysis.

If you have received feedback from the Critic, address their specific concerns in your revised research.

Return your findings as a detailed, well-organized summary with source attribution where possible.
Focus on factual claims that can be verified.

---
YOUR ANGLE: {angle_description}
"""

ANGLES = {
    "pro": (
        "You are the PRO researcher. Focus on finding evidence, mechanisms, applications, "
        "and arguments that SUPPORT or explain the topic in the query. Surface concrete "
        "examples, established findings, and reasons something works. You are not blindly "
        "promotional — stick to verifiable facts — but your lens is constructive and "
        "affirmative. Do not spend time enumerating limitations; the SKEPTICAL researcher "
        "covers that."
    ),
    "skeptical": (
        "You are the SKEPTICAL researcher. Focus on finding limitations, counter-evidence, "
        "failure modes, controversies, open problems, and reasons the topic might be "
        "misunderstood or overhyped. Surface dissenting views, methodological weaknesses, "
        "and edge cases. You are not contrarian for its own sake — stick to verifiable "
        "claims — but your lens is critical and probing. Do not spend time enumerating "
        "strengths; the PRO researcher covers that."
    ),
}


def _run_researcher(state: AgentState, angle: str) -> dict:
    """Shared researcher logic, parameterized by angle."""
    findings_key = f"research_findings_{angle}"
    sources_key = f"research_sources_{angle}"

    logger.info(
        "Researcher[%s] running (iteration %d)",
        angle, state.get("research_iteration", 0)
    )

    search_tool = get_search_tool()
    retriever_tool = get_retriever_tool()
    code_tool = get_code_exec_tool()

    tools = [search_tool, code_tool]
    if retriever_tool is not None:
        tools.append(retriever_tool)
    tools_by_name = {t.name: t for t in tools}

    llm = ChatGoogleGenerativeAI(
        model=config.GENERATION_MODEL,
        temperature=config.MODEL_TEMPERATURE,
    )
    llm_with_tools = llm.bind_tools(tools)

    critique = state.get("critique", {})
    feedback = critique.get("feedback", "") if critique.get("verdict") == "revise" else ""

    human_parts = [f"Query: {state['query']}"]
    if state.get(findings_key):
        human_parts.append(f"\nPrevious findings (to improve upon):\n{state[findings_key]}")
    if feedback:
        human_parts.append(f"\nFeedback from Critic (please address these issues):\n{feedback}")

    system_prompt = BASE_RESEARCHER_PROMPT.format(angle_description=ANGLES[angle])
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="\n".join(human_parts)),
    ]

    response = None
    sources: list[str] = []
    for _ in range(5):
        response = llm_with_tools.invoke(messages)
        if not response.tool_calls:
            break
        messages.append(response)
        for tc in response.tool_calls:
            sources.append(tc["name"])
            tool = tools_by_name.get(tc["name"])
            if tool:
                try:
                    result = tool.invoke(tc["args"])
                except Exception as exc:
                    result = f"Tool error: {exc}"
            else:
                result = f"Unknown tool: {tc['name']}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    findings = extract_text(response.content if response else "") or "No findings produced."

    logger.info("Researcher[%s] used tools: %s", angle, sources or "[none]")
    time.sleep(config.RATE_LIMIT_SLEEP)

    return {
        findings_key: findings,
        sources_key: sources,
        # Both branches increment, but LangGraph's last-write-wins on simple int fields
        # means the higher value is what we see downstream; that's fine for cap logic.
        "research_iteration": state.get("research_iteration", 0) + 1,
    }


def researcher_pro_node(state: AgentState) -> dict:
    return _run_researcher(state, "pro")


def researcher_skeptical_node(state: AgentState) -> dict:
    return _run_researcher(state, "skeptical")
