"""Single-agent baseline for evaluation comparison."""

import time
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src import config
from src.tools.web_search import get_search_tool
from src.tools.rag import get_retriever_tool
from src.tools.code_exec import get_code_exec_tool

logger = logging.getLogger(__name__)

BASELINE_SYSTEM_PROMPT = """Answer the following query as thoroughly and accurately as possible.
Use the available tools (web search, document retrieval, code execution) as needed.
Provide a well-structured answer with source attribution."""


def run_baseline(query: str) -> str:
    """Run single-agent baseline on a query and return the answer."""
    logger.info("Baseline running for: %s", query)

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

    messages = [
        SystemMessage(content=BASELINE_SYSTEM_PROMPT),
        HumanMessage(content=f"Query: {query}"),
    ]

    response = None
    for _ in range(5):
        response = llm_with_tools.invoke(messages)
        if not response.tool_calls:
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

    time.sleep(config.RATE_LIMIT_SLEEP)
    return (response.content if response else "") or "No answer produced."
