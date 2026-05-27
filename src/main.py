"""Entry point for the multi-agent research system."""

import sys
import logging
from src.graph import build_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("main")


SOURCE_LABELS = {
    "document_retriever": "📚 Local Corpus (RAG)",
    "tavily_search_results_json": "🌐 Web (Tavily)",
    "duckduckgo_search": "🌐 Web (DuckDuckGo)",
    "python_repl": "🐍 Python REPL",
}


def _format_sources(sources: list[str]) -> str:
    """Dedupe and label a tool-name list for display."""
    unique = list(dict.fromkeys(sources or []))
    if not unique:
        return "none (model knowledge only)"
    return " · ".join(SOURCE_LABELS.get(s, s) for s in unique)


def run(query: str) -> dict:
    """Run the multi-agent pipeline on a query and return the final state."""
    logger.info(f"Query: {query}")
    graph = build_graph()

    initial_state = {
        "query": query,
        "research_findings_pro": "",
        "research_findings_skeptical": "",
        "research_sources_pro": [],
        "research_sources_skeptical": [],
        "critique": {},
        "synthesis": "",
        "fact_check": {},
        "devils_advocate": {},
        "research_iteration": 0,
        "synthesis_iteration": 0,
        "final_answer": "",
    }

    return graph.invoke(initial_state)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.main \"Your research query here\"")
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    result = run(query)

    pro_sources = _format_sources(result.get("research_sources_pro", []))
    skep_sources = _format_sources(result.get("research_sources_skeptical", []))
    answer = result.get("final_answer") or result.get("synthesis") or "No answer produced."

    print("\n" + "=" * 60)
    print("SOURCES USED")
    print("=" * 60)
    print(f"🟢 Researcher · PRO       — {pro_sources}")
    print(f"🔴 Researcher · SKEPTICAL — {skep_sources}")

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()
