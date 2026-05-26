"""Entry point for the multi-agent research system."""

import sys
import logging
from src.graph import build_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("main")


def run(query: str) -> str:
    """Run the multi-agent pipeline on a query and return the final answer."""
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

    result = graph.invoke(initial_state)
    return result.get("final_answer", result.get("synthesis", "No answer produced."))


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.main \"Your research query here\"")
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    answer = run(query)
    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()
