"""Web search tool — Tavily primary, DuckDuckGo fallback."""

import os


def get_search_tool():
    """Return Tavily if API key is set, else DuckDuckGo."""
    if os.getenv("TAVILY_API_KEY"):
        from langchain_community.tools.tavily_search import TavilySearchResults
        return TavilySearchResults(max_results=3)
    else:
        from langchain_community.tools import DuckDuckGoSearchRun
        return DuckDuckGoSearchRun()
