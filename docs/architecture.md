# System Architecture

## Overview

Multi-agent LLM research system using LangGraph for orchestration and Gemini for inference.
Four specialized agents collaborate through a directed graph with feedback loops.

## Agent Pipeline

```
[Query] → Researcher → Critic ──(revise, iter<2)──→ Researcher
                           │
                        (accept or iter≥2)
                           ↓
                       Synthesizer → Fact-Checker ──(revise, iter<2)──→ Synthesizer
                                          │
                                       (pass or iter≥2)
                                          ↓
                                     [Final Answer]
```

## Agent Roles

### Researcher (`gemini-2.5-flash`)
- **Tools:** Web search (Tavily → DuckDuckGo fallback), Document retriever (FAISS/RAG), Python REPL
- **Input:** Query + optional Critic feedback from previous iteration
- **Output:** `research_findings` (detailed summary with sources)
- **Tool loop:** Up to 5 rounds of tool calls before finalizing

### Critic (`gemini-3.1-flash-lite`)
- **Tools:** None
- **Input:** Query + research_findings
- **Output:** `critique` — `{"verdict": "accept"|"revise", "feedback": str}`
- **Decision:** Routes to Synthesizer (accept) or back to Researcher (revise, ≤2 times)

### Synthesizer (`gemini-2.5-flash`)
- **Tools:** None
- **Input:** Query + research_findings + optional Fact-Checker issues
- **Output:** `synthesis` + `final_answer` (same content)

### Fact-Checker (`gemini-3.1-flash-lite`)
- **Tools:** Web search (spot-check)
- **Input:** Query + research_findings + synthesis
- **Output:** `fact_check` — `{"verdict": "pass"|"revise", "issues": list[str]}`
- **Decision:** Routes to END (pass) or back to Synthesizer (revise, ≤2 times)

## Key Design Decisions

1. **Prompt-based JSON** for Critic and Fact-Checker instead of `.with_structured_output()` — simpler, more reliable with Gemini.
2. **`MAX_ITERATIONS = 2`** on each feedback loop prevents infinite loops.
3. **`temperature = 1.0`** required for Gemini 2.5+ to avoid degraded quality.
4. **Synchronous execution** — no async complexity.
5. **`time.sleep(2)`** between every LLM call to respect free-tier rate limits.
6. **Graceful RAG degradation** — if `data/corpus/` has no PDFs, RAG tool is omitted silently.

## Shared State

```python
class AgentState(TypedDict):
    query: str                  # Input query (immutable)
    research_findings: str      # Set by Researcher
    critique: dict              # Set by Critic
    synthesis: str              # Set by Synthesizer
    fact_check: dict            # Set by Fact-Checker
    research_iteration: int     # Incremented by Researcher
    synthesis_iteration: int    # Incremented by Synthesizer
    final_answer: str           # Set by Synthesizer (same as synthesis)
```

## File Structure

```
src/
├── config.py          — Constants, model names, rate limits
├── state.py           — AgentState TypedDict
├── graph.py           — LangGraph build_graph()
├── main.py            — CLI entry point
├── agents/
│   ├── researcher.py  — Researcher node with tool calling
│   ├── critic.py      — Critic node with JSON parsing
│   ├── synthesizer.py — Synthesizer node
│   └── fact_checker.py — Fact-Checker node with JSON parsing
├── tools/
│   ├── web_search.py  — Tavily/DuckDuckGo selector
│   ├── rag.py         — FAISS vectorstore + Gemini embeddings
│   └── code_exec.py   — Safe Python sandbox
├── conflict/
│   └── resolution.py  — critic_router, factcheck_router
└── evaluation/
    ├── queries.json   — 8 test queries with expected key facts
    ├── baseline.py    — Single-agent baseline
    └── evaluate.py    — Scoring + comparison table
```
