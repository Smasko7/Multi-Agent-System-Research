# System Architecture

## Overview

Multi-agent LLM research system using **LangGraph** for orchestration. **Six**
specialized agents collaborate through a directed graph with two feedback loops,
parallel fan-out, and cross-model-family judgment.

The system intentionally separates **generation** (Gemini Flash-Lite) from
**judgment** (Groq Qwen3-32B) so the judges have independent training biases
from the agents they review.

---

## Agent Pipeline

```
                          [Query]
                         /        \
                        v          v
        researcher_pro     researcher_skeptical      Parallel fan-out
                        \          /                  (CNP task announcement)
                         v        v
                          critic                      Rubric judge (Groq Qwen3)
                       /         \
              revise   |           |  accept
              (iter<2) |           |
              fan-out  v           v
              to BOTH  ↳ researchers   synthesizer    Mediator (Gemini)
                                          |
                                          v
                                     fact_checker     Chain-of-Verification (Groq Qwen3)
                                          |
                                          v
                                    devils_advocate   Adversarial red-team (Groq Qwen3)
                                       /          \
                                revise |            | pass
                              (iter<2) v            v
                                  synthesizer      [Final Answer]
```

Both feedback loops are capped at `MAX_ITERATIONS = 2`. The critic's "revise"
branch returns a **list of two node names**, which LangGraph dispatches in
parallel — both researchers re-fire simultaneously with the critic's feedback.

---

## Agent Roles

### 🟢 Researcher · PRO (`gemini-3.1-flash-lite`)
- **Tools:** Web search (Tavily → DuckDuckGo fallback), document retriever (FAISS/RAG), Python REPL
- **Angle:** Find supporting evidence, mechanisms, applications, established findings
- **Input:** Query + optional Critic feedback from previous iteration
- **Output:** `research_findings_pro` (detailed summary with sources) + `research_sources_pro` (list of tool-names invoked)
- **Tool loop:** Up to 5 rounds of tool calls before finalizing

### 🔴 Researcher · SKEPTICAL (`gemini-3.1-flash-lite`)
- **Tools:** Identical to PRO
- **Angle:** Find limitations, counter-evidence, failure modes, controversies, open problems
- **Output:** `research_findings_skeptical` + `research_sources_skeptical`
- Runs **in parallel** with the PRO researcher (LangGraph parallel branches from `START`)

### ⚖️ Critic (`qwen/qwen3-32b` via Groq)
- **Tools:** Document retriever (independent verification against the local corpus)
- **Input:** Query + both researchers' findings
- **Output:** `critique` — `{rubric: {relevance, completeness, source_quality, balance}, verdict, feedback}`
- **Verdict rule:** Hard `revise` if any rubric dimension < 3 (enforced in code after parsing)
- **Routes:** revise → fan out to BOTH researchers; accept → Synthesizer

### ✍️ Synthesizer (`gemini-3.1-flash-lite`)
- **Tools:** None
- **Input:** Query + both researchers' findings + (optional) fact-check issues + (optional) devil's-advocate weaknesses
- **Role:** Mediator. Integrates PRO and SKEPTICAL viewpoints into one balanced answer using natural section headers (NOT "PRO Perspective" / "SKEPTICAL Perspective" labels — that would leak architecture).
- **Output:** `synthesis` + `final_answer`

### 🔎 Fact-Checker (`qwen/qwen3-32b` via Groq)
- **Tools:** Web search (mandatory — prompt enforces ≥1 invocation)
- **Process:** Chain-of-Verification — extracts 3-7 atomic claims, marks each `supported` / `verified` / `unsupported` / `contradicted`
- **Output:** `fact_check` — `{claims: [...], verdict, issues: [...]}`
- **Verdict rule:** Hard `revise` only if any claim is `contradicted` (unsupported claims are listed for transparency but don't block — missing attribution is too noisy a trigger)

### 😈 Devil's Advocate (`qwen/qwen3-32b` via Groq)
- **Tools:** None
- **Role:** Adversarial red-team. Identifies structural weaknesses (hidden assumptions, ignored audiences, framing bias, missing scope, overgeneralization) post fact-check
- **Output:** `devils_advocate` — `{weaknesses: [...], verdict}`
- **Verdict rule:** Hard `revise` if ≥2 substantive weaknesses (count-based override)
- **Routes:** revise → Synthesizer (revision loop); pass → END

---

## Key Design Decisions

1. **Parallel fan-out at START** — both researchers fire simultaneously via two `add_edge(START, ...)` calls; LangGraph joins them at the critic. Critic's revise branch returns `["researcher_pro", "researcher_skeptical"]` to re-fan-out.
2. **Model-family diversity** — Critic / Fact-Checker / Devil's Advocate run on Groq Qwen3-32B instead of Gemini so they have genuinely different training biases. Avoids the "Gemini-judging-Gemini" consensus blind spot.
3. **`extract_text` strips `<think>...</think>` blocks** — Qwen3 is a reasoning model that emits its thinking trace inline; the helper sanitizes output before JSON parsing.
4. **Prompt-based JSON** for the three judges (no `.with_structured_output()`) — tolerant `try direct → strip ```fences``` → first `{...}`` block` fallback parser handles both Gemini and Qwen reliably.
5. **Override rules enforce judge calibration** — code-level overrides after LLM parsing ensure: Critic revise on any rubric<3; Fact-Checker revise on any `contradicted` claim; Devil's Advocate revise on ≥2 weaknesses. These backstop the LLM in case prompts are ignored.
6. **`MAX_ITERATIONS = 2`** per feedback loop. Hard cap to prevent runaway revisions.
7. **`temperature = 1.0`** for all Gemini calls (Gemini 2.5+ degrades below 1.0).
8. **`time.sleep(2)` between LLM calls** to respect free-tier rate limits.
9. **Synchronous execution** — no async. Simpler debug, clearer state semantics.
10. **Local embeddings (sentence-transformers)** instead of cloud — sidesteps Windows TLS interception that was breaking the Gemini embeddings endpoint.
11. **`truststore.inject_into_ssl()` in `src/__init__.py`** — patches Python SSL to use the OS cert store so HuggingFace model downloads succeed on Windows with HTTPS-scanning AV.
12. **Graceful RAG degradation** — if `data/corpus/` has no PDFs, the retriever tool is omitted from the researchers' toolset silently.

---

## Shared State

```python
class AgentState(TypedDict):
    query: str

    # Parallel researchers — separate fields so they don't clobber each other
    research_findings_pro: str
    research_findings_skeptical: str
    research_sources_pro: list[str]
    research_sources_skeptical: list[str]

    critique: dict                    # {rubric, verdict, feedback}
    synthesis: str
    fact_check: dict                  # {claims, verdict, issues}
    devils_advocate: dict             # {weaknesses, verdict}

    # Reducer required: both parallel researchers write iteration+1
    # in the same step → InvalidUpdateError without `max` merging.
    research_iteration: Annotated[int, _take_max]
    synthesis_iteration: int          # Synthesizer-only writes — no reducer
    final_answer: str
```

The `_take_max` reducer is a wrapped `max(a, b)` (LangGraph cannot introspect
the builtin `max` directly). Both researchers write `iteration + 1` from the
same source value, so max gives a stable join.

---

## File Structure

```
src/
├── __init__.py            — truststore.inject_into_ssl() on import
├── config.py              — Model IDs, rate limits, RAG settings
├── state.py               — AgentState + _take_max reducer
├── graph.py               — LangGraph build_graph()
├── main.py                — CLI entry point, prints sources + final answer
├── app.py                 — Streamlit UI (arcade-themed)
├── arcade.py              — Pygame visual GUI
├── visualize.py           — Mermaid diagram of graph or last LangSmith run
├── agents/
│   ├── _common.py         — extract_text helper (strips <think> blocks)
│   ├── researcher.py      — Parameterized researcher; pro + skeptical wrappers
│   ├── critic.py          — Rubric judge with RAG tool
│   ├── synthesizer.py     — Mediator (natural section headers)
│   ├── fact_checker.py    — Chain-of-Verification with mandatory web search
│   └── devils_advocate.py — Red-team
├── tools/
│   ├── web_search.py      — Tavily/DuckDuckGo selector
│   ├── rag.py             — FAISS + sentence-transformers (local embeddings)
│   └── code_exec.py       — Safe Python sandbox
├── conflict/
│   └── resolution.py      — critic_router (fan-out list) + post_synthesis_router
└── evaluation/
    ├── queries.json       — 8 test queries with expected key facts
    ├── baseline.py        — Single-agent baseline
    └── evaluate.py        — Scoring + comparison table
```
