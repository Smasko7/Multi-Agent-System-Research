"""Streamlit UI for the multi-agent research system."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from src.graph import build_graph

st.set_page_config(page_title="Multi-Agent Research", layout="wide")
st.title("Multi-Agent Research System")
st.caption(
    "Researcher (PRO) ‖ Researcher (SKEPTICAL) → Critic → Synthesizer → "
    "Fact-Checker → Devil's Advocate"
)

SOURCE_LABELS = {
    "document_retriever": "📚 Local Corpus (RAG)",
    "tavily_search_results_json": "🌐 Web (Tavily)",
    "duckduckgo_search": "🌐 Web (DuckDuckGo)",
    "python_repl": "🐍 Python REPL",
}


def _render_sources(sources: list[str]) -> None:
    if sources:
        unique = list(dict.fromkeys(sources))
        badges = " · ".join(SOURCE_LABELS.get(s, s) for s in unique)
        st.caption(f"**Sources used:** {badges}")
    else:
        st.caption("**Sources used:** _none (model knowledge only)_")


query = st.text_input("Research query:", placeholder="What is retrieval-augmented generation?")
run_btn = st.button("Run", type="primary", disabled=not query)

if run_btn and query:
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

    st.divider()
    final_state = {}

    for step in graph.stream(initial_state):
        for node_name, updates in step.items():
            final_state.update(updates)

            if node_name == "researcher_pro":
                iteration = updates.get("research_iteration", 1)
                label = "Researcher · PRO" if iteration == 1 else f"Researcher · PRO (revision {iteration})"
                with st.expander(f"🟢 {label}", expanded=True):
                    _render_sources(updates.get("research_sources_pro", []))
                    st.markdown(updates.get("research_findings_pro", ""))

            elif node_name == "researcher_skeptical":
                iteration = updates.get("research_iteration", 1)
                label = "Researcher · SKEPTICAL" if iteration == 1 else f"Researcher · SKEPTICAL (revision {iteration})"
                with st.expander(f"🔴 {label}", expanded=True):
                    _render_sources(updates.get("research_sources_skeptical", []))
                    st.markdown(updates.get("research_findings_skeptical", ""))

            elif node_name == "critic":
                critique = updates.get("critique", {})
                verdict = critique.get("verdict", "unknown")
                icon = "✅" if verdict == "accept" else "🔄"
                with st.expander(f"⚖️ Critic — {icon} {verdict.upper()} (Groq Llama-3.3)", expanded=True):
                    rubric = critique.get("rubric", {})
                    if rubric:
                        cols = st.columns(len(rubric))
                        for col, (metric, score) in zip(cols, rubric.items()):
                            col.metric(metric.replace("_", " ").title(), f"{score}/5")
                    feedback = critique.get("feedback", "")
                    if verdict == "accept":
                        st.success(feedback or "All rubric dimensions ≥ 3.")
                    else:
                        st.warning(feedback)

            elif node_name == "synthesizer":
                iteration = updates.get("synthesis_iteration", 1)
                label = "Synthesizer" if iteration == 1 else f"Synthesizer (revision {iteration})"
                with st.expander(f"✍️ {label}", expanded=True):
                    st.markdown(updates.get("synthesis", ""))

            elif node_name == "fact_checker":
                fc = updates.get("fact_check", {})
                verdict = fc.get("verdict", "unknown")
                icon = "✅" if verdict == "pass" else "🔄"
                with st.expander(f"🔎 Fact-Checker — {icon} {verdict.upper()} (Groq Llama-3.3, CoV)", expanded=True):
                    claims = fc.get("claims", [])
                    if claims:
                        st.markdown("**Claims checked:**")
                        for c in claims:
                            if isinstance(c, dict):
                                status = c.get("status", "?")
                                emoji = {"supported": "✅", "verified": "🌐", "unsupported": "❌", "contradicted": "⚠️"}.get(status, "•")
                                st.markdown(f"- {emoji} **{status}** — {c.get('claim', '')}")
                                if c.get("note"):
                                    st.caption(f"  _{c['note']}_")
                    issues = fc.get("issues", [])
                    if issues:
                        st.markdown("**Issues:**")
                        for issue in issues:
                            st.warning(issue)
                    elif verdict == "pass" and not claims:
                        st.success("No factual issues found.")

            elif node_name == "devils_advocate":
                da = updates.get("devils_advocate", {})
                verdict = da.get("verdict", "unknown")
                icon = "✅" if verdict == "pass" else "🔄"
                with st.expander(f"😈 Devil's Advocate — {icon} {verdict.upper()} (Groq Llama-3.3)", expanded=True):
                    weaknesses = da.get("weaknesses", [])
                    if weaknesses:
                        for w in weaknesses:
                            st.warning(w)
                    else:
                        st.success("No structural weaknesses identified.")

    st.divider()
    st.subheader("Final Answer")
    final_answer = final_state.get("final_answer") or final_state.get("synthesis") or "No answer produced."
    st.markdown(final_answer)
