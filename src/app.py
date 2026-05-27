"""Streamlit UI for the multi-agent research system — arcade-themed edition."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from src.graph import build_graph

st.set_page_config(
    page_title="Multi-Agent Research",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }

/* Page background */
.stApp {
    background:
        radial-gradient(at 20% 0%, rgba(99, 102, 241, 0.08) 0px, transparent 50%),
        radial-gradient(at 80% 100%, rgba(255, 215, 0, 0.06) 0px, transparent 50%),
        linear-gradient(135deg, #0a0a14 0%, #15131f 50%, #0a0a14 100%);
}
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 4rem;
    max-width: 1400px;
}

/* Header */
.header-container {
    background: linear-gradient(90deg, #1e1b4b 0%, #312e81 50%, #1e1b4b 100%);
    padding: 1.6rem 2rem;
    border-radius: 14px;
    border: 2px solid #ffd700;
    margin-bottom: 1.5rem;
    box-shadow: 0 0 50px rgba(255, 215, 0, 0.15), inset 0 0 30px rgba(99, 102, 241, 0.1);
    position: relative;
    overflow: hidden;
}
.header-container::before {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(45deg, transparent, transparent 30px, rgba(255, 215, 0, 0.03) 30px, rgba(255, 215, 0, 0.03) 31px);
    pointer-events: none;
}
.header-title {
    font-family: 'Orbitron', sans-serif;
    font-weight: 900;
    font-size: 2.4rem;
    letter-spacing: 0.08em;
    background: linear-gradient(90deg, #ffd700, #fbbf24, #ffd700);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    text-align: center;
    text-shadow: 0 0 30px rgba(255, 215, 0, 0.4);
}
.header-subtitle {
    text-align: center;
    color: #c7d2fe;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    margin-top: 0.4rem;
    letter-spacing: 0.05em;
}

/* Section title */
.section-title {
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
    color: #ffd700;
    letter-spacing: 0.1em;
    font-size: 0.85rem;
    margin: 1.5rem 0 0.5rem 0;
    text-transform: uppercase;
}

/* Agent card */
.agent-card {
    background: rgba(28, 28, 44, 0.85);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0 1rem 0;
    border-left: 4px solid;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.agent-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45);
}
.agent-card.pro { border-left-color: #22c55e; }
.agent-card.skeptical { border-left-color: #ef4444; }
.agent-card.critic { border-left-color: #f59e0b; }
.agent-card.synthesizer { border-left-color: #8b5cf6; }
.agent-card.factchecker { border-left-color: #06b6d4; }
.agent-card.devil { border-left-color: #dc267f; }

.agent-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.8rem;
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    color: #f3f4f6;
}
.agent-header .iter-badge {
    background: rgba(255, 215, 0, 0.2);
    border: 1px solid #ffd700;
    color: #ffd700;
    border-radius: 999px;
    padding: 0.1rem 0.6rem;
    font-size: 0.75rem;
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
}
.agent-content {
    color: #d1d5db;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    line-height: 1.6;
}

/* Source badge */
.source-badge {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    margin: 0.15rem 0.2rem 0.15rem 0;
    border-radius: 999px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #c7d2fe;
}

/* Verdict pill */
.verdict-pill {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 999px;
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    margin-left: auto;
}
.verdict-accept, .verdict-pass {
    background: rgba(34, 197, 94, 0.18);
    color: #4ade80;
    border: 1px solid #22c55e;
    box-shadow: 0 0 12px rgba(34, 197, 94, 0.2);
}
.verdict-revise {
    background: rgba(245, 158, 11, 0.18);
    color: #fbbf24;
    border: 1px solid #f59e0b;
    box-shadow: 0 0 12px rgba(245, 158, 11, 0.2);
}

/* Pipeline timeline */
.timeline {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    background: rgba(20, 20, 35, 0.7);
    padding: 1rem 1.5rem;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    margin: 0.5rem 0 1.5rem 0;
}
.tl-step { display: flex; flex-direction: column; align-items: center; flex: 1; position: relative; }
.tl-step::after {
    content: '';
    position: absolute;
    top: 18px;
    left: calc(50% + 24px);
    right: calc(-50% + 24px);
    height: 2px;
    background: rgba(255, 255, 255, 0.1);
}
.tl-step:last-child::after { display: none; }
.tl-icon {
    width: 38px; height: 38px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.05rem;
    background: rgba(255, 255, 255, 0.04);
    border: 2px solid rgba(255, 255, 255, 0.18);
    transition: all 0.3s ease;
    z-index: 1;
}
.tl-icon.done {
    background: rgba(34, 197, 94, 0.25);
    border-color: #22c55e;
    box-shadow: 0 0 16px rgba(34, 197, 94, 0.4);
}
.tl-icon.active {
    background: rgba(255, 215, 0, 0.3);
    border-color: #ffd700;
    animation: pulse-gold 1.4s infinite;
}
@keyframes pulse-gold {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255, 215, 0, 0.6); transform: scale(1); }
    50% { box-shadow: 0 0 0 10px rgba(255, 215, 0, 0); transform: scale(1.08); }
}
.tl-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #9ca3af;
    margin-top: 0.4rem;
    text-align: center;
}
.tl-label.active { color: #ffd700; }
.tl-label.done { color: #4ade80; }

/* Rubric metric grid */
.rubric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.8rem;
    margin: 1rem 0;
}
.rubric-cell {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 10px;
    padding: 0.7rem;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.rubric-cell .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.rubric-cell .value {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.6rem;
    font-weight: 900;
    color: #ffd700;
    margin-top: 0.2rem;
}
.rubric-cell .bar {
    height: 4px; background: rgba(255, 255, 255, 0.08); border-radius: 2px; margin-top: 0.4rem; overflow: hidden;
}
.rubric-cell .bar-fill {
    height: 100%; background: linear-gradient(90deg, #f59e0b, #ffd700);
}

/* Claim list */
.claim {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    background: rgba(255, 255, 255, 0.03);
    padding: 0.7rem 0.9rem;
    border-radius: 8px;
    margin: 0.35rem 0;
    border-left: 3px solid #06b6d4;
}
.claim.unsupported { border-left-color: #f59e0b; }
.claim.contradicted { border-left-color: #ef4444; }
.claim.verified { border-left-color: #22c55e; }
.claim .icon { font-size: 1.1rem; flex-shrink: 0; }
.claim .body {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.5;
    color: #d1d5db;
}
.claim .note {
    color: #9ca3af;
    font-size: 0.75rem;
    font-style: italic;
    margin-top: 0.2rem;
}

/* Weakness item */
.weakness {
    background: rgba(220, 38, 127, 0.08);
    border-left: 3px solid #dc267f;
    padding: 0.6rem 0.9rem;
    border-radius: 8px;
    margin: 0.3rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #fbcfe8;
    line-height: 1.5;
}

/* Final answer card */
.final-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e1b4b 100%);
    border: 3px solid #ffd700;
    border-radius: 18px;
    padding: 2rem 2.4rem;
    margin: 2.5rem 0 1.5rem 0;
    box-shadow: 0 0 60px rgba(255, 215, 0, 0.25), inset 0 0 60px rgba(99, 102, 241, 0.1);
    position: relative;
}
.final-card::before {
    content: '';
    position: absolute;
    inset: -3px;
    border-radius: 18px;
    padding: 3px;
    background: linear-gradient(135deg, #ffd700, transparent, #ffd700);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    pointer-events: none;
}
.final-title {
    font-family: 'Orbitron', sans-serif;
    font-weight: 900;
    color: #ffd700;
    text-align: center;
    font-size: 1.6rem;
    letter-spacing: 0.15em;
    margin-bottom: 1.5rem;
    text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
}

/* Stats row */
.stat-card {
    background: rgba(28, 28, 44, 0.7);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.08);
    transition: all 0.2s;
}
.stat-card:hover { border-color: rgba(255, 215, 0, 0.4); }
.stat-value {
    font-family: 'Orbitron', sans-serif;
    font-weight: 900;
    font-size: 2rem;
    background: linear-gradient(90deg, #ffd700, #fbbf24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.2rem;
}

/* Primary run button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #ffd700 0%, #f59e0b 100%);
    color: #1a1a2e !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    letter-spacing: 0.08em;
    transition: all 0.2s ease;
    box-shadow: 0 4px 16px rgba(255, 215, 0, 0.3);
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(255, 215, 0, 0.5);
}

/* Secondary buttons (example chips) */
.stButton > button[kind="secondary"] {
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #c7d2fe;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    border-radius: 999px;
    padding: 0.4rem 0.9rem;
    transition: all 0.15s;
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(99, 102, 241, 0.25);
    border-color: #6366f1;
    transform: translateY(-1px);
}

/* Text input */
div[data-baseweb="input"] > div {
    background: rgba(20, 20, 35, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 10px !important;
}
input {
    font-family: 'JetBrains Mono', monospace !important;
    color: #f3f4f6 !important;
}

/* Spinner color */
.stSpinner > div { border-top-color: #ffd700 !important; }

/* Download button */
.stDownloadButton > button {
    background: rgba(255, 215, 0, 0.12);
    border: 1px solid #ffd700;
    color: #ffd700;
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
    border-radius: 8px;
    transition: all 0.2s;
}
.stDownloadButton > button:hover {
    background: rgba(255, 215, 0, 0.25);
    box-shadow: 0 4px 16px rgba(255, 215, 0, 0.3);
}
</style>
""",
    unsafe_allow_html=True,
)


# ─── HEADER ───────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="header-container">
    <h1 class="header-title">⚡ MULTI-AGENT RESEARCH ⚡</h1>
    <div class="header-subtitle">PRO ‖ SKEPTICAL → CRITIC → SYNTHESIZER → FACT-CHECKER → DEVIL'S ADVOCATE</div>
</div>
""",
    unsafe_allow_html=True,
)


# ─── INPUT AREA ───────────────────────────────────────────────────────────────

EXAMPLES = [
    "What is retrieval-augmented generation?",
    "What is AutoresearchClaw?",
    "Compare BDI and reactive agent architectures",
    "How does FAISS compare to ChromaDB?",
]

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

st.markdown('<div class="section-title">▸ Quick examples</div>', unsafe_allow_html=True)
ex_cols = st.columns(len(EXAMPLES))
for col, ex in zip(ex_cols, EXAMPLES):
    if col.button(ex, key=f"ex_{hash(ex)}", use_container_width=True):
        st.session_state.query_input = ex
        st.rerun()

st.markdown('<div class="section-title">▸ Your query</div>', unsafe_allow_html=True)
query = st.text_input(
    "query",
    value=st.session_state.query_input,
    placeholder="What would you like to research?",
    label_visibility="collapsed",
    key="query_field",
)
btn_col, _ = st.columns([1, 5])
run_btn = btn_col.button("🚀 RUN PIPELINE", type="primary", disabled=not query, use_container_width=True)


# ─── CONSTANTS ────────────────────────────────────────────────────────────────

SOURCE_LABELS = {
    "document_retriever": ("📚", "RAG"),
    "tavily_search_results_json": ("🌐", "Web · Tavily"),
    "duckduckgo_search": ("🌐", "Web · DDG"),
    "python_repl": ("🐍", "Python"),
}

CLAIM_ICONS = {
    "supported": ("✅", "supported"),
    "verified": ("🌐", "verified"),
    "unsupported": ("❌", "unsupported"),
    "contradicted": ("⚠️", "contradicted"),
}

PIPELINE_NODES = [
    ("researcher_pro", "🟢", "PRO"),
    ("researcher_skeptical", "🔴", "SKEPTIC"),
    ("critic", "⚖️", "CRITIC"),
    ("synthesizer", "✍️", "SYNTH"),
    ("fact_checker", "🔎", "FACT"),
    ("devils_advocate", "😈", "DEVIL"),
]


def render_timeline(completed: set, active: str | None = None) -> str:
    parts = []
    for node, icon, label in PIPELINE_NODES:
        cls = ""
        if node == active:
            cls = "active"
        elif node in completed:
            cls = "done"
        parts.append(
            f'<div class="tl-step">'
            f'  <div class="tl-icon {cls}">{icon}</div>'
            f'  <div class="tl-label {cls}">{label}</div>'
            f'</div>'
        )
    return f'<div class="timeline">{"".join(parts)}</div>'


def render_source_badges(sources: list[str]) -> str:
    unique = list(dict.fromkeys(sources or []))
    if not unique:
        return '<span class="source-badge">📭 LLM knowledge only</span>'
    return "".join(
        f'<span class="source-badge">{SOURCE_LABELS.get(s, ("•", s))[0]} {SOURCE_LABELS.get(s, ("•", s))[1]}</span>'
        for s in unique
    )


def render_verdict(verdict: str) -> str:
    v = (verdict or "?").lower()
    return f'<span class="verdict-pill verdict-{v}">{v.upper()}</span>'


def render_agent_card(klass: str, header_html: str, body_md: str) -> None:
    st.markdown(
        f'<div class="agent-card {klass}"><div class="agent-header">{header_html}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(body_md)


def render_rubric(rubric: dict) -> str:
    if not rubric:
        return ""
    cells = []
    for k, v in rubric.items():
        try:
            val = int(v)
        except (TypeError, ValueError):
            val = 0
        pct = max(0, min(100, val * 20))
        label = k.replace("_", " ")
        cells.append(
            f'<div class="rubric-cell">'
            f'  <div class="label">{label}</div>'
            f'  <div class="value">{val}<span style="font-size:0.9rem;color:#9ca3af">/5</span></div>'
            f'  <div class="bar"><div class="bar-fill" style="width:{pct}%"></div></div>'
            f'</div>'
        )
    return f'<div class="rubric-grid">{"".join(cells)}</div>'


# ─── PIPELINE EXECUTION ───────────────────────────────────────────────────────

if run_btn and query:
    st.session_state.query_input = query

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

    # Timeline placeholder (updates as agents complete)
    timeline_ph = st.empty()
    completed_nodes: set[str] = set()
    timeline_ph.markdown(render_timeline(completed_nodes), unsafe_allow_html=True)

    start_time = time.time()
    final_state: dict = {}
    graph = build_graph()

    with st.spinner("🧠 Agents are thinking..."):
        for step in graph.stream(initial_state):
            for node_name, updates in step.items():
                final_state.update(updates)
                completed_nodes.add(node_name)
                timeline_ph.markdown(render_timeline(completed_nodes), unsafe_allow_html=True)

                if node_name == "researcher_pro":
                    iteration = updates.get("research_iteration", 1)
                    iter_badge = f'<span class="iter-badge">REV {iteration}</span>' if iteration > 1 else ""
                    badges = render_source_badges(updates.get("research_sources_pro", []))
                    header = f'🟢 Researcher · PRO{iter_badge}<br><div style="margin-top:0.4rem">{badges}</div>'
                    render_agent_card("pro", header, updates.get("research_findings_pro", ""))

                elif node_name == "researcher_skeptical":
                    iteration = updates.get("research_iteration", 1)
                    iter_badge = f'<span class="iter-badge">REV {iteration}</span>' if iteration > 1 else ""
                    badges = render_source_badges(updates.get("research_sources_skeptical", []))
                    header = f'🔴 Researcher · SKEPTICAL{iter_badge}<br><div style="margin-top:0.4rem">{badges}</div>'
                    render_agent_card("skeptical", header, updates.get("research_findings_skeptical", ""))

                elif node_name == "critic":
                    critique = updates.get("critique", {})
                    verdict_pill = render_verdict(critique.get("verdict", "?"))
                    header = f'⚖️ Critic · Groq Qwen3-32B{verdict_pill}'
                    st.markdown(
                        f'<div class="agent-card critic"><div class="agent-header">{header}</div>'
                        f'{render_rubric(critique.get("rubric", {}))}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    feedback = critique.get("feedback", "")
                    if feedback:
                        st.markdown(f"> {feedback}")

                elif node_name == "synthesizer":
                    iteration = updates.get("synthesis_iteration", 1)
                    iter_badge = f'<span class="iter-badge">REV {iteration}</span>' if iteration > 1 else ""
                    header = f'✍️ Synthesizer{iter_badge}'
                    render_agent_card("synthesizer", header, updates.get("synthesis", ""))

                elif node_name == "fact_checker":
                    fc = updates.get("fact_check", {})
                    verdict_pill = render_verdict(fc.get("verdict", "?"))
                    header = f'🔎 Fact-Checker · CoV{verdict_pill}'
                    claims_html_parts = []
                    for c in fc.get("claims", []):
                        if not isinstance(c, dict):
                            continue
                        status = c.get("status", "")
                        icon, status_word = CLAIM_ICONS.get(status, ("•", status))
                        text = c.get("claim", "")
                        note = c.get("note", "")
                        note_html = f'<div class="note">{note}</div>' if note else ""
                        claims_html_parts.append(
                            f'<div class="claim {status}"><div class="icon">{icon}</div>'
                            f'<div class="body"><strong>{status_word}</strong> — {text}{note_html}</div></div>'
                        )
                    issues_md = "\n".join(f"- ⚠ {i}" for i in fc.get("issues", []))
                    st.markdown(
                        f'<div class="agent-card factchecker"><div class="agent-header">{header}</div>'
                        f'{"".join(claims_html_parts)}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if issues_md:
                        st.markdown("**Issues to fix:**")
                        st.markdown(issues_md)

                elif node_name == "devils_advocate":
                    da = updates.get("devils_advocate", {})
                    verdict_pill = render_verdict(da.get("verdict", "?"))
                    header = f'😈 Devil\'s Advocate · Red Team{verdict_pill}'
                    weaknesses_html = "".join(
                        f'<div class="weakness">▸ {w}</div>' for w in da.get("weaknesses", [])
                    )
                    if not weaknesses_html:
                        weaknesses_html = '<div class="weakness" style="border-color:#22c55e;color:#86efac;background:rgba(34,197,94,0.08)">▸ No structural weaknesses identified</div>'
                    st.markdown(
                        f'<div class="agent-card devil"><div class="agent-header">{header}</div>'
                        f'{weaknesses_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    elapsed = time.time() - start_time

    # ─── FINAL ANSWER ─────────────────────────────────────────────────────────
    final_answer = final_state.get("final_answer") or final_state.get("synthesis") or "No answer produced."

    st.markdown(
        f'<div class="final-card"><div class="final-title">★  FINAL ANSWER  ★</div>',
        unsafe_allow_html=True,
    )
    st.markdown(final_answer)
    st.markdown("</div>", unsafe_allow_html=True)

    # Download + copy actions
    dl_col, _ = st.columns([1, 5])
    dl_col.download_button(
        "⬇ DOWNLOAD ANSWER",
        data=f"# {query}\n\n{final_answer}\n",
        file_name="research_answer.md",
        mime="text/markdown",
        use_container_width=True,
    )

    # ─── STATS PANEL ──────────────────────────────────────────────────────────
    n_research_revs = max(final_state.get("research_iteration", 1), 1)
    n_synth_revs = max(final_state.get("synthesis_iteration", 1), 1)
    all_sources = list(
        dict.fromkeys(
            (final_state.get("research_sources_pro") or [])
            + (final_state.get("research_sources_skeptical") or [])
        )
    )
    critic_verdict = (final_state.get("critique") or {}).get("verdict", "?")
    fc_verdict = (final_state.get("fact_check") or {}).get("verdict", "?")
    da_verdict = (final_state.get("devils_advocate") or {}).get("verdict", "?")
    n_claims = len((final_state.get("fact_check") or {}).get("claims", []))

    st.markdown('<div class="section-title">▸ Run statistics</div>', unsafe_allow_html=True)
    stat_cols = st.columns(5)
    stats = [
        (f"{elapsed:.1f}s", "Total time"),
        (str(n_research_revs), "Research rounds"),
        (str(n_synth_revs), "Synthesis rounds"),
        (str(len(all_sources)), "Unique sources"),
        (str(n_claims), "Claims checked"),
    ]
    for col, (val, lbl) in zip(stat_cols, stats):
        col.markdown(
            f'<div class="stat-card"><div class="stat-value">{val}</div>'
            f'<div class="stat-label">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    # Verdict summary
    st.markdown('<div class="section-title">▸ Final verdicts</div>', unsafe_allow_html=True)
    v_cols = st.columns(3)
    for col, (label, verdict) in zip(
        v_cols,
        [("⚖️ Critic", critic_verdict), ("🔎 Fact-Checker", fc_verdict), ("😈 Devil's Advocate", da_verdict)],
    ):
        col.markdown(
            f'<div class="stat-card" style="text-align:left">'
            f'<div class="stat-label" style="text-align:left">{label}</div>'
            f'<div style="margin-top:0.5rem">{render_verdict(verdict)}</div></div>',
            unsafe_allow_html=True,
        )

else:
    # ─── IDLE STATE ───────────────────────────────────────────────────────────
    st.markdown(
        """
<div style="text-align:center;padding:3rem 1rem;color:#9ca3af;font-family:'JetBrains Mono',monospace">
    <div style="font-size:4rem;margin-bottom:1rem">🧠</div>
    <div style="font-size:1rem">Pick an example above or type your own query, then hit RUN.</div>
    <div style="font-size:0.85rem;margin-top:0.5rem;color:#6b7280">Six agents will debate, verify, and red-team the answer before you see it.</div>
</div>
""",
        unsafe_allow_html=True,
    )
