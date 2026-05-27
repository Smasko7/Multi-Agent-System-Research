# Theory Mapping

This document explains how the six-agent system instantiates the three
multi-agent theories the project is graded against: **BDI**, **FIPA ACL**, and
**Contract Net Protocol**. Each mapping covers what the theory specifies, what
the implementation does, and where the code lives.

---

## 1. BDI Architecture (Belief-Desire-Intention)

| BDI Component | System Mapping |
|---|---|
| **Beliefs** | The `AgentState` TypedDict acts as the shared belief base. Each agent reads only the state fields relevant to its role and updates its own output fields. Beliefs include retrieved corpus passages, web search results, prior research findings, the critic's rubric scores, fact-check claim statuses, and devil's-advocate weaknesses. |
| **Desires** | Each agent's role-specific goal, encoded in its system prompt. The PRO researcher *desires* supporting evidence; the SKEPTICAL researcher *desires* counter-evidence; the Critic *desires* high rubric scores; the Fact-Checker *desires* corroborated claims; the Devil's Advocate *desires* surfaced weaknesses; the Synthesizer *desires* a balanced answer. |
| **Intentions** | The currently-executing node in the LangGraph pipeline, plus the tool-call sequence each agent commits to during execution. Intention reconsideration happens at conditional edges: when Critic returns `revise`, the system updates its intention to re-run both researchers; when Fact-Checker or Devil's Advocate returns `revise`, the system re-runs the Synthesizer. |

**Practical instantiation:**
- Beliefs are typed: `research_findings_pro: str`, `critique: dict`, etc. See `src/state.py`.
- Desires are baked into system prompts (e.g. `RESEARCHER_SYSTEM_PROMPT` in `src/agents/researcher.py` explicitly states the PRO and SKEPTICAL angles).
- Intentions manifest as tool-call loops in researcher / critic / fact-checker nodes — each invokes its bound tools up to N rounds before producing a final output.

---

## 2. FIPA ACL (Agent Communication Language)

| FIPA Performative | System Equivalent |
|---|---|
| `request` | Initial query entering the pipeline via `main.py` / `app.py` / `arcade.py` |
| `inform` | Researchers returning `research_findings_pro` and `research_findings_skeptical` to the shared state |
| `propose` | Synthesizer returning `synthesis` to the state |
| `evaluate-proposal` | Critic evaluating both researchers' findings with a structured rubric; Fact-Checker running Chain-of-Verification on the synthesis; Devil's Advocate red-teaming the synthesis |
| `accept-proposal` | Critic verdict `accept` → route to Synthesizer; Fact-Checker verdict `pass` + Devil's Advocate verdict `pass` → route to END |
| `reject-proposal` | Critic verdict `revise` → re-fan-out to both researchers; Fact-Checker verdict `revise` OR Devil's Advocate verdict `revise` → route back to Synthesizer |
| `confirm` | Final answer surfaced by the entry-point (CLI prints it, Streamlit displays the hero card, arcade shows the centered modal) |
| `failure` (implicit) | `MAX_ITERATIONS = 2` cap exits the loop even if reviewers still want revision — analogous to a FIPA `failure` notification that the contract could not be completed within the agreed budget |

**Message structure:**
- The judges' JSON outputs are FIPA-style structured messages. Critic emits `{rubric: {…}, verdict, feedback}` — directly modeling an `evaluate-proposal` payload. Fact-Checker emits per-claim `{claim, status, note}` records, modeling a fine-grained `inform` performative. Devil's Advocate emits `{weaknesses: [...], verdict}` — an adversarial counter-proposal.
- **Multiple reviewers, OR-aggregated rejection:** the synthesizer accepts the post-synthesis stage only when BOTH the Fact-Checker AND the Devil's Advocate accept. Either one's `reject-proposal` triggers revision. This is a FIPA-style multi-party agreement.

---

## 3. Contract Net Protocol (CNP)

| CNP Role / Step | System Equivalent |
|---|---|
| **Manager** | The Critic in the research phase; the Synthesizer↔Reviewers loop in the synthesis phase |
| **Contractors** | The two parallel Researchers (PRO + SKEPTICAL) submitting "bids" = research outputs; the Synthesizer submitting a "bid" = drafted answer |
| **Call for Proposals** | LangGraph `START` dispatches to both researchers in parallel (two `add_edge(START, …)` edges) |
| **Bid / Proposal** | `research_findings_pro` and `research_findings_skeptical`, produced independently and submitted to the Critic |
| **Multiple parallel bids** | The two researchers' independent outputs constitute parallel CNP bids — the Critic evaluates them as a set, not individually |
| **Award** | Critic verdict `accept` → pipeline advances to the Synthesizer |
| **Re-announcement** | Critic verdict `revise` returns `["researcher_pro", "researcher_skeptical"]` from `critic_router` — LangGraph dispatches both researchers in parallel for a second round of bidding, with the critic's structured feedback attached |
| **MAX_ITERATIONS cap** | Bounds the contract: at most 2 re-announcements before forced acceptance, ensuring termination |

**Why this is a "real" CNP and not just naming:**
- The two researchers are genuinely independent — they share no state during their execution, only when they both write to the shared `AgentState` upon completion.
- They have **different angles** baked into their system prompts (one is told to find supporting evidence, the other counter-evidence), so their bids are intentionally divergent — the Critic is selecting from genuinely distinct proposals, not duplicate ones.
- The Critic's evaluation is **structured** (rubric scoring) — analogous to a CNP manager evaluating bids on multiple dimensions, not just yes/no.
- Re-announcement is **broadcast, not unicast**: the critic's feedback is attached to both researchers' second attempts via the `critique.feedback` state field. Both contractors revise based on the same management feedback.

The post-synthesis stage (Fact-Checker + Devil's Advocate reviewing the Synthesizer's draft) can be modeled as a **second CNP** where the Synthesizer is the contractor and the two reviewers act as a "review committee" — both must accept the proposal, and either's rejection triggers a re-submission.

---

## Why This Mapping Matters

The mappings are not retrospective gloss — they constrained the architecture
during design:

- **BDI** drove the strict separation of beliefs (state) from intentions (system prompts), so each agent has a typed input/output contract.
- **FIPA ACL** drove the use of structured JSON for judge outputs instead of free-text "the answer is fine I think". This is what makes the `revise`/`pass` routing logic reliable.
- **CNP** drove the choice to have **two researchers in parallel rather than one** — single-bidder Contract Net is degenerate. The pro/skeptical split makes the bidding meaningful.

Without these constraints, the easy default would have been a single linear
chain of LLM calls with free-text outputs and ad-hoc routing — which would not
have mapped to any multi-agent theory in a recognizable way.
