# Theory Mapping

## BDI Architecture Mapping

| BDI Component | System Mapping |
|---|---|
| **Beliefs** | `AgentState` — shared state dict holding query, findings, critique, synthesis, fact_check |
| **Desires** | Each agent's goal: Researcher → find information; Critic → ensure quality; Synthesizer → produce coherent answer; Fact-Checker → ensure accuracy |
| **Intentions** | The currently executing node in the LangGraph pipeline; committed plans implemented as system prompts + tool calls |

The `AgentState` TypedDict acts as the shared belief base. Each agent reads relevant state fields (beliefs) and updates only its own outputs (intention execution). The LangGraph conditional edges model intention reconsideration: when the Critic or Fact-Checker returns "revise", the system revises its intention to re-run the upstream agent.

## FIPA ACL Mapping

| FIPA Performative | System Equivalent |
|---|---|
| `request` | Initial query entering the pipeline via `main.py` |
| `inform` | Researcher returning `research_findings` to the state |
| `propose` | Synthesizer returning `synthesis` to the state |
| `evaluate-proposal` | Critic evaluating `research_findings`; Fact-Checker evaluating `synthesis` |
| `accept-proposal` | Critic verdict `"accept"` → route to Synthesizer; Fact-Checker verdict `"pass"` → route to END |
| `reject-proposal` | Critic verdict `"revise"` → route back to Researcher; Fact-Checker verdict `"revise"` → route back to Synthesizer |
| `confirm` | Final answer returned from `run()` in `main.py` |

The structured JSON responses from Critic and Fact-Checker (`{"verdict": ..., "feedback/issues": ...}`) model FIPA ACL message content.

## Contract Net Protocol Mapping

| CNP Role/Step | System Equivalent |
|---|---|
| **Manager** | Critic agent (announces task, evaluates bids, awards or re-announces) |
| **Contractor** | Researcher agent (submits bids = research findings) |
| **Call for Proposals** | Query entering the Researcher node |
| **Bid/Proposal** | `research_findings` returned by Researcher |
| **Award** | Critic verdict `"accept"` → pipeline continues to Synthesizer |
| **Re-announcement** | Critic verdict `"revise"` + feedback → Researcher re-runs with feedback |
| **MAX_ITERATIONS cap** | Ensures the contract terminates (no infinite re-announcement loops) |

The same pattern applies to the Synthesizer ↔ Fact-Checker loop, where Fact-Checker acts as manager and Synthesizer as contractor.
