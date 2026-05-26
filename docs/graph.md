# Agent Graph (last run)

```mermaid
flowchart LR
    START([START]) --> researcher[Researcher]
    researcher[Researcher] --> critic[Critic]
    critic[Critic] --> synthesizer[Synthesizer]
    synthesizer[Synthesizer] --> fact_checker[Fact_checker]
    fact_checker[Fact_checker] --> END([END])
    style START fill:#6366f1,color:#fff,stroke:#4f46e5
    style END fill:#6366f1,color:#fff,stroke:#4f46e5
    style researcher fill:#22c55e,color:#fff,stroke:#16a34a
    style critic fill:#22c55e,color:#fff,stroke:#16a34a
    style synthesizer fill:#22c55e,color:#fff,stroke:#16a34a
    style fact_checker fill:#22c55e,color:#fff,stroke:#16a34a
```
