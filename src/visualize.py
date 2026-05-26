"""Generate a Mermaid diagram — last-run execution trace (via LangSmith) or static graph."""

import sys
from pathlib import Path
from src.graph import build_graph

AGENT_NODES = {"researcher", "critic", "synthesizer", "fact_checker"}


def _fetch_last_run_nodes() -> list[str] | None:
    """Return ordered node names from the most recent LangSmith run."""
    try:
        from langsmith import Client
        client = Client()

        roots = list(client.list_runs(
            project_name="multi-agent-research",
            limit=1,
            is_root=True,
        ))
        if not roots:
            print("No runs found in LangSmith project 'multi-agent-research'.")
            return None

        root = roots[0]
        children = list(client.list_runs(
            project_name="multi-agent-research",
            parent_run_id=root.id,
        ))
        children.sort(key=lambda r: r.start_time)
        nodes = [r.name for r in children if r.name in AGENT_NODES]
        print(f"Last run: {root.name} ({root.start_time:%Y-%m-%d %H:%M:%S})")
        print(f"Execution path: {' → '.join(nodes)}")
        return nodes
    except Exception as exc:
        import traceback
        print(f"Could not fetch LangSmith trace: {exc}")
        traceback.print_exc()
        return None


def _execution_mermaid(nodes: list[str]) -> str:
    """Build a Mermaid flowchart for the actual execution path."""
    path = ["START"] + nodes + ["END"]
    lines = ["flowchart LR"]
    for i in range(len(path) - 1):
        src = path[i]
        dst = path[i + 1]
        src_shape = f'([{src}])' if src in ("START", "END") else f'[{src.capitalize()}]'
        dst_shape = f'([{dst}])' if dst in ("START", "END") else f'[{dst.capitalize()}]'
        lines.append(f"    {src}{src_shape} --> {dst}{dst_shape}")
    lines.append("    style START fill:#6366f1,color:#fff,stroke:#4f46e5")
    lines.append("    style END fill:#6366f1,color:#fff,stroke:#4f46e5")
    for node in nodes:
        lines.append(f"    style {node} fill:#22c55e,color:#fff,stroke:#16a34a")
    return "\n".join(lines)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "last-run"

    if mode == "static":
        graph = build_graph()
        mermaid = graph.get_graph().draw_mermaid()
        label = "static graph"
    else:
        nodes = _fetch_last_run_nodes()
        if nodes:
            mermaid = _execution_mermaid(nodes)
            label = "last run"
        else:
            print("Falling back to static graph.")
            graph = build_graph()
            mermaid = graph.get_graph().draw_mermaid()
            label = "static graph"

    out = Path("docs/graph.md")
    out.write_text(f"# Agent Graph ({label})\n\n```mermaid\n{mermaid}\n```\n")
    print(f"\nDiagram written to {out}")
    print()
    print(mermaid)


if __name__ == "__main__":
    main()
