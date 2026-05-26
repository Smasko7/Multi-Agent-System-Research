"""Evaluation harness — compares multi-agent pipeline vs single-agent baseline."""

import json
import time
import logging
from pathlib import Path

from src import config
from src.main import run as run_pipeline
from src.evaluation.baseline import run_baseline

logger = logging.getLogger(__name__)

QUERIES_FILE = Path(__file__).parent / "queries.json"
RESULTS_FILE = Path("evaluation_results.json")

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for AI-generated research answers.
Score the answer on these four dimensions (0-5 each):
- correctness: How many of the expected key facts are present and accurately stated?
- completeness: How thorough and comprehensive is the answer?
- coherence: How readable, well-organized, and clear is the answer?
- hallucination_penalty: How many unsupported or clearly wrong claims appear? (0=many errors, 5=none)

Respond with EXACTLY this JSON and nothing else:
{
  "correctness": <0-5>,
  "completeness": <0-5>,
  "coherence": <0-5>,
  "hallucination_penalty": <0-5>,
  "reasoning": "brief explanation"
}"""


def _llm_judge(query: str, answer: str, key_facts: list[str]) -> dict:
    """Use Gemini Pro as LLM judge to score an answer."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage
    import re

    llm = ChatGoogleGenerativeAI(
        model=config.EVAL_MODEL,
        temperature=config.MODEL_TEMPERATURE,
    )
    human_content = (
        f"Query: {query}\n\n"
        f"Expected key facts:\n" + "\n".join(f"- {f}" for f in key_facts) + "\n\n"
        f"Answer to evaluate:\n{answer}"
    )
    response = llm.invoke([
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ])
    content = response.content

    # Parse JSON
    for pattern in [r"```(?:json)?\s*(.*?)\s*```", r"\{.*\}"]:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1) if "```" in pattern else match.group(0))
            except json.JSONDecodeError:
                pass
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    logger.warning("Judge parse failed; returning zeros. Raw: %s", content[:200])
    return {"correctness": 0, "completeness": 0, "coherence": 0, "hallucination_penalty": 0, "reasoning": "parse error"}


def _key_fact_score(answer: str, key_facts: list[str]) -> float:
    """Simple string-match correctness score (0-5)."""
    hits = sum(1 for fact in key_facts if any(word.lower() in answer.lower() for word in fact.split() if len(word) > 4))
    return round(min(5.0, hits / max(len(key_facts), 1) * 5), 2)


def evaluate(use_llm_judge: bool = False):
    """Run full evaluation and save results."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    with open(QUERIES_FILE) as f:
        queries = json.load(f)

    results = []

    for i, item in enumerate(queries):
        query = item["query"]
        key_facts = item["key_facts"]
        logger.info("[%d/%d] Evaluating: %s", i + 1, len(queries), query[:60])

        pipeline_answer = run_pipeline(query)
        time.sleep(config.RATE_LIMIT_SLEEP)

        baseline_answer = run_baseline(query)
        time.sleep(config.RATE_LIMIT_SLEEP)

        if use_llm_judge:
            logger.info("  LLM-judging pipeline answer...")
            pipeline_scores = _llm_judge(query, pipeline_answer, key_facts)
            time.sleep(15)  # Gemini Pro 5 RPM limit

            logger.info("  LLM-judging baseline answer...")
            baseline_scores = _llm_judge(query, baseline_answer, key_facts)
            time.sleep(15)
        else:
            pipeline_scores = {
                "correctness": _key_fact_score(pipeline_answer, key_facts),
                "completeness": 0,
                "coherence": 0,
                "hallucination_penalty": 5,
                "reasoning": "string-match only",
            }
            baseline_scores = {
                "correctness": _key_fact_score(baseline_answer, key_facts),
                "completeness": 0,
                "coherence": 0,
                "hallucination_penalty": 5,
                "reasoning": "string-match only",
            }

        results.append({
            "query": query,
            "pipeline": {"answer": pipeline_answer, "scores": pipeline_scores},
            "baseline": {"answer": baseline_answer, "scores": baseline_scores},
        })

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", RESULTS_FILE)

    _print_table(results)
    return results


def _print_table(results: list):
    """Print comparison table to stdout."""
    metrics = ["correctness", "completeness", "coherence", "hallucination_penalty"]
    header = f"{'Query':<45} {'Metric':<22} {'Pipeline':>8} {'Baseline':>8}"
    print("\n" + "=" * len(header))
    print("EVALUATION RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for item in results:
        query_short = item["query"][:43] + ".." if len(item["query"]) > 43 else item["query"]
        for j, metric in enumerate(metrics):
            p_score = item["pipeline"]["scores"].get(metric, 0)
            b_score = item["baseline"]["scores"].get(metric, 0)
            label = query_short if j == 0 else ""
            print(f"{label:<45} {metric:<22} {p_score:>8.2f} {b_score:>8.2f}")
        print("-" * len(header))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-judge", action="store_true", help="Use Gemini Pro as LLM judge")
    args = parser.parse_args()
    evaluate(use_llm_judge=args.llm_judge)
