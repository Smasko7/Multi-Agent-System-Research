"""Central configuration for the multi-agent system."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Model Configuration ---
# Generation-heavy agents (Researchers, Synthesizer) — Gemini 3.1 Flash-Lite
# 15 RPM / 500 RPD free tier (vs. 2.5 Flash's 5 RPM which the parallel fan-out blows past)
GENERATION_MODEL = "gemini-3.1-flash-lite"
# Judgment agents (Critic, Fact-Checker, Devil's Advocate) — Groq Qwen3-32B
# Different model family from generation agents → genuinely independent opinions.
# Qwen3 is a reasoning model that emits <think>...</think> blocks; the shared
# extract_text helper strips them before JSON parsing.
JUDGMENT_MODEL = "qwen/qwen3-32b"
# Evaluation LLM-as-judge — Gemini Flash-Lite (gemini-2.5-pro blocked on free tier)
EVAL_MODEL = "gemini-3.1-flash-lite"

# Gemini 2.5+ requires temperature >= 1.0 to avoid degraded output
MODEL_TEMPERATURE = 1.0

# --- Workflow ---
MAX_ITERATIONS = 2  # Hard cap on feedback loops per gate
RATE_LIMIT_SLEEP = 2  # Seconds to sleep between LLM calls

# --- RAG ---
CORPUS_DIR = "data/corpus"
VECTORSTORE_DIR = "data/vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
