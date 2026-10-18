"""Defaults in one place. Values that a user might reasonably change can be set by env var."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF = PROJECT_ROOT / "Sample handbook.pdf"
CACHE_DIR = PROJECT_ROOT / ".cache"

# Chunking: ~15% overlap. 2,000 characters keeps almost every chunk under the embedding
# model's 512-token input limit (the handbook averages ~4.7 characters per token).
CHUNK_CHARS = 2000
OVERLAP_CHARS = 300

# Retrieval
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 5
# Hybrid search: vector and BM25 rankings each vote 1 / (RRF_K + rank) for their top
# FUSION_DEPTH chunks. 60 is the standard constant from the reciprocal rank fusion paper.
FUSION_DEPTH = 50
RRF_K = 60
# If no retrieved chunk has a vector score at least this high, the answer is
# "the handbook doesn't cover this". BM25 only affects ranking, not this check.
# Picked from measured scores: off-topic questions topped out at 0.54, on-topic started at 0.655.
MIN_SCORE: float | None = 0.60

# Answering
ANSWER_MODEL = os.environ.get("HANDBOOK_QA_MODEL", "claude-sonnet-5-5")
# How hard Claude thinks before answering. Answering from 5 excerpts is simple, and a
# terminal user is waiting, so start low.
ANSWER_EFFORT = os.environ.get("HANDBOOK_QA_EFFORT", "low")
