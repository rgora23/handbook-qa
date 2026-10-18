"""Hybrid search over the chunks: by meaning (vectors) and by exact words (BM25).

Vectors: every chunk is embedded once and cached on disk. About 1,000 chunks x 384 numbers
is ~1.5MB, so a brute-force dot product against every chunk is instant. No vector database.

BM25: classic keyword scoring. It catches exact terms the vectors rank low, like "401k".
It's rebuilt from the chunk text at startup, which takes well under a second.

The two rankings are merged with reciprocal rank fusion.
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

from handbook_qa import config
from handbook_qa.chunker import Chunk, Segment, chunk_document
from handbook_qa.pdf_loader import load_pdf

# Bump when chunking or cache layout changes in a way the settings below don't capture.
CACHE_VERSION = 3


_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # vector (cosine) similarity, higher is closer. Used for the "not covered" cutoff.


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: TextEmbedding):
        self.chunks = chunks
        self.vectors = vectors  # (n_chunks, dim), rows normalized to length 1
        self._embedder = embedder
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])

    def search(self, query: str, top_k: int) -> list[Hit]:
        """The top_k chunks by fused rank. Each Hit carries its vector score."""
        query_vector = _normalize(next(iter(self._embedder.query_embed(query))))
        vector_scores = self.vectors @ query_vector
        keyword_scores = self._bm25.get_scores(_tokenize(query))
        fused = _reciprocal_rank_fusion([vector_scores, keyword_scores])
        top = np.argsort(-fused)[:top_k]
        return [Hit(chunk=self.chunks[i], score=float(vector_scores[i])) for i in top]


def _reciprocal_rank_fusion(score_lists: list[np.ndarray]) -> np.ndarray:
    """Each ranking votes 1 / (RRF_K + rank) for its top FUSION_DEPTH chunks.

    Using ranks, not raw scores, means we never have to compare a cosine score with a
    BM25 score, which live on different scales. A chunk near the top of either list rises.
    """
    fused = np.zeros(len(score_lists[0]))
    for scores in score_lists:
        for rank, i in enumerate(np.argsort(-scores)[: config.FUSION_DEPTH], start=1):
            if scores[i] <= 0:
                break  # no match at all (e.g. no shared words for BM25), so no vote
            fused[i] += 1 / (config.RRF_K + rank)
    return fused


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def is_cached(pdf_path: Path) -> bool:
    return all(p.exists() for p in _cache_paths(pdf_path))


def load_index(pdf_path: Path) -> Index:
    """Load the index from the cache, or build it (slow: minutes for ~1,000 pages)."""
    embedder = TextEmbedding(config.EMBED_MODEL, cache_dir=str(config.CACHE_DIR / "models"))
    chunks_path, vectors_path = _cache_paths(pdf_path)

    if chunks_path.exists() and vectors_path.exists():
        chunks = [_chunk_from_dict(d) for d in json.loads(chunks_path.read_text())]
        vectors = np.load(vectors_path)
        return Index(chunks, vectors, embedder)

    doc = load_pdf(pdf_path)
    chunks = chunk_document(doc, config.CHUNK_CHARS, config.OVERLAP_CHARS)
    if not chunks:
        raise ValueError(f"No text found in {pdf_path.name}. Is it a scanned PDF?")
    vectors = np.array([_normalize(v) for v in embedder.passage_embed([c.text for c in chunks])])

    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, vectors)
    # Chunks last: a cache only counts once both files exist.
    chunks_path.write_text(json.dumps([asdict(c) for c in chunks]))
    return Index(chunks, vectors, embedder)


def _cache_paths(pdf_path: Path) -> tuple[Path, Path]:
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    settings = f"{CACHE_VERSION}|{config.EMBED_MODEL}|{config.CHUNK_CHARS}|{config.OVERLAP_CHARS}"
    key = hashlib.sha256(f"{digest}|{settings}".encode()).hexdigest()[:16]
    return config.CACHE_DIR / f"{key}.chunks.json", config.CACHE_DIR / f"{key}.vectors.npy"


def _chunk_from_dict(d: dict) -> Chunk:
    return Chunk(
        id=d["id"],
        text=d["text"],
        source=d["source"],
        segments=tuple(Segment(**s) for s in d["segments"]),
    )


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)
