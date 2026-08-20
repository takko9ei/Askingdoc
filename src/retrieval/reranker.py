"""
Cross-Encoder reranking (RAG pipeline: step (5) precision ranking -- the
second stage of the project's "recall-then-rank" architecture).

vector_search / bm25_search / rrf_fuse in search.py are the cheap, coarse
RECALL stage: they narrow the full corpus down to a small candidate set
(cfg.retrieval.recall_top_k, 20 by default) fast enough to run on every
query. This module is the expensive, precise RANK stage: a Cross-Encoder
reads the query and each candidate's full text TOGETHER (not as two
independently-computed vectors) and re-scores just that small candidate
set. See this STEP's 【输出后请解释】 answer #1/#2 for why this two-stage
split exists and why it's near-universal in real search/recommendation
systems.
"""

from __future__ import annotations

import time

from sentence_transformers import CrossEncoder

from src.config import Config
from src.retrieval.search import Hit

# Cross-Encoder models are full transformer models loaded off disk (or
# downloaded the very first time -- already done in this project's STEP0
# "landmine 2" check). Loading one per query would dominate total
# latency, so it's cached exactly like search.py's chroma/embedding/BM25
# resources. Fingerprinted on the model name so a cfg pointing at a
# different reranker (e.g. STEP0's mentioned bge-reranker-base fallback)
# doesn't silently keep using a model instance built for a different one.
_reranker_models: dict[str, CrossEncoder] = {}


def _reranker_for(cfg: Config) -> CrossEncoder:
    model_name = cfg.models.reranker
    if model_name not in _reranker_models:
        print(f"[reranker] loading {model_name} (first call only)...")
        start = time.monotonic()
        _reranker_models[model_name] = CrossEncoder(model_name)
        elapsed = time.monotonic() - start
        print(f"[reranker] loaded {model_name} in {elapsed:.1f}s")
    return _reranker_models[model_name]


def reset_reranker() -> None:
    """Drop the cached CrossEncoder instance(s). Mirrors search.py's
    reset_clients() for the same "blunt, explicit start-fresh" reason --
    though unlike the Chroma collection, a loaded reranker model isn't
    dev/full-sensitive (the model doesn't care which corpus produced the
    candidates it's scoring), so this mainly matters for freeing memory
    or forcing a reload after switching cfg.models.reranker."""
    _reranker_models.clear()


def rerank(query: str, hits: list[Hit], cfg: Config) -> list[Hit]:
    """Re-score `hits` with a Cross-Encoder and return them sorted by the
    new score, descending.

    Unlike vector_search/bm25_search -- which score each candidate
    independently of the others and independently of the query, then
    compare the results after the fact -- a Cross-Encoder scores each
    (query, candidate) PAIR jointly in one forward pass. That joint
    scoring is what makes it more precise (it can weigh subtle relevance
    signals a cosine-similarity or term-overlap comparison can't see) and
    also what makes it too slow to run over an entire corpus: nothing
    about it can be precomputed ahead of the query, so it only runs here,
    over the already-small `hits` list (typically ~20), never the full
    index.
    """
    model = _reranker_for(cfg)
    pairs = [(query, hit.text) for hit in hits]

    start = time.monotonic()
    scores = model.predict(pairs, batch_size=cfg.retrieval.rerank_batch)
    elapsed = time.monotonic() - start

    print(f"[reranker] scored {len(pairs)} pairs in {elapsed:.2f}s")
    if elapsed > 3.0:
        print(
            "[reranker] WARNING: this rerank call took over 3s -- on CPU that will make "
            "the CLI feel sluggish per question. Consider reducing "
            "config.retrieval.recall_top_k (fewer candidates to rerank) or switching "
            "config.models.reranker to BAAI/bge-reranker-base (smaller/faster, small "
            "accuracy trade-off -- this is the STEP0-documented fallback)."
        )

    reranked = [
        Hit(
            cid=hit.cid,
            text=hit.text,
            page_no=hit.page_no,
            score=hit.score,  # original recall-stage score, kept for comparison
            parent_id=hit.parent_id,
            rerank_score=float(score),
        )
        for hit, score in zip(hits, scores)
    ]
    reranked.sort(key=lambda h: -h.rerank_score)
    return reranked
