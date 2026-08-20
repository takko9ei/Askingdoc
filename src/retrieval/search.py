"""
Retrieval layer -- the project's core module (RAG pipeline: step (4)
retrieval, with step (5) fusion/rerank and small-to-big expansion stubbed
in as TODOs for STEP4/STEP5).

search() is deliberately ONE function with four sequential if-branches --
vector / +BM25 / +rerank / +small-to-big -- gated by cfg.retrieval.use_bm25
/ use_rerank / use_small_to_big. This is what lets the four ablation
configurations (baseline/hybrid/rerank/full) be "one codebase, four config
flags" instead of four hand-maintained near-duplicates: eval/harness.py
(STEP3) will call this exact function four times with four different `cfg`
objects, so any difference in the resulting Recall@5/MRR numbers is
guaranteed to come ONLY from the flag that changed.

This slice implements just the unconditional first line (vector_search).
The other three branches stay `pass` TODOs on purpose -- filling them in
now would let this slice quietly do STEP4/5's work too, which breaks the
per-step review discipline this project runs on (see CLAUDE.md section 4).

Note on adapting the sketch skeleton to this project's actual Config shape:
the playbook's pseudocode writes `cfg.use_bm25` / `cfg.recall_top_k`
directly; this project's real Config (src/config.py, STEP0) nests those
under `cfg.retrieval.*`, so the branches below read `cfg.retrieval.use_bm25`
etc. instead. The BRANCH STRUCTURE (four sequential ifs, in this order) is
unchanged -- only the attribute paths were corrected to match the actual
typed Config dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
from openai import OpenAI

from src.config import Config


@dataclass
class Hit:
    """One retrieved chunk -- structured, not a pre-formatted string.
    eval/harness.py (STEP3) needs to read `page_no` off of this
    mechanically to compute Recall@5/Recall@20/MRR; a formatted
    "page 87: some text..." string would make that fragile to parse
    back apart, and would tie retrieval's output shape to one particular
    way of displaying it (cli.py's, say) instead of letting every
    consumer (CLI, eval harness, future API) format it however it needs.
    """

    cid: str
    text: str
    page_no: int
    score: float
    parent_id: str


# --- module-level singletons ---------------------------------------------
# Opening a new PersistentClient / OpenAI client on every single query
# would mean re-opening the on-disk Chroma database and redoing HTTP
# client setup per question -- wasteful for a process expected to answer
# many questions in a row (the CLI loop below, STEP7's API service).
#
# These caches are keyed by a "fingerprint" string/tuple derived from the
# cfg fields that actually determine which physical resource to use --
# NOT cached unconditionally on "has this been created yet". That
# distinction is what makes this safe across dev/full switching: dev and
# full cfgs produce different fingerprints (different collection_name),
# so they land in different cache slots and can coexist in the same
# process without one silently shadowing the other, even if the caller
# never calls reset_clients() at all.
_chroma_clients: dict[str, chromadb.ClientAPI] = {}
_collections: dict[str, chromadb.Collection] = {}
_embedding_clients: dict[tuple, OpenAI] = {}


def _collection_for(cfg: Config) -> chromadb.Collection:
    """Return the (cached) Chroma collection for this cfg's chroma_dir +
    collection_name. Both are part of the fingerprint -- chroma_dir alone
    wouldn't distinguish dev/full (they share one on-disk Chroma store per
    config.yaml; only collection_name differs between them)."""
    fingerprint = f"{cfg.paths.chroma_dir}::{cfg.index.collection_name}"
    if fingerprint not in _collections:
        if fingerprint not in _chroma_clients:
            _chroma_clients[fingerprint] = chromadb.PersistentClient(path=str(cfg.paths.chroma_dir))
        _collections[fingerprint] = _chroma_clients[fingerprint].get_or_create_collection(
            name=cfg.index.collection_name
        )
    return _collections[fingerprint]


def _embedding_client_for(cfg: Config) -> OpenAI:
    """Return the (cached) embedding API client for this cfg's credentials.
    Fingerprinted on (api_key, base_url) so pointing cfg at a different
    embedding endpoint doesn't silently keep reusing a client built for
    the old one."""
    fingerprint = (cfg.embedding_api_key, cfg.embedding_base_url)
    if fingerprint not in _embedding_clients:
        _embedding_clients[fingerprint] = OpenAI(
            api_key=cfg.embedding_api_key, base_url=cfg.embedding_base_url
        )
    return _embedding_clients[fingerprint]


def reset_clients() -> None:
    """Drop every cached client/collection handle.

    The fingerprint-keyed caches above already handle the common case of
    switching cfg mid-process correctly on their own (dev/full simply get
    different cache slots). reset_clients() is a blunt, unambiguous
    "forget everything, start fresh" escape hatch on top of that --
    useful e.g. after a `--rebuild` recreated a collection out from under
    an already-cached handle, or just to release held connections/file
    handles between test runs. In a long-running process that switches
    dev/full mode repeatedly (a test session, a notebook), calling this
    explicitly is cheap insurance even though it's not strictly required
    for correctness given the fingerprinting above.
    """
    _chroma_clients.clear()
    _collections.clear()
    _embedding_clients.clear()


def vector_search(query: str, cfg: Config, top_k: int) -> list[Hit]:
    """Embed `query` and return its top_k nearest children from the vector
    index, nearest (most similar) first.

    Score conversion: this project's Chroma collections use the default
    'l2' HNSW space (squared L2 distance -- confirmed via collection
    configuration inspection), and BAAI/bge-m3 embeddings are
    unit-normalized (confirmed empirically: ||v|| ~= 1.0 on a real query
    vector). For unit vectors, squared L2 distance and cosine similarity
    relate exactly by d^2 = 2 - 2*cos_sim, so `1 - distance / 2` recovers
    cosine similarity (bigger = more similar) from Chroma's raw distance
    (smaller = more similar). This keeps Hit.score meaning "bigger is
    better" consistently everywhere -- important once STEP4 needs to
    reason about vector scores and BM25 scores together (RRF sidesteps
    needing them on the same scale, but rerank's score comparisons and any
    manual debugging both benefit from one consistent convention).
    """
    collection = _collection_for(cfg)
    embedding_client = _embedding_client_for(cfg)

    response = embedding_client.embeddings.create(model=cfg.models.embedding, input=[query])
    query_vector = response.data[0].embedding

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[Hit] = []
    for cid, text, metadata, distance in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append(
            Hit(
                cid=cid,
                text=text,
                page_no=metadata["page_no"],
                score=1.0 - distance / 2.0,
                parent_id=metadata["parent_id"],
            )
        )
    return hits


def search(query: str, cfg: Config) -> list[Hit]:
    """The project's single retrieval entry point. Every ablation
    configuration (baseline/hybrid/rerank/full) runs through this exact
    function -- only cfg.retrieval.* flags differ between calls. See the
    module docstring for why the branch structure below must not be
    restructured when STEP4/5 fill in the TODOs.
    """
    hits = vector_search(query, cfg, top_k=cfg.retrieval.recall_top_k)

    if cfg.retrieval.use_bm25:
        pass  # TODO STEP4: BM25 recall (src/ingest/tokenizer.tokenize) + RRF fusion with the hits above

    if cfg.retrieval.use_rerank:
        pass  # TODO STEP4: Cross-Encoder rerank of the (possibly fused) candidates

    hits = hits[: cfg.retrieval.final_top_k]

    if cfg.retrieval.use_small_to_big:
        pass  # TODO STEP5: swap each child Hit for its parent, dedup by parent_id, keep max score

    return hits
