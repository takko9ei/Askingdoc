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

STEP4 filled in the use_bm25 branch: bm25_search() (sparse retrieval) plus
src/retrieval/fusion.py's rrf_fuse() combine with the dense vector_search()
results above. This slice fills in use_rerank: src/retrieval/reranker.py's
rerank() re-scores the (possibly fused) candidates with a Cross-Encoder.
use_small_to_big stays a `pass` TODO for STEP5.

Note on adapting the sketch skeleton to this project's actual Config shape:
the playbook's pseudocode writes `cfg.use_bm25` / `cfg.recall_top_k`
directly; this project's real Config (src/config.py, STEP0) nests those
under `cfg.retrieval.*`, so the branches below read `cfg.retrieval.use_bm25`
etc. instead. The BRANCH STRUCTURE (four sequential ifs, in this order) is
unchanged -- only the attribute paths were corrected to match the actual
typed Config dataclass.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass

import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi

from src.config import Config, load_config
from src.ingest.tokenizer import tokenize

# NOTE: src.retrieval.fusion.rrf_fuse and src.retrieval.reranker.rerank are
# both imported locally inside the functions that use them (search(),
# explain_fusion(), explain_rerank()), not at module level here. Both
# fusion.py and reranker.py import Hit from this module -- a top-level
# import in both directions would be a circular import. Deferring this one
# side to function-call time breaks the cycle without needing to move Hit
# into a third module.


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
    # Set only after reranking (src/retrieval/reranker.py); None otherwise.
    # Kept SEPARATE from `score` (rather than overwriting it) so a reranked
    # Hit still carries its pre-rerank score for comparison/debugging --
    # explain_rerank() below prints both side by side.
    rerank_score: float | None = None


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
_bm25_indices: dict[str, BM25Okapi] = {}
_bm25_cids: dict[str, list[str]] = {}
_children_maps: dict[str, dict[str, dict]] = {}


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


def _bm25_index_for(cfg: Config) -> tuple[BM25Okapi, list[str]]:
    """Return the (cached) BM25Okapi index + its parallel cids list for
    cfg.paths.bm25.

    Rebuilding BM25Okapi from the pickled tokenized corpus is cheap (a
    fraction of a second even at full-document scale -- see
    src/ingest/indexer.py's build_bm25_index docstring for why only the
    tokenized corpus is persisted, not a pickled BM25Okapi object), but
    still wasteful to redo on every single query in a process that answers
    many questions in a row -- same rationale as the Chroma/embedding
    caches above. Fingerprinted on cfg.paths.bm25 (already dev/full-
    isolated via the {suffix} mechanism), so this coexists safely with a
    dev cache slot and a full cache slot in the same process.
    """
    fingerprint = str(cfg.paths.bm25)
    if fingerprint not in _bm25_indices:
        with cfg.paths.bm25.open("rb") as f:
            data = pickle.load(f)
        _bm25_indices[fingerprint] = BM25Okapi(data["corpus_tokens"])
        _bm25_cids[fingerprint] = data["cids"]
    return _bm25_indices[fingerprint], _bm25_cids[fingerprint]


def _children_map_for(cfg: Config) -> dict[str, dict]:
    """Return the (cached) cid -> child-record map for cfg.paths.children.

    BM25's get_scores() only knows about token overlap -- it returns a
    score per corpus position, not text/page_no/parent_id. This map is
    what turns a bare cid back into a full Hit, mirroring what ChromaDB's
    metadata/documents fields already give vector_search for free (the
    vector index stores that data alongside each embedding; the BM25
    pickle deliberately does not, see indexer.py, so this file has to look
    it up separately).
    """
    fingerprint = str(cfg.paths.children)
    if fingerprint not in _children_maps:
        with cfg.paths.children.open("r", encoding="utf-8") as f:
            children = [json.loads(line) for line in f]
        _children_maps[fingerprint] = {c["cid"]: c for c in children}
    return _children_maps[fingerprint]


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
    _bm25_indices.clear()
    _bm25_cids.clear()
    _children_maps.clear()


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


def bm25_search(query: str, cfg: Config, top_k: int) -> list[Hit]:
    """Keyword search over the same child corpus vector_search uses,
    returning the top_k highest-BM25-score children.

    【关键约束】`query` is tokenized with the exact same tokenize() function
    (src/ingest/tokenizer.py) that src/ingest/indexer.py used to build
    corpus_tokens at index time. BM25 matches purely on token-string
    equality -- if this called a different tokenizer, or the same
    tokenizer with different settings, tokens that are "the same word" to
    a human could come out as different strings here vs. in the index,
    and every query would silently underperform with no error anywhere
    to explain why. This is not a style preference; it's a correctness
    requirement enforced by importing the one shared function rather than
    writing tokenization logic locally.

    Score conversion: BM25's raw score is unbounded (depends on term
    rarity and document length -- see this project's own STEP4 idf
    inspection, where values ranged ~0.5 to ~2.0 on a 20-doc sample and
    grow with corpus size), so unlike vector_search's score (normalized to
    a ~0-1 cosine similarity), Hit.score here is the raw BM25 score
    AS-IS. This is fine specifically because BM25 hits are never compared
    to vector hits by score -- rrf_fuse() combines the two lists by RANK,
    exactly to avoid needing them on a common scale.
    """
    bm25_index, cids = _bm25_index_for(cfg)
    children_map = _children_map_for(cfg)

    query_tokens = tokenize(query)
    scores = bm25_index.get_scores(query_tokens)

    ranked = sorted(zip(cids, scores), key=lambda pair: -pair[1])[:top_k]

    hits: list[Hit] = []
    for cid, score in ranked:
        child = children_map.get(cid)
        if child is None:
            # Defensive only -- bm25.pkl and children.jsonl are both built
            # from the same children list by indexer.py in the same run,
            # so they should never disagree. Skip rather than crash if
            # they ever do drift out of sync (e.g. a stale bm25.pkl from
            # before a chunking parameter change).
            continue
        hits.append(
            Hit(
                cid=cid,
                text=child["text"],
                page_no=child["page_no"],
                score=float(score),
                parent_id=child["parent_id"],
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
        from src.retrieval.fusion import rrf_fuse  # local import: see module import comment

        sparse_hits = bm25_search(query, cfg, top_k=cfg.retrieval.recall_top_k)
        hits = rrf_fuse(hits, sparse_hits, k=cfg.retrieval.rrf_k)
        # rrf_fuse returns the UNION of both lists (up to 2 * recall_top_k
        # if dense and sparse barely overlap), not capped at recall_top_k --
        # truncate back down here so recall_top_k means the same thing
        # ("how many candidates move on to ranking") whether or not BM25
        # fusion is on. Without this, enabling both use_bm25 and use_rerank
        # together silently reranks up to ~2x as many pairs as intended
        # (observed: 38 pairs instead of 20 on a real query), which is
        # exactly the "候选数太大" cost this STEP's 【输出后请解释】 answer #3
        # discusses.
        hits = hits[: cfg.retrieval.recall_top_k]

    if cfg.retrieval.use_rerank:
        from src.retrieval.reranker import rerank  # local import: see module import comment

        hits = rerank(query, hits, cfg)

    hits = hits[: cfg.retrieval.final_top_k]

    if cfg.retrieval.use_small_to_big:
        pass  # TODO STEP5: swap each child Hit for its parent, dedup by parent_id, keep max score

    return hits


def explain_fusion(query: str, cfg: Config | None = None) -> None:
    """Debug helper: print the dense (vector) top-10, the sparse (BM25)
    top-10, and the RRF-fused top-10 side by side, labeling each fused
    result with which path(s) it came from and its original rank(s)
    there.

    Use this on a keyword_exact golden_qa question to build direct
    intuition for what RRF actually does -- e.g. "识别优先级设置"
    (docs/capability_log.md): a menu-index page PyMuPDF mangled into
    unparseable prose, where dense search finds nothing usable but BM25's
    literal "优先级设置" substring match should surface it, and the fused
    ranking should show that rescue happening.
    """
    cfg = cfg or load_config()
    from src.retrieval.fusion import rrf_fuse  # local import: see module import comment

    dense_hits = vector_search(query, cfg, top_k=10)
    sparse_hits = bm25_search(query, cfg, top_k=10)
    fused = rrf_fuse(dense_hits, sparse_hits, k=cfg.retrieval.rrf_k)[:10]

    dense_rank = {h.cid: r for r, h in enumerate(dense_hits, start=1)}
    sparse_rank = {h.cid: r for r, h in enumerate(sparse_hits, start=1)}

    print(f"--- explain_fusion: {query!r} ---")

    print("\n[dense/vector top-10]")
    for r, h in enumerate(dense_hits, start=1):
        print(f"  #{r:2d} {h.cid}  page={h.page_no:>3}  score={h.score:.3f}")

    print("\n[sparse/BM25 top-10]")
    for r, h in enumerate(sparse_hits, start=1):
        print(f"  #{r:2d} {h.cid}  page={h.page_no:>3}  score={h.score:.3f}")

    print(f"\n[fused top-10 (RRF, k={cfg.retrieval.rrf_k})]")
    for r, h in enumerate(fused, start=1):
        d_rank = dense_rank.get(h.cid)
        s_rank = sparse_rank.get(h.cid)
        sources = []
        if d_rank:
            sources.append(f"dense#{d_rank}")
        if s_rank:
            sources.append(f"sparse#{s_rank}")
        print(f"  #{r:2d} {h.cid}  page={h.page_no:>3}  rrf_score={h.score:.4f}  from={'+'.join(sources)}")


def explain_rerank(query: str, cfg: Config | None = None) -> None:
    """Debug helper: run the SAME recall pipeline search() itself would use
    (vector-only, or vector+BM25 fusion if cfg.retrieval.use_bm25 is on),
    then rerank it, and print a before/after top-10 comparison -- original
    rank, new rank, original score, rerank_score -- with the biggest
    rank-change entries called out separately.

    Use this on a question where the recall stage's own ordering looked
    "off" -- e.g. docs/capability_log.md's "怎么清理CMOS" and "如何显示拍摄
    设置的列表", both of which dropped from rank 1 (vector-only) to rank
    2-3 once RRF fusion mixed in BM25's disagreeing order (see STEP4
    Slice 4-1's capability_log entry) -- to see concretely whether the
    Cross-Encoder's joint (query, document) reading restores them to
    rank 1, independent of what either recall path's score said.
    """
    cfg = cfg or load_config()
    from src.retrieval.reranker import rerank  # local import: see module import comment

    hits = vector_search(query, cfg, top_k=cfg.retrieval.recall_top_k)
    if cfg.retrieval.use_bm25:
        from src.retrieval.fusion import rrf_fuse  # local import: see module import comment

        sparse_hits = bm25_search(query, cfg, top_k=cfg.retrieval.recall_top_k)
        hits = rrf_fuse(hits, sparse_hits, k=cfg.retrieval.rrf_k)
        hits = hits[: cfg.retrieval.recall_top_k]  # mirror search()'s post-fusion truncation, see its comment

    before = hits[:10]
    reranked = rerank(query, hits, cfg)[:10]
    before_rank = {h.cid: r for r, h in enumerate(before, start=1)}

    print(f"--- explain_rerank: {query!r} ---")

    print("\n[before rerank, top-10]")
    for r, h in enumerate(before, start=1):
        print(f"  #{r:2d} {h.cid}  page={h.page_no:>3}  score={h.score:.3f}")

    print("\n[after rerank, top-10]")
    changes: list[tuple[int | None, Hit, int | None, int]] = []
    for new_rank, h in enumerate(reranked, start=1):
        old_rank = before_rank.get(h.cid)
        delta = (old_rank - new_rank) if old_rank else None
        changes.append((delta, h, old_rank, new_rank))
        old_rank_str = str(old_rank) if old_rank else "not in top-10"
        print(
            f"  #{new_rank:2d} {h.cid}  page={h.page_no:>3}  old_rank={old_rank_str:>13}  "
            f"score={h.score:.3f}  rerank_score={h.rerank_score:.3f}"
        )

    print("\n[biggest rank changes]")
    ranked_changes = sorted((c for c in changes if c[0] is not None), key=lambda c: -abs(c[0]))
    for delta, h, old_rank, new_rank in ranked_changes[:5]:
        direction = "up" if delta > 0 else "down"
        print(f"  {h.cid} (page={h.page_no}): rank {old_rank} -> {new_rank} ({direction} {abs(delta)})")
