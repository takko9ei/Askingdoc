"""
Reciprocal Rank Fusion (RRF) -- combines two independently-ranked
candidate lists (dense/vector, sparse/BM25) into one ranked list, using
each document's RANK in each list rather than its raw score.

Why rank and not score: a dense hit carries a cosine-similarity-derived
score (roughly 0-1, see search.py's vector_search docstring for the exact
conversion), while a BM25 hit carries an unbounded raw score whose
magnitude depends on term rarity and corpus size (see the idf inspection
in this project's STEP4 discussion -- values ranged from ~0.5 to ~2.0 on
just a 20-doc sample, and grow with corpus size). There's no principled
way to add "0.65" and "7.3" together and have the sum mean anything
without normalizing both onto a shared scale first -- and normalization
is itself fragile: min-max normalization shifts every time the candidate
set changes (today's max score isn't tomorrow's), and z-scoring needs a
stable mean/std that a small top-20 candidate list doesn't reliably
provide. Rank sidesteps the whole problem: "rank 1" means the same thing
-- best result in this particular list -- no matter what scoring function
produced it underneath.
"""

from __future__ import annotations

from src.retrieval.search import Hit


def rrf_fuse(dense_hits: list[Hit], sparse_hits: list[Hit], k: int) -> list[Hit]:
    """Fuse two ranked Hit lists via score(d) = sum(1 / (k + rank_r(d)))
    over every list r that contains d (rank is 1-indexed within each
    list). A document appearing in BOTH lists accumulates two positive
    terms and naturally outranks a document appearing in only one -- this
    is what makes RRF a "consensus" fusion rather than a plain
    concatenation: agreement between the two retrieval paths is rewarded
    without either path needing to "trust" the other's score.

    Returns new Hit objects, RRF score in place of the original score;
    text/page_no/parent_id are copied from whichever list first produced
    that cid (dense checked first -- an arbitrary but harmless tie-break,
    since both lists describe the same underlying child and must agree on
    those fields regardless of which one supplied them).
    """
    rrf_scores: dict[str, float] = {}
    hit_by_cid: dict[str, Hit] = {}

    for hits in (dense_hits, sparse_hits):
        for rank, hit in enumerate(hits, start=1):
            rrf_scores[hit.cid] = rrf_scores.get(hit.cid, 0.0) + 1.0 / (k + rank)
            hit_by_cid.setdefault(hit.cid, hit)

    fused = [
        Hit(
            cid=cid,
            text=hit_by_cid[cid].text,
            page_no=hit_by_cid[cid].page_no,
            score=score,
            parent_id=hit_by_cid[cid].parent_id,
        )
        for cid, score in rrf_scores.items()
    ]
    fused.sort(key=lambda h: -h.score)
    return fused
