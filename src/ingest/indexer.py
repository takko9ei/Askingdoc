"""
Build the two retrieval indices from children.jsonl (RAG pipeline: step
between chunking and search -- turns static text into searchable state).

Vector index (ChromaDB): each child's text is embedded and stored with
metadata (cid, parent_id, page_no) in a collection named
config.index.collection_name -- already dev/full-isolated by the
{suffix} mechanism in src/config.py, so running this in dev mode can never
overwrite the full collection's data or vice versa.

BM25 index (rank_bm25): each child's text is tokenized (via the shared
src/ingest/tokenizer.py, which STEP4's search-time code will import too)
and the tokenized corpus is pickled to config.paths.bm25. Only vector
search is wired into search() today -- BM25's actual retrieval logic is
STEP4's job; this module only produces the artifact it will read.

Usage:
    python -m src.ingest.indexer               # incremental build, all children
    python -m src.ingest.indexer --limit 20     # small-scale smoke test
    python -m src.ingest.indexer --rebuild      # wipe and rebuild both indices
"""

from __future__ import annotations

import argparse
import json
import pickle
import time

import chromadb
from openai import OpenAI
from tqdm import tqdm

from src.config import Config, load_config
from src.ingest.chunker import estimate_tokens
from src.ingest.tokenizer import tokenize

BATCH_SIZE = 64
MAX_RETRIES = 3


def _embed_with_retry(client: OpenAI, model: str, texts: list[str]) -> list[list[float]] | None:
    """Call the embeddings API for one batch, retrying up to MAX_RETRIES
    times with exponential backoff (1s, 2s, 4s) on failure.

    Returns None (instead of raising) once retries are exhausted, so the
    caller can skip just this one batch and keep the rest of the run
    going -- a transient failure on batch 5 of 30 shouldn't cost the
    embeddings already successfully stored from batches 1-4.
    """
    delay = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: any
            # failure mode (rate limit, timeout, connection reset, malformed
            # response) should trigger the same retry-then-skip behavior.
            print(f"  [embed] attempt {attempt}/{MAX_RETRIES} failed: {exc!r}")
            if attempt < MAX_RETRIES:
                time.sleep(delay)
                delay *= 2
    return None


def build_vector_index(cfg: Config, children: list[dict], rebuild: bool = False) -> dict:
    """Embed and store every child in `children` into the Chroma collection
    named cfg.index.collection_name. Unless rebuild=True, children whose
    cid is already present in the collection are skipped -- this is what
    makes re-running this script after a partial/failed run cheap instead
    of re-spending API budget on work that already succeeded.
    """
    cfg.paths.chroma_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(cfg.paths.chroma_dir))

    if rebuild:
        try:
            chroma_client.delete_collection(name=cfg.index.collection_name)
        except Exception:
            pass  # collection didn't exist yet on a first-ever run -- nothing to delete
    collection = chroma_client.get_or_create_collection(name=cfg.index.collection_name)

    # include=[] means "just give me the ids" -- skips pulling back every
    # stored embedding/document/metadata just to compute a membership set,
    # which matters once the full collection has thousands of entries.
    existing_cids = set() if rebuild else set(collection.get(include=[])["ids"])

    to_index = [c for c in children if c["cid"] not in existing_cids]
    skipped = len(children) - len(to_index)

    embedding_client = OpenAI(api_key=cfg.embedding_api_key, base_url=cfg.embedding_base_url)

    indexed = 0
    failed = 0
    failed_cids: list[str] = []
    total_tokens = 0.0

    batches = [to_index[i : i + BATCH_SIZE] for i in range(0, len(to_index), BATCH_SIZE)]
    for batch in tqdm(batches, desc="embedding batches", disable=not batches):
        texts = [c["text"] for c in batch]
        vectors = _embed_with_retry(embedding_client, cfg.models.embedding, texts)
        if vectors is None:
            failed += len(batch)
            failed_cids.extend(c["cid"] for c in batch)
            continue

        collection.add(
            ids=[c["cid"] for c in batch],
            embeddings=vectors,
            # `documents` stores the raw text alongside the vector -- this
            # lets search() read a hit's text straight out of the Chroma
            # query result with no separate children.jsonl lookup needed.
            documents=texts,
            # cid is redundant with the id list above, but is kept in
            # metadata too because Chroma's query results hand back
            # metadatas dicts directly -- callers that only look at
            # metadata (not the parallel ids list) still get it for free.
            metadatas=[
                {"cid": c["cid"], "parent_id": c["parent_id"], "page_no": c["page_no"]} for c in batch
            ],
        )
        indexed += len(batch)
        total_tokens += sum(estimate_tokens(t) for t in texts)

    if failed_cids:
        print(f"  [embed] FAILED cids after {MAX_RETRIES} retries: {failed_cids}")

    return {
        "indexed": indexed,
        "skipped": skipped,
        "failed": failed,
        "estimated_tokens": total_tokens,
        "collection_count": collection.count(),
    }


def build_bm25_index(cfg: Config, children: list[dict]) -> dict:
    """Tokenize every child's text and pickle {cids, corpus_tokens} to
    cfg.paths.bm25.

    Deliberately NOT pickling a rebuilt BM25Okapi object:
      (a) BM25Okapi precomputes IDF tables / doc-frequency stats that are
          cheap to recompute and would just bloat the file for no benefit
      (b) a pickled class instance is fragile across rank_bm25 version
          upgrades (an internal attribute rename can break unpickling
          silently or loudly); a plain {list, list} dict has neither risk
    Rebuilding BM25Okapi from these tokens at search time costs a fraction
    of a second even at full-document scale -- a good trade for a small,
    stable, human-inspectable file.
    """
    cids = [c["cid"] for c in children]
    corpus_tokens = [tokenize(c["text"]) for c in children]

    cfg.paths.bm25.parent.mkdir(parents=True, exist_ok=True)
    with cfg.paths.bm25.open("wb") as f:
        pickle.dump({"cids": cids, "corpus_tokens": corpus_tokens}, f)

    avg_tokens = sum(len(t) for t in corpus_tokens) / len(corpus_tokens) if corpus_tokens else 0.0
    return {"documents": len(cids), "avg_tokens_per_doc": avg_tokens}


def _print_stats(cfg: Config, vector_stats: dict, bm25_stats: dict, elapsed: float) -> None:
    mode = "DEV" if cfg.is_dev else "FULL"
    print("=" * 60)
    print(f"indexer.py stats -- mode: {mode}")
    print("=" * 60)
    print(f"  chroma collection        : {cfg.index.collection_name}")
    print(f"  vector -- indexed        : {vector_stats['indexed']}")
    print(f"  vector -- skipped (dupe) : {vector_stats['skipped']}")
    print(f"  vector -- failed         : {vector_stats['failed']}")
    print(f"  vector -- collection size: {vector_stats['collection_count']}")
    print(f"  estimated tokens sent    : {vector_stats['estimated_tokens']:.0f}")
    print(f"  bm25 -- documents        : {bm25_stats['documents']}")
    print(f"  bm25 -- avg tokens/doc   : {bm25_stats['avg_tokens_per_doc']:.1f}")
    print(f"  bm25 path                : {cfg.paths.bm25}")
    print(f"  elapsed                  : {elapsed:.1f}s")
    print("=" * 60)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build vector (Chroma) and BM25 indices from children.jsonl.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Clear and rebuild both indices from scratch (default: incremental, skips already-indexed cids).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N children -- for a cheap smoke test before running the full set.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    cfg = load_config()
    cfg.describe()

    with cfg.paths.children.open("r", encoding="utf-8") as f:
        children = [json.loads(line) for line in f]
    if args.limit is not None:
        children = children[: args.limit]
        print(f"[indexer] --limit {args.limit}: only processing the first {len(children)} children")

    start = time.monotonic()
    vector_stats = build_vector_index(cfg, children, rebuild=args.rebuild)
    bm25_stats = build_bm25_index(cfg, children)
    elapsed = time.monotonic() - start

    _print_stats(cfg, vector_stats, bm25_stats, elapsed)


if __name__ == "__main__":
    main()
