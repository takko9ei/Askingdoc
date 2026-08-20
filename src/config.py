"""
Central configuration loader for Askingdoc.

Design contract (see CLAUDE.md sections 2 & 7 for the "why" in more depth):

- Every tunable parameter lives in config.yaml; only secrets (API keys) live
  in .env, and .env is never merged into config.yaml or committed to git.
- config.yaml embeds a literal "{suffix}" placeholder inside path and
  collection-name strings. This module resolves that placeholder exactly
  once, based on dev.enabled, so every other module in the project always
  reads an already-concrete path / collection name and never has to know
  that dev/full switching exists at all.
- load_config(overrides) lets any caller override individual config keys
  by a dotted path (e.g. "retrieval.use_bm25" or "dev.enabled"). This one
  mechanism backs two otherwise-unrelated features:
    1. Ablation experiments: eval/configs.py flips retrieval.use_bm25 /
       use_rerank / use_small_to_big to produce the four ablation rows
       without four copies of search().
    2. Dev -> full switching (STEP6): flipping dev.enabled re-derives
       every path and collection name for the full dataset, through this
       same function.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Anchor every relative path in config.yaml to the project root (the
# directory containing this file's parent), not to whatever directory the
# interpreter happens to be launched from -- otherwise `python -m src.cli`
# from a different cwd would silently read/write the wrong storage/ files.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_YAML_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"

# Keys under `paths:` in config.yaml. Listed explicitly (rather than just
# "every key under paths") so a typo'd new path key in config.yaml fails
# loudly here instead of silently staying a raw string with no Path type.
_PATH_KEYS = ("pdf", "blocks", "parents", "children", "chroma_dir", "bm25")


@dataclass
class DevConfig:
    enabled: bool
    max_pages: int
    suffix: str


@dataclass
class PathsConfig:
    pdf: Path
    blocks: Path
    parents: Path
    children: Path
    chroma_dir: Path
    bm25: Path


@dataclass
class IndexConfig:
    collection_name: str


@dataclass
class ChunkingConfig:
    parent_size: int
    child_size: int
    overlap: int


@dataclass
class RetrievalConfig:
    use_bm25: bool
    use_rerank: bool
    use_small_to_big: bool
    rrf_k: int
    recall_top_k: int
    final_top_k: int
    rerank_batch: int


@dataclass
class ModelsConfig:
    embedding: str
    embedding_dim: int
    llm: str
    reranker: str


@dataclass
class Config:
    dev: DevConfig
    paths: PathsConfig
    index: IndexConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    models: ModelsConfig
    # Embedding and LLM are frequently different providers (different
    # pricing/rate-limit/latency profiles), so each gets its own key/url
    # pair rather than sharing one -- see .env.example for the concrete
    # names read from .env.
    embedding_api_key: str | None
    embedding_base_url: str | None
    llm_api_key: str | None
    llm_base_url: str | None

    @property
    def is_dev(self) -> bool:
        """True when running against the 150-page dev slice instead of the
        full document. Modules should branch on this only for things the
        suffixed paths/collection name can't already express (e.g. STEP1's
        "only parse the first max_pages pages" loop bound) -- for storage
        locations, just use self.paths.* / self.index.collection_name
        directly, they're already dev/full-correct."""
        return self.dev.enabled

    def describe(self) -> None:
        """Print an unambiguous snapshot of the active mode.

        Dev and full run through the exact same code path with no branch
        visible in the source -- the only way to know "am I about to touch
        150 pages or the full 500?" at a glance is to print it. Every
        entry-point script (parser/chunker/indexer/cli/harness) should call
        this once at startup so a stale terminal / wrong flag is obvious
        immediately instead of showing up as a confusing result an hour
        later.
        """
        mode = "DEV" if self.is_dev else "FULL"
        print("=" * 60)
        print(f"Askingdoc config snapshot -- mode: {mode}")
        print("=" * 60)
        print(f"  max_pages        : {self.dev.max_pages if self.is_dev else 'ALL'}")
        print(f"  suffix           : {self.dev.suffix!r}")
        print(f"  chroma collection: {self.index.collection_name}")
        print(f"  bm25 path        : {self.paths.bm25}")
        print(f"  blocks path      : {self.paths.blocks}")
        print(f"  parents path     : {self.paths.parents}")
        print(f"  children path    : {self.paths.children}")
        print(
            "  retrieval flags  : "
            f"use_bm25={self.retrieval.use_bm25} "
            f"use_rerank={self.retrieval.use_rerank} "
            f"use_small_to_big={self.retrieval.use_small_to_big}"
        )
        print(f"  embedding model  : {self.models.embedding} (dim={self.models.embedding_dim})")
        print(f"  llm model        : {self.models.llm}")
        print(f"  reranker model   : {self.models.reranker}")
        embed_key_status = "set" if self.embedding_api_key else "MISSING"
        llm_key_status = "set" if self.llm_api_key else "MISSING"
        print(f"  embedding secret : api_key={embed_key_status} base_url={self.embedding_base_url or 'MISSING'}")
        print(f"  llm secret       : api_key={llm_key_status} base_url={self.llm_base_url or 'MISSING'}")
        print("=" * 60)


def _dot_get(d: dict, dotted_key: str) -> Any:
    """Walk a dotted path like 'retrieval.use_bm25' through nested dicts."""
    node = d
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(
                f"Override key {dotted_key!r} does not exist in config.yaml "
                f"(failed at {part!r}) -- check for a typo."
            )
        node = node[part]
    return node


def _dot_set(d: dict, dotted_key: str, value: Any) -> None:
    """Set a value at a dotted path, requiring every segment -- including
    the leaf -- to already exist in config.yaml.

    Overrides exist to flip *existing* switches (ablation flags, dev.enabled),
    never to introduce a brand-new key. Requiring pre-existence turns a typo
    like "retrieval.use_bm26" into an immediate KeyError instead of a dead
    key that search() silently never reads, which would otherwise look like
    a config change that quietly did nothing.
    """
    parts = dotted_key.split(".")
    node = d
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(
                f"Override key {dotted_key!r} does not exist in config.yaml "
                f"(failed at {part!r}) -- check for a typo."
            )
        node = node[part]
    leaf = parts[-1]
    if leaf not in node:
        raise KeyError(
            f"Override key {dotted_key!r} does not exist in config.yaml -- "
            f"check for a typo."
        )
    node[leaf] = value


def _resolve_suffix(node: Any, suffix: str) -> Any:
    """Recursively replace the literal placeholder "{suffix}" inside every
    string value of the parsed YAML tree.

    Deliberately generic (walks the whole tree) rather than hardcoded to
    "only paths.* and index.collection_name": any future suffixed field
    (e.g. a sweep experiment's collection name in STEP8) gets the same
    substitution for free instead of requiring this function to be edited
    every time a new {suffix} usage is added.
    """
    if isinstance(node, dict):
        return {k: _resolve_suffix(v, suffix) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_suffix(v, suffix) for v in node]
    if isinstance(node, str):
        return node.replace("{suffix}", suffix)
    return node


def load_config(overrides: dict[str, Any] | None = None) -> Config:
    """Load config.yaml + .env into a typed, suffix-resolved Config object.

    `overrides` takes dotted-path keys (e.g. {"retrieval.use_bm25": True})
    and is applied to the raw YAML dict BEFORE suffix resolution. Order
    matters: an override can itself be "dev.enabled", which changes what
    the suffix resolves to, so suffix resolution must see the overridden
    value, not the file's original one.
    """
    with CONFIG_YAML_PATH.open("r", encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f)

    for dotted_key, value in (overrides or {}).items():
        _dot_set(raw, dotted_key, value)

    suffix = raw["dev"]["suffix"] if raw["dev"]["enabled"] else ""
    raw = _resolve_suffix(raw, suffix)

    dev = DevConfig(**raw["dev"])
    paths = PathsConfig(**{key: PROJECT_ROOT / raw["paths"][key] for key in _PATH_KEYS})
    index = IndexConfig(**raw["index"])
    chunking = ChunkingConfig(**raw["chunking"])
    retrieval = RetrievalConfig(**raw["retrieval"])
    models = ModelsConfig(**raw["models"])

    # Secrets never live in config.yaml (and therefore never in git) -- they
    # only come from environment variables, loaded from .env if present.
    # load_dotenv() is a harmless no-op (returns False) when .env doesn't
    # exist yet, e.g. before STEP0's landmine-3 API check has been done.
    # Embedding and LLM each get their own key/url pair since they're
    # frequently different providers -- see .env.example.
    load_dotenv(ENV_PATH)
    embedding_api_key = os.getenv("EMBEDDING_API_KEY")
    embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
    llm_api_key = os.getenv("LLM_API_KEY")
    llm_base_url = os.getenv("LLM_BASE_URL")

    return Config(
        dev=dev,
        paths=paths,
        index=index,
        chunking=chunking,
        retrieval=retrieval,
        models=models,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
    )


if __name__ == "__main__":
    # Manual smoke test: `python -m src.config`
    load_config().describe()
