"""
Preset override dicts for the project's four ablation configurations.

Each dict is fed straight into src.config.load_config(overrides) -- this
IS the entire mechanism that turns one search() implementation into four
different behaviors (see src/retrieval/search.py's module docstring for
the other half of this story). Adding a fifth configuration later (e.g. a
STEP8 sweep experiment) means adding one more dict here, never touching
search() or eval/harness.py.
"""

ABLATION_CONFIGS: dict[str, dict] = {
    "baseline": {
        "retrieval.use_bm25": False,
        "retrieval.use_rerank": False,
        "retrieval.use_small_to_big": False,
    },
    "hybrid": {
        "retrieval.use_bm25": True,
        "retrieval.use_rerank": False,
        "retrieval.use_small_to_big": False,
    },
    "rerank": {
        "retrieval.use_bm25": True,
        "retrieval.use_rerank": True,
        "retrieval.use_small_to_big": False,
    },
    "full": {
        "retrieval.use_bm25": True,
        "retrieval.use_rerank": True,
        "retrieval.use_small_to_big": True,
    },
}
