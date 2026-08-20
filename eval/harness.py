"""
eval/harness.py -- retrieval + generation quality evaluation framework.

This is the project's core evaluation asset. Per project discipline it is
LOCKED after this STEP: no later STEP may edit this file or
eval/golden_qa.jsonl, because any change to how a metric is computed makes
the four ablation configs' numbers incomparable across STEPs -- the whole
point of an eval harness is that it stays the same ruler while the thing
being measured (search()) changes underneath it.

What each metric measures and why all four are kept (not just Recall@5):

- Recall@5: the PRIMARY metric -- whether at least one of the top-5
  retrieved children lands on a gold page. This is what a real user
  actually sees, since config.retrieval.final_top_k defaults to 5.
- Recall@20: the retrieval CEILING. The gap between Recall@20 and
  Recall@5 is the diagnostic signal -- big gap means the right chunk IS
  being found somewhere in the first 20 candidates but not ranked into
  the top 5 (a ranking problem, STEP4's rerank is the fix); small gap
  with both low means the right chunk isn't even being found at all (a
  recall problem -- chunking/BM25/embedding is the fix, reranking a list
  that doesn't contain the answer can't help).
- MRR: rewards ranking the correct chunk EARLY. Two runs can have
  identical Recall@20 while one buries the answer at rank 18 and the
  other finds it at rank 2 -- Recall@20 can't tell them apart, MRR can.
- Abstention rate: measured ONLY on `negative` questions (empty
  gold_pages), via the FULL answer() pipeline rather than search() alone,
  because correctly declining to answer is a GENERATION behavior (does
  the anti-hallucination prompt constraint actually hold under a question
  it has no good context for) -- not something a retrieval-only metric
  can observe.

Usage:
    python -m eval.harness --config baseline
    python -m eval.harness --config hybrid --verbose
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from eval.configs import ABLATION_CONFIGS
from src.config import Config, load_config
from src.generation.answerer import answer
from src.retrieval.search import Hit, search

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_QA_PATH = PROJECT_ROOT / "eval" / "golden_qa.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

# The exact refusal string in src/generation/prompts.py's ANSWER_PROMPT is
# "文档中未提及相关信息。" -- "未提及" alone already catches that. The rest
# are fallback variants in case the model paraphrases instead of quoting
# the instructed phrase verbatim (observed to happen occasionally in
# manual testing, see docs/capability_log.md).
ABSTENTION_PATTERNS = ("未提及", "没有提到", "未找到", "无法回答", "无相关信息", "未查到", "无法确定")

console = Console()


def load_golden_qa(cfg: Config) -> list[dict]:
    """Load eval/golden_qa.jsonl, filtered by the active dev/full mode.

    dev.enabled=True runs ONLY in_dev_range=True questions: the 150-page
    dev corpus cannot possibly retrieve content past page 150, so a
    question written against the full document would deterministically
    score 0 for a reason that has nothing to do with retrieval quality --
    filtering keeps the numbers meaningful for whichever corpus is
    actually loaded. dev.enabled=False (STEP6+) runs every question.
    """
    with GOLDEN_QA_PATH.open("r", encoding="utf-8") as f:
        all_questions = [json.loads(line) for line in f]

    if cfg.is_dev:
        questions = [q for q in all_questions if q["in_dev_range"]]
    else:
        questions = all_questions

    mode = "DEV" if cfg.is_dev else "FULL"
    print(f"[harness] loaded {len(all_questions)} total questions, {len(questions)} in scope for mode={mode}")
    return questions


def _first_hit_rank(hits: list[Hit], gold_pages: list[int]) -> int | None:
    """1-indexed rank of the first hit whose page_no is ANY of gold_pages,
    or None if no hit in `hits` lands on a gold page. "Any" (not "all") is
    the deliberate reading of "hit.page_no 在 gold_pages 里" for multi_hop
    questions too -- Recall@k here measures "was relevant content found at
    all", not "was every gold page found"; whether a multi_hop ANSWER is
    fully assembled from all its sources is a generation-quality question,
    out of scope for this retrieval metric.
    """
    for rank, hit in enumerate(hits, start=1):
        if hit.page_no in gold_pages:
            return rank
    return None


def _is_abstention(answer_text: str) -> bool:
    return any(p in answer_text for p in ABSTENTION_PATTERNS)


def evaluate_question(q: dict, retrieval_cfg: Config, production_cfg: Config) -> dict:
    """Evaluate one golden_qa question against one ablation configuration.

    Two different cfg objects, on purpose:
    - retrieval_cfg has retrieval.final_top_k overridden to 20, purely so
      search() returns enough candidates to compute Recall@20 without
      being truncated by the production default of 5.
    - production_cfg is the UNMODIFIED ablation config (final_top_k stays
      whatever config.yaml says, normally 5) -- negative questions must be
      evaluated against what a real user session actually sees, not an
      artificially widened top-20 context that no real query would get.
    See this STEP's 【输出后请解释】 answer #2 for the full reasoning.
    """
    if q["type"] == "negative":
        result = answer(q["question"], production_cfg)
        abstained = _is_abstention(result["answer"])
        return {
            "qid": q["qid"],
            "type": q["type"],
            "question": q["question"],
            "gold_pages": q["gold_pages"],
            "generated_answer": result["answer"],
            "abstained": abstained,
        }

    hits = search(q["question"], retrieval_cfg)
    rank = _first_hit_rank(hits, q["gold_pages"])
    return {
        "qid": q["qid"],
        "type": q["type"],
        "question": q["question"],
        "gold_pages": q["gold_pages"],
        "retrieved_pages_top20": [h.page_no for h in hits],
        "hit_rank": rank,
        "recall_at_5": rank is not None and rank <= 5,
        "recall_at_20": rank is not None,
    }


def _aggregate(results: list[dict]) -> dict:
    """Recall@5 / Recall@20 / MRR over the non-negative subset of
    `results`; abstention_rate over the negative subset. Called both for
    the overall totals and once per `type` group -- a group that's purely
    negative (or purely non-negative) correctly gets 0-length denominators
    on the metric that doesn't apply to it (handled as None/omitted in
    the printed report, not as a misleading 0.0)."""
    retrieval_results = [r for r in results if r["type"] != "negative"]
    negative_results = [r for r in results if r["type"] == "negative"]

    n = len(retrieval_results)
    recall_5 = sum(r["recall_at_5"] for r in retrieval_results) / n if n else 0.0
    recall_20 = sum(r["recall_at_20"] for r in retrieval_results) / n if n else 0.0
    mrr = sum((1.0 / r["hit_rank"]) if r["hit_rank"] else 0.0 for r in retrieval_results) / n if n else 0.0

    n_neg = len(negative_results)
    abstention_rate = (sum(r["abstained"] for r in negative_results) / n_neg) if n_neg else None

    return {
        "n_retrieval_questions": n,
        "recall_at_5": recall_5,
        "recall_at_20": recall_20,
        "mrr": mrr,
        "n_negative_questions": n_neg,
        "abstention_rate": abstention_rate,
    }


def _aggregate_by_type(results: list[dict]) -> dict[str, dict]:
    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)
    return {type_name: _aggregate(rows) for type_name, rows in by_type.items()}


def run_evaluation(config_name: str, verbose: bool = False) -> dict:
    """Run the full (mode-filtered) golden_qa set through one ablation
    configuration; print a report and write eval/results/{mode}_{config}.json.
    """
    base_overrides = dict(ABLATION_CONFIGS[config_name])
    production_cfg = load_config(base_overrides)

    retrieval_overrides = dict(base_overrides)
    retrieval_overrides["retrieval.final_top_k"] = 20
    retrieval_cfg = load_config(retrieval_overrides)

    questions = load_golden_qa(production_cfg)

    start = time.monotonic()
    per_question = [evaluate_question(q, retrieval_cfg, production_cfg) for q in questions]
    elapsed = time.monotonic() - start

    mode = "dev" if production_cfg.is_dev else "full"
    result = {
        "meta": {
            "mode": mode,
            "config_name": config_name,
            "n_questions": len(questions),
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "overall": _aggregate(per_question),
        "by_type": _aggregate_by_type(per_question),
        "per_question": per_question,
    }

    _print_report(result, verbose=verbose)
    _save_result(result)
    return result


def _print_report(result: dict, verbose: bool = False) -> None:
    meta = result["meta"]
    overall = result["overall"]

    console.print(
        f"\n[bold]eval/harness[/bold] -- mode={meta['mode']} config={meta['config_name']} "
        f"({meta['n_questions']} questions, {meta['elapsed_seconds']}s)"
    )

    table = Table(title="Overall")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Recall@5", f"{overall['recall_at_5']:.1%}")
    table.add_row("Recall@20", f"{overall['recall_at_20']:.1%}")
    table.add_row("MRR", f"{overall['mrr']:.3f}")
    if overall["abstention_rate"] is not None:
        table.add_row("Abstention Rate", f"{overall['abstention_rate']:.1%}")
    console.print(table)

    type_table = Table(title="By Type")
    for col in ("Type", "N", "Recall@5", "Recall@20", "MRR", "Abstention"):
        type_table.add_column(col, justify="right" if col != "Type" else "left")
    for type_name, stats in result["by_type"].items():
        n = stats["n_retrieval_questions"] or stats["n_negative_questions"]
        has_retrieval = stats["n_retrieval_questions"] > 0
        type_table.add_row(
            type_name,
            str(n),
            f"{stats['recall_at_5']:.1%}" if has_retrieval else "-",
            f"{stats['recall_at_20']:.1%}" if has_retrieval else "-",
            f"{stats['mrr']:.3f}" if has_retrieval else "-",
            f"{stats['abstention_rate']:.1%}" if stats["abstention_rate"] is not None else "-",
        )
    console.print(type_table)

    if verbose:
        console.print("\n[bold]Per-question detail (--verbose):[/bold]")
        for r in result["per_question"]:
            if r["type"] == "negative":
                status = "abstained" if r["abstained"] else "DID NOT ABSTAIN (hallucination risk)"
                console.print(f"  [{r['qid']}] {r['question']}")
                console.print(f"      -> {status}: {r['generated_answer'][:80]!r}")
            else:
                mark = "HIT@5" if r["recall_at_5"] else ("hit@20" if r["recall_at_20"] else "MISS")
                console.print(
                    f"  [{r['qid']}] {mark:6s} {r['question']}  "
                    f"gold={r['gold_pages']} top5={r['retrieved_pages_top20'][:5]} rank={r['hit_rank']}"
                )


def _save_result(result: dict) -> None:
    meta = result["meta"]
    # dev and full results are written to DIFFERENT filenames on purpose --
    # see this STEP's 【输出后请解释】 answer #4.
    out_path = RESULTS_DIR / f"{meta['mode']}_{meta['config_name']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[harness] results written to {out_path}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the golden_qa evaluation against one ablation config.")
    parser.add_argument("--config", choices=list(ABLATION_CONFIGS.keys()), default="baseline")
    parser.add_argument(
        "--verbose", action="store_true", help="Print each question's top-5 pages vs gold_pages for failure analysis."
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_evaluation(args.config, verbose=args.verbose)


if __name__ == "__main__":
    main()
