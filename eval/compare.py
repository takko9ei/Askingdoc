"""
eval/compare.py -- read eval/results/{mode}_*.json for one mode and print
the four-row ablation comparison table, plus a copy-pasteable Markdown
table for README's 消融实验结果 section.

Usage: python -m eval.compare --mode dev
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

# Fixed display order -- these are meant to read as a progression (each
# row adds exactly one capability on top of the previous row), not an
# alphabetical or discovery order.
CONFIG_ORDER = ["baseline", "hybrid", "rerank", "full"]

console = Console()


def load_results(mode: str) -> dict[str, dict]:
    """Load whichever eval/results/{mode}_*.json files exist, keyed by
    config name. Missing configs (not yet run) are simply absent from the
    returned dict -- callers render them as "-" rather than erroring, so
    this is usable mid-STEP4 with only baseline+hybrid present."""
    results = {}
    for config_name in CONFIG_ORDER:
        path = RESULTS_DIR / f"{mode}_{config_name}.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                results[config_name] = json.load(f)
    return results


def print_table(results: dict[str, dict], mode: str) -> None:
    table = Table(title=f"Ablation comparison -- mode={mode}")
    for col in ("Config", "Recall@5", "Recall@20", "MRR", "Abstention"):
        table.add_column(col, justify="right" if col != "Config" else "left")

    for config_name in CONFIG_ORDER:
        if config_name not in results:
            table.add_row(config_name, "-", "-", "-", "-", style="dim")
            continue
        overall = results[config_name]["overall"]
        abst = f"{overall['abstention_rate']:.1%}" if overall["abstention_rate"] is not None else "-"
        table.add_row(
            config_name,
            f"{overall['recall_at_5']:.1%}",
            f"{overall['recall_at_20']:.1%}",
            f"{overall['mrr']:.3f}",
            abst,
        )
    console.print(table)


def print_markdown(results: dict[str, dict], mode: str) -> None:
    print(f"\n### 消融实验结果（{mode}）\n")
    print("| 配置 | Recall@5 | Recall@20 | MRR | Abstention |")
    print("|---|---|---|---|---|")
    for config_name in CONFIG_ORDER:
        if config_name not in results:
            print(f"| {config_name} | - | - | - | - |")
            continue
        overall = results[config_name]["overall"]
        abst = f"{overall['abstention_rate']:.1%}" if overall["abstention_rate"] is not None else "-"
        print(
            f"| {config_name} | {overall['recall_at_5']:.1%} | "
            f"{overall['recall_at_20']:.1%} | {overall['mrr']:.3f} | {abst} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ablation results for one mode.")
    parser.add_argument("--mode", choices=["dev", "full"], default="dev")
    args = parser.parse_args()

    results = load_results(args.mode)
    if not results:
        print(f"No results found for mode={args.mode} in {RESULTS_DIR} -- run `python -m eval.harness` first.")
        return

    print_table(results, args.mode)
    print_markdown(results, args.mode)


if __name__ == "__main__":
    main()
