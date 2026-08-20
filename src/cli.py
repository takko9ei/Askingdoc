"""
Interactive question-answering loop.

This is where playbook's "self-test with 5 real questions" step happens --
not measuring metrics (that's STEP3's eval/harness.py), but building a gut
feel for whether retrieved passages are actually relevant and whether the
answer reads as grounded rather than made up. That judgment call needs a
human eyeballing real output; this file exists to make that easy.

Usage: python -m src.cli
"""

from __future__ import annotations

from src.config import load_config
from src.generation.answerer import answer


def _print_result(result: dict) -> None:
    print()
    print("-" * 60)
    print(result["answer"])
    print("-" * 60)
    print("检索到的片段:")
    for hit in result["hits"]:
        preview = hit.text[:40].replace("\n", " ")
        print(f"  [p.{hit.page_no}] score={hit.score:.3f}  {preview!r}")


def main() -> None:
    cfg = load_config()
    print("Askingdoc CLI -- 输入问题回车提问 | ':config' 查看当前配置 | ':quit' 退出")
    cfg.describe()

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break

        if not query:
            continue
        if query == ":quit":
            print("再见")
            break
        if query == ":config":
            cfg.describe()
            continue

        result = answer(query, cfg)
        _print_result(result)


if __name__ == "__main__":
    main()
