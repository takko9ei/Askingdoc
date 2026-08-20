"""
PDF -> page-numbered text blocks (RAG pipeline step (1): parsing).

*** CRITICAL PAGE-NUMBERING RULE (do not "simplify" this) ***
PyMuPDF's internal page index is 0-based. Every block's `page_no` written
to disk is that index + 1, i.e. the real, physical page number a human
would read off the printed PDF. This holds EVEN IN DEV MODE, where only
the first `config.dev.max_pages` pages are processed: we never renumber
pages 1..150 as if they were the whole book. If we did, STEP6's switch to
the full dataset would silently invalidate every `gold_pages` entry in the
evaluation set, because "page 87" would mean two different physical pages
depending on which mode produced it. Page numbers are the one field this
whole project's citation feature depends on, and they can only be captured
here -- there is no way to recover the true page number from plain text
after this step.

Usage:
    python -m src.ingest.parser [--max-pages N]
"""

from __future__ import annotations

import argparse
import json
import re

import pymupdf as fitz  # `import fitz` still works but is deprecated as of PyMuPDF 1.28

from src.config import Config, load_config

# --- Filtering thresholds -------------------------------------------------
# NOT read from config.yaml: this prompt's 改动范围 explicitly forbids
# touching src/config.py, so there is no typed field to hang these off of
# yet. Kept as clearly-named module constants instead, so they're still a
# single easy place to tune. Recommended follow-up (left for manual
# addition, as requested): an `ingest:` section in config.yaml with
# min_block_chars / header_footer_margin_pct / header_footer_min_chars,
# plus a matching IngestConfig dataclass in src/config.py.
MIN_BLOCK_CHARS = 20             # blocks shorter than this are pure noise
HEADER_FOOTER_MARGIN_PCT = 0.05  # top/bottom 5% of page height
HEADER_FOOTER_MIN_CHARS = 50     # blocks inside that margin are kept only if >= this long


def _is_cjk(ch: str) -> bool:
    """True for a CJK ideograph / full-width punctuation character.

    Used only to decide whether two wrapped lines should be joined with
    no space (Chinese has no inter-word spaces) or with a space (English).
    """
    if not ch:
        return False
    return "一" <= ch <= "鿿" or "　" <= ch <= "〿" or "＀" <= ch <= "￯"


def _merge_wrapped_lines(text: str) -> str:
    """Collapse PyMuPDF's within-block line breaks.

    A block's raw text typically looks like "这是一段很长\n的说明文字" -- the
    "\n" there is a TYPOGRAPHIC line break caused by the block wrapping at
    the page/column width, not a semantic paragraph break (PyMuPDF's block
    detection has already grouped genuinely separate paragraphs into
    separate blocks; see the module docstring in chunker.py, STEP2, for how
    that assumption gets used downstream). So within one block we always
    merge every line back into a single logical line:
      - if the previous line ends with "-" and the next starts with an
        ASCII letter, treat it as a hyphenated word broken across the
        wrap ("informa-\ntion" -> "information")
      - if either side of the join is a CJK character, join with no space
        (Chinese doesn't use inter-word spaces)
      - otherwise join with a single space (English word wrap)
    Blank lines inside a block are extraction artifacts, not paragraph
    markers, so they're dropped rather than preserved.
    """
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""
    merged = lines[0]
    for line in lines[1:]:
        if merged.endswith("-") and line[:1].isascii() and line[:1].isalpha():
            merged = merged[:-1] + line
        elif _is_cjk(merged[-1:]) or _is_cjk(line[:1]):
            merged += line
        else:
            merged += " " + line
    return merged


def _clean_block_text(text: str) -> str:
    """Merge wrapped lines, then normalize runs of spaces/tabs to one space.

    Note: we only collapse [ \\t]+, not all whitespace -- _merge_wrapped_lines
    has already turned "\n" into either nothing or a single space, so there's
    no newline left to worry about here, and this ordering keeps the two
    concerns (line-wrap semantics vs. whitespace cosmetics) separate.
    """
    merged = _merge_wrapped_lines(text)
    return re.sub(r"[ \t]+", " ", merged).strip()


def parse_pdf(cfg: Config) -> list[dict]:
    """Parse cfg.paths.pdf into page-numbered, noise-filtered text blocks
    and write them as JSONL to cfg.paths.blocks. Returns the same records
    in memory (useful for quick checks without re-reading the file).
    """
    doc = fitz.open(cfg.paths.pdf)
    total_pages = len(doc)
    pages_to_process = min(cfg.dev.max_pages, total_pages) if cfg.is_dev else total_pages

    kept_blocks: list[dict] = []
    stats = {
        "total_pages_in_pdf": total_pages,
        "pages_processed": pages_to_process,
        "raw_blocks_seen": 0,
        "dropped_non_text_block": 0,
        "dropped_short": 0,
        "dropped_numeric_symbol": 0,
        "dropped_header_footer": 0,
    }

    block_counter = 0
    for page_index in range(pages_to_process):
        page = doc[page_index]
        page_no = page_index + 1  # see module docstring: real, human-facing page number
        page_height = page.rect.height

        # get_text("blocks") returns tuples in the page's natural reading
        # order for simple single-column layouts; multi-column pages (rare
        # in this manual, mostly the spec tables already flagged as a known
        # limitation) can come back in a less intuitive order -- acceptable
        # here since block order doesn't affect page_no correctness, only
        # the order blocks appear within a page in the output file.
        for x0, y0, x1, y1, text, block_no, block_type in page.get_text("blocks"):
            stats["raw_blocks_seen"] += 1

            if block_type != 0:  # 0 = text block, 1 = image block (no usable text)
                stats["dropped_non_text_block"] += 1
                continue

            cleaned = _clean_block_text(text)

            if len(cleaned) < MIN_BLOCK_CHARS:
                stats["dropped_short"] += 1
                continue

            if not any(ch.isalpha() for ch in cleaned):
                # No letter at all (CJK counts as alpha via str.isalpha()) --
                # this is a block of page numbers, ruler marks, etc.
                stats["dropped_numeric_symbol"] += 1
                continue

            # bbox origin is the page's top-left corner, y grows downward
            # (see explanation in the reply below) -- a block is "in the
            # margin" if it lies ENTIRELY above the top 5% line or entirely
            # below the bottom 5% line.
            in_top_margin = y1 <= page_height * HEADER_FOOTER_MARGIN_PCT
            in_bottom_margin = y0 >= page_height * (1 - HEADER_FOOTER_MARGIN_PCT)
            if (in_top_margin or in_bottom_margin) and len(cleaned) < HEADER_FOOTER_MIN_CHARS:
                stats["dropped_header_footer"] += 1
                continue

            block_counter += 1
            kept_blocks.append(
                {
                    "block_id": f"b_{block_counter:05d}",
                    "page_no": page_no,
                    "text": cleaned,
                    "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                }
            )

    doc.close()

    cfg.paths.blocks.parent.mkdir(parents=True, exist_ok=True)
    with cfg.paths.blocks.open("w", encoding="utf-8") as f:
        for record in kept_blocks:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    _print_stats(cfg, stats, kept_blocks)
    return kept_blocks


def _print_stats(cfg: Config, stats: dict, kept_blocks: list[dict]) -> None:
    mode = "DEV" if cfg.is_dev else "FULL"
    lengths = [len(b["text"]) for b in kept_blocks]
    total_dropped = stats["raw_blocks_seen"] - len(kept_blocks)

    print("=" * 60)
    print(f"parser.py stats -- mode: {mode}")
    print("=" * 60)
    print(f"  pages in PDF          : {stats['total_pages_in_pdf']}")
    print(f"  pages processed       : {stats['pages_processed']}")
    print(f"  raw blocks seen       : {stats['raw_blocks_seen']}")
    print(f"  kept blocks           : {len(kept_blocks)}")
    print(f"  dropped total         : {total_dropped}")
    print(f"    - non-text (image)  : {stats['dropped_non_text_block']}")
    print(f"    - too short (<{MIN_BLOCK_CHARS} chars): {stats['dropped_short']}")
    print(f"    - numeric/symbol only: {stats['dropped_numeric_symbol']}")
    print(f"    - header/footer margin: {stats['dropped_header_footer']}")
    if lengths:
        print(f"  avg block length      : {sum(lengths) / len(lengths):.1f} chars")
        print(f"  min / max block length: {min(lengths)} / {max(lengths)} chars")
    if stats["pages_processed"]:
        print(f"  avg blocks per page   : {len(kept_blocks) / stats['pages_processed']:.2f}")
    print(f"  output written to     : {cfg.paths.blocks}")
    print("=" * 60)


def preview_page(page_no: int, cfg: Config | None = None) -> None:
    """Debug helper: print every RAW block PyMuPDF extracts from a single
    physical page (1-indexed, same convention as the page_no written to
    disk), with no noise filtering applied. Use this to flip to that page
    in the actual PDF and eyeball whether the text really belongs there --
    this is the manual page-number spot check the playbook requires.
    """
    cfg = cfg or load_config()
    doc = fitz.open(cfg.paths.pdf)
    try:
        if not (1 <= page_no <= len(doc)):
            raise ValueError(f"page_no must be within 1..{len(doc)}, got {page_no}")
        page = doc[page_no - 1]
        blocks = page.get_text("blocks")
        print(f"--- page_no={page_no} (PyMuPDF index {page_no - 1}), {len(blocks)} raw blocks ---")
        for i, (x0, y0, x1, y1, text, block_no, block_type) in enumerate(blocks):
            kind = "text" if block_type == 0 else "image"
            preview = text.strip().replace("\n", " \\n ")[:120]
            print(f"  [{i}] type={kind} bbox=({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}) text={preview!r}")
    finally:
        doc.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse target.pdf into page-numbered text blocks.")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help=(
            "Override dev.max_pages for this run and force dev mode on "
            "(regardless of config.yaml's dev.enabled), for quick test runs."
        ),
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    overrides = None
    if args.max_pages is not None:
        overrides = {"dev.enabled": True, "dev.max_pages": args.max_pages}

    cfg = load_config(overrides)
    cfg.describe()
    parse_pdf(cfg)


if __name__ == "__main__":
    main()
