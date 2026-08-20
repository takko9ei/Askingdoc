"""
Parent/child two-level chunking, i.e. small-to-big (RAG pipeline: chunking,
between step (1) parsing and step (3) indexing).

Why two levels: retrieval wants small, topic-focused chunks (a big chunk's
embedding blurs together too many topics, hurting match precision); answer
generation wants large, context-complete chunks (a chunk that's too small
may be missing the qualifier/condition that makes an answer correct). We
resolve the conflict by cutting the same text twice, at two granularities:

  - parent (~config.chunking.parent_size tokens): built here by walking
    STEP1's blocks in order and accumulating them until the running token
    estimate would exceed parent_size. Coarse and cheap -- blocks are
    already paragraph-sized units, so this is just "keep appending until
    full".
  - child (~config.chunking.child_size tokens): built by running
    LlamaIndex's SentenceSplitter over each parent's text. Fine-grained
    sentence-boundary-aware splitting is a solved problem; per this
    project's design principle ("hand-roll the retrieval pipeline, but
    let LlamaIndex do generic text splitting"), we don't reimplement it.

Only children get embedded and searched (STEP2 slice 2-2 / STEP4). Once a
child is retrieved, STEP5's small-to-big expansion swaps it back to its
parent before generation -- this module only builds the data both steps
will need; it doesn't do the retrieval-time swap itself.

*** PAGE NUMBER PROPAGATION (the trickiest part of this file) ***
A parent's `pages` field can legitimately span several physical pages (a
parent that started accumulating near the bottom of page 86 might run into
page 88 before it fills up). A child cut out of that parent's text is NOT
automatically "page 86" just because the parent started there -- if the
child's own text happens to fall in the part of the parent that came from
page 87, it must be recorded as page 87. See `_build_parents` and
`_page_no_for_offset` for how this is tracked precisely via character
offsets rather than guessed.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from llama_index.core.node_parser import SentenceSplitter

from src.config import Config, load_config

# Matches LlamaIndex's SentenceSplitter default paragraph_separator. Passed
# explicitly (not left to the library default) so parent-building and child
# splitting are guaranteed to agree on it even if a future LlamaIndex
# version changes its own default.
PARAGRAPH_SEPARATOR = "\n\n\n"


def _is_cjk_char(ch: str) -> bool:
    """Same range check as parser.py's _is_cjk. Duplicated rather than
    imported: this module's 改动范围 is "only create chunker.py", and a
    2-line range test isn't worth reaching into another module for."""
    if not ch:
        return False
    return "一" <= ch <= "鿿" or "　" <= ch <= "〿" or "＀" <= ch <= "￯"


def estimate_tokens(text: str) -> float:
    """Cheap token-count estimate for mixed Chinese/English text, used only
    to decide when a parent has accumulated ~parent_size worth of content --
    this is NOT a real tokenizer call (no model download, no API call, just
    arithmetic on character classes):
      - CJK characters: ~1 token each (a Han character is close to one BPE
        token in most tokenizers)
      - everything else (ASCII letters/digits/punctuation/whitespace):
        ~4 characters per token (typical English BPE ratio)

    Spot-checked against LlamaIndex's actual (tiktoken-based) tokenizer on
    this document's mostly-Chinese text: ~100 Chinese characters measured
    as ~101 real tokens, i.e. this heuristic and the real tokenizer used
    for child-splitting agree closely here, so parent_size (measured by
    this function) and child_size (measured by SentenceSplitter's own
    tokenizer) stay roughly comparable units despite being counted two
    different ways.
    """
    cjk_count = sum(1 for ch in text if _is_cjk_char(ch))
    other_count = len(text) - cjk_count
    return cjk_count + other_count / 4


@dataclass
class _ParentBuild:
    """Internal-only pairing of a parent's on-disk record with the
    block-offset map needed to resolve each of its children's page_no.
    `block_spans` is NEVER written to parents.jsonl -- it's scaffolding
    that only this module needs, and only while building children."""

    record: dict
    block_spans: list[tuple[int, int, int]] = field(default_factory=list)  # (start, end, page_no)


def _build_parents(blocks: list[dict], parent_size: int) -> list[_ParentBuild]:
    """Greedily accumulate STEP1 blocks (in their original page order) into
    parents, cutting a new parent once adding the next block would push the
    running token estimate over parent_size.

    Deliberately block-level, not sentence-level: blocks are already
    paragraph-sized, so "keep appending whole blocks" is enough -- we only
    need SentenceSplitter's finer sentence-aware logic at the child layer,
    where getting a mid-sentence cut wrong would actually hurt retrieval.

    A single block bigger than parent_size (didn't happen on this document
    -- longest observed block was 613 chars from STEP1's stats -- but must
    still be handled) becomes a parent by itself rather than being split
    further; parents are allowed to slightly exceed parent_size in that
    case, which is an acceptable, rare edge case.
    """
    parent_builds: list[_ParentBuild] = []
    current_group: list[dict] = []
    current_tokens = 0.0

    def flush() -> None:
        nonlocal current_group, current_tokens
        if not current_group:
            return
        parent_builds.append(_finalize_parent(current_group, len(parent_builds) + 1))
        current_group = []
        current_tokens = 0.0

    for block in blocks:
        block_tokens = estimate_tokens(block["text"])
        if current_group and current_tokens + block_tokens > parent_size:
            flush()
        current_group.append(block)
        current_tokens += block_tokens
    flush()

    return parent_builds


def _finalize_parent(blocks_in_group: list[dict], parent_index: int) -> _ParentBuild:
    """Join a group of blocks into one parent's text with PARAGRAPH_SEPARATOR
    between them, while recording each block's exact character interval
    inside that joined text (`block_spans`). This is the offset bookkeeping
    that later lets us map "a child starting at character 812" back to
    "which original block (and therefore which page) that character came
    from" -- without it, a child's page_no would have to be guessed (e.g.
    "just use the parent's first page"), which is exactly the kind of
    silent inaccuracy the whole citation feature can't tolerate.
    """
    text_parts: list[str] = []
    block_spans: list[tuple[int, int, int]] = []
    offset = 0
    for block in blocks_in_group:
        if text_parts:
            offset += len(PARAGRAPH_SEPARATOR)  # account for the separator join() will insert
        start = offset
        end = start + len(block["text"])
        block_spans.append((start, end, block["page_no"]))
        text_parts.append(block["text"])
        offset = end

    text = PARAGRAPH_SEPARATOR.join(text_parts)
    pages = sorted({b["page_no"] for b in blocks_in_group})
    record = {
        "pid": f"p_{parent_index:04d}",
        "text": text,
        "pages": pages,
        "block_ids": [b["block_id"] for b in blocks_in_group],
    }
    return _ParentBuild(record=record, block_spans=block_spans)


def _page_no_for_offset(block_spans: list[tuple[int, int, int]], offset: int) -> int:
    """Given a character offset into a parent's joined text, return the
    page_no of the block that offset falls inside.

    Normal case: `offset` lands strictly within one block's [start, end)
    interval -- return that block's page_no directly.

    Edge case: `offset` lands inside a PARAGRAPH_SEPARATOR gap BETWEEN two
    blocks (possible if a child chunk's boundary happens to start exactly
    at/inside the "\n\n\n" we inserted). We resolve that by attributing the
    child to the block it's about to enter (the next block whose start is
    >= offset) rather than the block just before it -- a child's page_no
    should reflect what it's actually the start of, not trailing
    separator whitespace from the previous block.
    """
    for start, end, page_no in block_spans:
        if start <= offset < end:
            return page_no
    for start, end, page_no in block_spans:
        if start >= offset:
            return page_no
    return block_spans[-1][2]  # offset is past the end of the text -- fall back to the last block


def _build_children(parent_builds: list[_ParentBuild], splitter: SentenceSplitter) -> list[dict]:
    """Split every parent's text into children via SentenceSplitter, and
    resolve each child's page_no from where it starts inside its parent.

    SentenceSplitter.split_text() returns child strings, not offsets, so we
    have to relocate each child inside the parent text ourselves via
    str.find(). This is safe because SentenceSplitter does not rewrite or
    normalize the text it splits (verified empirically: every child text is
    an exact, byte-for-byte substring of its parent) -- if that ever
    changed in a future LlamaIndex version, `start` would come back -1 and
    we fall back loudly (a printed warning) rather than silently mis-map a
    page number.

    `cursor` advances by 1 (not by the full child length) after each find,
    which is what makes this correct even with overlapping children
    (chunk_overlap > 0 means consecutive children share a text prefix/
    suffix, so the next child's start can be BEFORE the previous child's
    end -- searching from `previous_start + 1` still finds it, whereas
    searching from `previous_start + len(child)` could overshoot past it).
    """
    children: list[dict] = []
    child_counter = 0

    for pb in parent_builds:
        parent_text = pb.record["text"]
        child_texts = splitter.split_text(parent_text)
        cursor = 0
        for child_text in child_texts:
            start = parent_text.find(child_text, cursor)
            if start == -1:
                start = parent_text.find(child_text)
            if start == -1:
                print(
                    f"WARNING: child text not found verbatim inside parent "
                    f"{pb.record['pid']!r} -- page_no for this child defaults to "
                    f"the parent's first page and may be wrong. Investigate if "
                    f"this happens (SentenceSplitter may have started rewriting text)."
                )
                start = 0

            page_no = _page_no_for_offset(pb.block_spans, start)
            child_counter += 1
            children.append(
                {
                    "cid": f"c_{child_counter:04d}",
                    "parent_id": pb.record["pid"],
                    "text": child_text,
                    "page_no": page_no,
                }
            )
            cursor = start + 1

    return children


def build_index_units(cfg: Config) -> tuple[list[dict], list[dict]]:
    """Read cfg.paths.blocks, build parents then children, write both to
    cfg.paths.parents / cfg.paths.children as JSONL, print stats, and
    return (parent_records, child_records) for in-process reuse (e.g. by
    the CLI's --limit test runs or a future notebook)."""
    with cfg.paths.blocks.open("r", encoding="utf-8") as f:
        blocks = [json.loads(line) for line in f]
    if not blocks:
        raise RuntimeError(f"{cfg.paths.blocks} is empty -- run `python -m src.ingest.parser` first.")

    parent_builds = _build_parents(blocks, cfg.chunking.parent_size)

    splitter = SentenceSplitter(
        chunk_size=cfg.chunking.child_size,
        chunk_overlap=cfg.chunking.overlap,
        paragraph_separator=PARAGRAPH_SEPARATOR,
    )
    child_records = _build_children(parent_builds, splitter)
    parent_records = [pb.record for pb in parent_builds]

    cfg.paths.parents.parent.mkdir(parents=True, exist_ok=True)
    with cfg.paths.parents.open("w", encoding="utf-8") as f:
        for record in parent_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    cfg.paths.children.parent.mkdir(parents=True, exist_ok=True)
    with cfg.paths.children.open("w", encoding="utf-8") as f:
        for record in child_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    _print_stats(cfg, parent_records, child_records)
    return parent_records, child_records


def _print_stats(cfg: Config, parent_records: list[dict], child_records: list[dict]) -> None:
    mode = "DEV" if cfg.is_dev else "FULL"
    n_parents = len(parent_records)
    n_children = len(child_records)
    cross_page = sum(1 for p in parent_records if len(p["pages"]) > 1)
    child_lengths = [len(c["text"]) for c in child_records]

    print("=" * 60)
    print(f"chunker.py stats -- mode: {mode}")
    print("=" * 60)
    print(f"  parents                : {n_parents}")
    print(f"  children               : {n_children}")
    if n_parents:
        print(f"  avg children / parent  : {n_children / n_parents:.2f}")
        print(f"  cross-page parents     : {cross_page} / {n_parents} ({cross_page / n_parents * 100:.1f}%)")
    if child_lengths:
        print(f"  avg child length       : {sum(child_lengths) / len(child_lengths):.1f} chars")
        print(f"  min / max child length : {min(child_lengths)} / {max(child_lengths)} chars")
    print(f"  parents written to     : {cfg.paths.parents}")
    print(f"  children written to    : {cfg.paths.children}")
    print("=" * 60)


def verify_page_mapping(n: int = 10, cfg: Config | None = None) -> None:
    """Debug helper: randomly sample n children from cfg.paths.children and
    print their text preview + page_no, for human spot-checking (flip to
    that physical page in the PDF and confirm the previewed text is really
    there). random.sample means re-running this gives a different sample
    each time rather than always checking the same handful of children.
    """
    cfg = cfg or load_config()
    with cfg.paths.children.open("r", encoding="utf-8") as f:
        children = [json.loads(line) for line in f]
    if not children:
        print(f"No children found in {cfg.paths.children} -- run build_index_units() first.")
        return

    sample = random.sample(children, min(n, len(children)))
    print(f"--- verify_page_mapping: {len(sample)} random children (of {len(children)} total) ---")
    for c in sample:
        preview = c["text"][:50].replace("\n", " \\n ")
        print(f"  [{c['cid']}] page_no={c['page_no']:>4}  parent={c['parent_id']:<8}  text={preview!r}")


def main() -> None:
    cfg = load_config()
    cfg.describe()
    build_index_units(cfg)


if __name__ == "__main__":
    main()
