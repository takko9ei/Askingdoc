"""
Unified tokenization for BM25.

*** MUST be the exact same function used both when BUILDING the BM25 index
(src/ingest/indexer.py, this STEP) and when tokenizing a QUERY at search
time (src/retrieval/search.py / fusion.py, STEP4) ***

BM25 scores documents purely on token overlap with the query -- there is no
fuzzy/semantic fallback. If the corpus were tokenized one way at index time
and a query tokenized a subtly different way at search time (different
jieba mode, different case folding, different punctuation handling), tokens
that are "obviously the same word" to a human would silently fail to
overlap, and BM25 would return near-random results with no error raised
anywhere to say why. Import this one function everywhere tokenization is
needed; never reimplement it locally "just for this one call site".
"""

from __future__ import annotations

import jieba


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text for BM25.

    - jieba.lcut() handles the mixed-language segmentation: it splits
      Chinese into dictionary words and keeps contiguous ASCII runs
      (e.g. model names like "ILCE-7CM2") as their own tokens.
    - Every token is lowercased, so a query for "RAW" matches a corpus
      token "raw" -- BM25 itself has no case-insensitivity built in, it
      only does exact string-equality lookups in its term-frequency table.
    - Tokens with no alphanumeric character at all (pure punctuation like
      "，" "。" "（" "）" or stray whitespace jieba sometimes emits) are
      dropped: they occur in nearly every document, so keeping them would
      only add noise to term matching, never useful discriminating signal.
    """
    tokens: list[str] = []
    for tok in jieba.lcut(text):
        tok = tok.strip().lower()
        if not tok:
            continue
        if not any(ch.isalnum() for ch in tok):
            continue
        tokens.append(tok)
    return tokens
