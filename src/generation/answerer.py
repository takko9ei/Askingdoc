"""
Answer generation (RAG pipeline: step (7) -- stitching retrieval output
into a grounded, page-labeled LLM answer). This STEP wires generation
end-to-end for the first time; STEP5 strengthens citation formatting and
adds abstention checks beyond the prompt-level constraints in prompts.py.
"""

from __future__ import annotations

from openai import OpenAI

from src.config import Config
from src.generation.prompts import ANSWER_PROMPT, format_context
from src.retrieval.search import Hit, search

# Same fingerprint-cache pattern as search.py's singletons (see that
# module's comments for the full rationale) -- avoids rebuilding an HTTP
# client on every single question in a long-running CLI/API process.
_llm_clients: dict[tuple, OpenAI] = {}


def _llm_client_for(cfg: Config) -> OpenAI:
    fingerprint = (cfg.llm_api_key, cfg.llm_base_url)
    if fingerprint not in _llm_clients:
        _llm_clients[fingerprint] = OpenAI(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url)
    return _llm_clients[fingerprint]


def answer(query: str, cfg: Config) -> dict:
    """Run retrieval then generation for one question.

    Returns {"answer": str, "hits": list[Hit]} -- `hits` travels alongside
    the answer text (not just returned as a side effect / logged) because
    both cli.py (page numbers to display) and eval/harness.py (STEP3,
    Recall@k / MRR computation) need the raw retrieval result, not just
    the final prose.
    """
    hits = search(query, cfg)

    context = format_context(hits)
    prompt = ANSWER_PROMPT.format(context=context, query=query)

    client = _llm_client_for(cfg)
    response = client.chat.completions.create(
        model=cfg.models.llm,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # deterministic-as-possible: this is a factual lookup
        # task grounded in retrieved text, not creative writing -- we want
        # the model to stick close to the provided片段 rather than sample
        # varied phrasings, which also makes manual QA runs reproducible.
    )
    answer_text = response.choices[0].message.content

    return {"answer": answer_text, "hits": hits}
