"""
Prompt templates for answer generation (RAG pipeline: step (7) generation).

Only ANSWER_PROMPT is defined this STEP. Citation-format enforcement (the
strict [p.87] tagging scheme, multi-source tags, few-shot examples) is
STEP5's job -- this version only lays down the anti-hallucination hard
constraints and labels each retrieved片段 with its page number in natural
language, so a future citation pass has something to tighten rather than
build from nothing.
"""

from __future__ import annotations

from src.retrieval.search import Hit

ANSWER_PROMPT = """你是一个说明书问答助手。你的回答必须严格遵守以下规则：

1. 只能依据下面【参考片段】里的内容回答问题，不允许使用你自己的知识、常识或做任何超出片段内容的推测。
2. 如果【参考片段】中没有能回答这个问题的信息，只能回复："文档中未提及相关信息。"不要试图用猜测或常识拼凑一个看似合理的答案。
3. 回答中提到具体信息时，请指出这来自第几页（例如"第87页提到..."）。严格的引用标注格式（如 [p.87]）会在后续版本加强，这一版先用自然语言标注即可。

【参考片段】
{context}

【问题】
{query}

【回答】
"""


def format_context(hits: list[Hit]) -> str:
    """Render retrieved hits into the {context} block ANSWER_PROMPT expects,
    each片段 prefixed with its page number so the LLM has something concrete
    to cite -- without this label, rule 3 above would be unenforceable no
    matter how the prompt is worded, since the page number lives in
    Hit.page_no, not in hit.text itself.
    """
    return "\n\n".join(f"[来源: 第{hit.page_no}页]\n{hit.text}" for hit in hits)
