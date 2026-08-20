"""Hand-rolled retrieval: vector search, BM25 + RRF fusion, cross-encoder rerank,
small-to-big parent expansion. All four are toggled by config.retrieval.* flags
inside a single search() function -- see search.py once it exists (STEP2+)."""
