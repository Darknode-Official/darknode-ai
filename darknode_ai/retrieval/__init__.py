"""Darknode AI — retrieval (RAG) layer.

Facts belong in retrieval, not baked into model weights. This package builds a
small, dependency-free knowledge store over Darknode's corpus (and any local
CVE/knowledge exports) and returns RETRIEVED evidence to ground the model's
answers. It works for both tracks (from-scratch and foundation) and keeps the
model honest: what it states as RETRIEVED is traceable to a source chunk.

No external dependencies -- BM25 is implemented directly (stdlib only), matching
the repo's from-scratch ethos and keeping the serving path lightweight.
"""
