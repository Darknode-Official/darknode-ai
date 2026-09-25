"""RAG glue -- turn retrieved chunks into RETRIEVED evidence and ground answers.

Two things:
  1. format_evidence(hits): render store hits as a Darknode evidence block, each
     line tagged RETRIEVED with its source so the model (and the operator) can
     trace every fact.
  2. DarknodeRAG: compose a KnowledgeStore with any chat provider (the foundation
     OllamaProvider, or anything exposing .chat(user_content)) so answers are
     grounded in the corpus instead of the model's parametric guesswork.

The retrieved corpus is Darknode's own trusted knowledge, so it is placed in the
prompt as context. If you ever index untrusted external text, tag its provenance
accordingly -- the model is instructed to weight RETRIEVED by source.
"""
from __future__ import annotations

from dataclasses import dataclass

from darknode_ai.retrieval.store import KnowledgeStore, Hit


def format_evidence(hits: list[Hit], max_chars: int = 700) -> str:
    if not hits:
        return "RETRIEVED: (no matching source in the knowledge base) -> treat as UNKNOWN."
    lines = []
    for h in hits:
        snippet = h.doc.text.strip().replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rsplit(" ", 1)[0] + " ..."
        lines.append(f"RETRIEVED [{h.doc.source} | {h.doc.provenance}]: {snippet}")
    return "\n".join(lines)


def build_grounded_prompt(query: str, hits: list[Hit]) -> str:
    evidence = format_evidence(hits)
    return (
        "Use the retrieved evidence below to ground your answer. Cite it as "
        "RETRIEVED and mark anything not supported by it as INFERRED or UNKNOWN.\n\n"
        f"{evidence}\n\n"
        f"Operator question: {query}"
    )


@dataclass
class DarknodeRAG:
    store: KnowledgeStore
    provider: object  # anything with .chat(user_content, **kw) -> str
    k: int = 4

    def retrieve(self, query: str) -> list[Hit]:
        return self.store.search(query, k=self.k)

    def answer(self, query: str, **chat_kw) -> dict:
        hits = self.retrieve(query)
        grounded = build_grounded_prompt(query, hits)
        completion = self.provider.chat(grounded, **chat_kw)
        return {
            "completion": completion,
            "evidence": [{"source": h.doc.source, "provenance": h.doc.provenance,
                          "score": h.score} for h in hits],
        }
