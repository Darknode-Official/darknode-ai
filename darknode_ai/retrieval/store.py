"""KnowledgeStore -- a small BM25 retriever over Darknode's corpus.

Pure standard library. Documents are chunked on paragraph boundaries, indexed
with BM25 (Okapi, k1/b tunable), and searched by query. Each chunk keeps its
source and provenance so retrieved evidence is traceable.

    store = KnowledgeStore()
    store.ingest_dir("data/authored")
    store.ingest_jsonl("cve.jsonl", text_field="description",
                       source="nvd", provenance="NVD")
    store.build()
    hits = store.search("kerberoasting detection", k=4)

Persisted as one JSON file (store.save / KnowledgeStore.load).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def chunk_text(text: str, target: int = 1000) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > target:
            chunks.append(cur.strip())
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


@dataclass
class Doc:
    id: int
    text: str
    source: str
    provenance: str


@dataclass
class Hit:
    doc: Doc
    score: float


@dataclass
class KnowledgeStore:
    k1: float = 1.5
    b: float = 0.75
    docs: list[Doc] = field(default_factory=list)
    # index state (populated by build())
    _df: dict = field(default_factory=dict)          # term -> doc frequency
    _tf: list = field(default_factory=list)           # per-doc term counts
    _len: list = field(default_factory=list)          # per-doc token length
    _avgdl: float = 0.0
    _built: bool = False

    # -- ingestion ------------------------------------------------------------
    def add(self, text: str, source: str, provenance: str) -> None:
        text = text.strip()
        if text:
            self.docs.append(Doc(len(self.docs), text, source, provenance))
            self._built = False

    def ingest_text(self, text: str, source: str, provenance: str,
                    chunk: bool = True) -> int:
        pieces = chunk_text(text) if chunk else [text]
        for p in pieces:
            self.add(p, source, provenance)
        return len(pieces)

    def ingest_dir(self, directory: str | Path, provenance: str = "darknode-authored",
                   pattern: str = "*.txt") -> int:
        d = Path(directory)
        n = 0
        for path in sorted(d.glob(pattern)):
            n += self.ingest_text(path.read_text(encoding="utf-8", errors="replace"),
                                  source=path.name, provenance=provenance)
        return n

    def ingest_jsonl(self, path: str | Path, text_field: str,
                     source: str = "external", provenance: str = "external",
                     id_field: str | None = None) -> int:
        n = 0
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(text_field)
            if not text:
                continue
            src = f"{source}:{obj[id_field]}" if id_field and obj.get(id_field) else source
            self.add(str(text), source=src, provenance=provenance)
            n += 1
        return n

    # -- indexing / search ----------------------------------------------------
    def build(self) -> "KnowledgeStore":
        self._df, self._tf, self._len = {}, [], []
        for doc in self.docs:
            toks = tokenize(doc.text)
            counts = Counter(toks)
            self._tf.append(counts)
            self._len.append(len(toks))
            for term in counts:
                self._df[term] = self._df.get(term, 0) + 1
        self._avgdl = (sum(self._len) / len(self._len)) if self._len else 0.0
        self._built = True
        return self

    def _idf(self, term: str, n: int) -> float:
        df = self._df.get(term, 0)
        # BM25 idf with +1 floor to keep it non-negative
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 4, min_score: float = 0.0) -> list[Hit]:
        if not self._built:
            self.build()
        n = len(self.docs)
        if n == 0:
            return []
        q_terms = tokenize(query)
        scored: list[Hit] = []
        for i, doc in enumerate(self.docs):
            tf, dl = self._tf[i], self._len[i]
            s = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self._idf(term, n)
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
                s += idf * (f * (self.k1 + 1)) / (denom or 1)
            if s > min_score:
                scored.append(Hit(doc, round(s, 4)))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    # -- persistence ----------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "k1": self.k1, "b": self.b,
            "docs": [{"id": d.id, "text": d.text, "source": d.source,
                      "provenance": d.provenance} for d in self.docs],
        }, ensure_ascii=False), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeStore":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls(k1=obj.get("k1", 1.5), b=obj.get("b", 0.75))
        store.docs = [Doc(d["id"], d["text"], d["source"], d["provenance"])
                      for d in obj["docs"]]
        return store.build()
