"""Dataset pipeline with governance, dedup, redaction and packing.

Sources are described by a manifest (source, license, category, provenance) so
nothing is trained on without recorded rights — the spec's data-governance
requirement. Each document flows: load -> redact -> normalise -> dedup ->
tokenize -> pack into a uint16 memmap (nanoGPT-style .bin) with a meta.json
that records a content hash used as the dataset version.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from darknode_ai.data.redact import redact, RedactionReport
from darknode_ai.tokenizer.bpe import BPETokenizer

_WS = re.compile(r"[ \t]+")
_NL = re.compile(r"\n{3,}")


@dataclass
class Source:
    path: str
    license: str
    category: str
    provenance: str
    allowed_use: str = "train"


@dataclass
class DatasetStats:
    docs: int = 0
    docs_kept: int = 0
    docs_deduped: int = 0
    train_tokens: int = 0
    val_tokens: int = 0
    vocab_size: int = 0
    redaction: RedactionReport = field(default_factory=RedactionReport)
    version: str = ""
    per_category: dict[str, int] = field(default_factory=dict)


def normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = _NL.sub("\n\n", text)
    return text.strip()


def _doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_manifest(manifest_path: str | Path) -> list[Source]:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return [Source(**s) for s in data["sources"]]


def build_dataset(
    manifest_path: str | Path,
    tokenizer: BPETokenizer,
    out_dir: str | Path,
    val_fraction: float = 0.1,
    max_secret_density: float = 5.0,
    seed: int = 1337,
) -> DatasetStats:
    root = Path(manifest_path).parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = load_manifest(manifest_path)

    eot = tokenizer.special.get("<|endoftext|>")
    stats = DatasetStats(vocab_size=tokenizer.vocab_size)
    seen: set[str] = set()
    docs: list[list[int]] = []

    for src in sources:
        if src.allowed_use != "train":
            continue
        for fp in sorted((root / src.path).glob("**/*") if (root / src.path).is_dir()
                         else [root / src.path]):
            if not fp.is_file():
                continue
            raw = fp.read_text(encoding="utf-8", errors="replace")
            # documents separated by a blank-line-delimited <<<DOC>>> marker or whole file
            chunks = raw.split("\n<<<DOC>>>\n") if "<<<DOC>>>" in raw else [raw]
            for chunk in chunks:
                stats.docs += 1
                text = normalise(chunk)
                if len(text) < 16:
                    continue
                text, rep = redact(text)
                stats.redaction.merge(rep)
                h = _doc_hash(text)
                if h in seen:
                    stats.docs_deduped += 1
                    continue
                seen.add(h)
                ids = tokenizer.encode(text, allowed_special=False)
                if eot is not None:
                    ids.append(eot)
                docs.append(ids)
                stats.docs_kept += 1
                stats.per_category[src.category] = stats.per_category.get(src.category, 0) + 1

    if stats.redaction.density > max_secret_density:
        raise ValueError(
            f"Secret density {stats.redaction.density:.2f}/1k chars exceeds "
            f"limit {max_secret_density}. Refusing to build — inspect the source.")

    rng = np.random.default_rng(seed)
    rng.shuffle(docs)
    n_val = max(1, int(len(docs) * val_fraction)) if len(docs) > 1 else 0
    val_docs, train_docs = docs[:n_val], docs[n_val:]

    def _write(split_docs, name):
        flat = [t for doc in split_docs for t in doc]
        arr = np.array(flat, dtype=np.uint16)
        arr.tofile(out_dir / f"{name}.bin")
        return len(flat)

    stats.train_tokens = _write(train_docs, "train")
    stats.val_tokens = _write(val_docs, "val") if val_docs else 0

    payload = json.dumps({
        "train_tokens": stats.train_tokens,
        "val_tokens": stats.val_tokens,
        "docs_kept": stats.docs_kept,
        "vocab_size": stats.vocab_size,
    }, sort_keys=True)
    stats.version = hashlib.sha256(payload.encode()).hexdigest()[:16]

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": stats.train_tokens,
        "val_tokens": stats.val_tokens,
        "docs": stats.docs,
        "docs_kept": stats.docs_kept,
        "docs_deduped": stats.docs_deduped,
        "per_category": stats.per_category,
        "redaction": {"total": stats.redaction.total,
                       "density_per_1k": round(stats.redaction.density, 3),
                       "counts": stats.redaction.counts},
        "dataset_version": stats.version,
        "dtype": "uint16",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return stats
