"""Darknode AI — byte-level BPE tokenizer, implemented from scratch.

No pretrained tokenizer, no external model. This trains a byte-pair-encoding
vocabulary directly on a Darknode security corpus. Byte-level means every
possible input encodes losslessly (IPs, hashes, base64, shell, JSON, logs),
which matters for security text.

Design mirrors the classic BPE algorithm (Sennrich et al., 2016) with a
GPT-2-style regex pre-tokenizer so merges never cross word/whitespace
boundaries in ways that hurt technical tokens.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

# GPT-2 style pattern, extended slightly so common security identifiers
# (CVE-2021-44228, 10.0.0.1, sha256 hashes, file paths) stay coherent.
SPLIT_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)|\ ?[A-Za-z]+|\ ?\d+|\ ?[^\sA-Za-z\d]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)

# Reserved special tokens. Kept out of the mergeable byte space and appended
# at the top of the id range so their ids are stable across vocab sizes.
SPECIAL_TOKENS = [
    "<|pad|>",
    "<|endoftext|>",   # document boundary
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|evidence|>",     # marks evidence-typed spans (OBSERVED/INFERRED/...)
    "<|tool|>",
    "<|endtool|>",
]


def _get_stats(ids: list[int], counts: dict[tuple[int, int], int] | None = None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special: dict[str, int] = {}
        self._special_pat: re.Pattern | None = None

    # ---- training -------------------------------------------------------
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256 + len(SPECIAL_TOKENS)
        num_merges = vocab_size - 256 - len(SPECIAL_TOKENS)

        chunks = re.findall(SPLIT_PATTERN, text)
        ids_chunks = [list(ch.encode("utf-8")) for ch in chunks]

        merges: dict[tuple[int, int], int] = {}
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        for m in range(num_merges):
            stats: dict[tuple[int, int], int] = {}
            for ids in ids_chunks:
                _get_stats(ids, stats)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            if stats[pair] < 2:
                break  # nothing worth merging
            new_id = 256 + m
            ids_chunks = [_merge(ids, pair, new_id) for ids in ids_chunks]
            merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            if verbose and (m % 200 == 0 or m == num_merges - 1):
                print(f"merge {m + 1}/{num_merges}: {pair} -> {new_id} "
                      f"({vocab[new_id]!r}) count={stats[pair]}")

        self.merges = merges
        self.vocab = vocab
        self._register_specials(next_id=256 + len(merges))

    def _register_specials(self, next_id: int) -> None:
        self.special = {}
        for i, tok in enumerate(SPECIAL_TOKENS):
            tid = next_id + i
            self.special[tok] = tid
            self.vocab[tid] = tok.encode("utf-8")
        self._special_pat = re.compile(
            "(" + "|".join(re.escape(s) for s in self.special) + ")"
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ---- encode / decode ------------------------------------------------
    def _encode_chunk(self, ids: list[int]) -> list[int]:
        while len(ids) >= 2:
            stats = _get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for ch in re.findall(SPLIT_PATTERN, text):
            out.extend(self._encode_chunk(list(ch.encode("utf-8"))))
        return out

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        """Encode text. Special-token strings are recognised only when
        allowed_special is True (so untrusted content can be encoded as
        ordinary bytes — see the injection-defense boundary in serve/)."""
        if not allowed_special or not self.special:
            return self._encode_ordinary(text)
        out: list[int] = []
        for part in self._special_pat.split(text):
            if part in self.special:
                out.append(self.special[part])
            elif part:
                out.extend(self._encode_ordinary(part))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        parts = bytes()
        pieces: list[bytes] = []
        for i in ids:
            pieces.append(self.vocab.get(i, b"\xef\xbf\xbd"))
        parts = b"".join(pieces)
        return parts.decode("utf-8", errors="replace")

    # ---- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model = {
            "version": 1,
            "merges": [[list(k), v] for k, v in self.merges.items()],
            "special": self.special,
            "vocab_size": self.vocab_size,
        }
        path.write_text(json.dumps(model), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.merges = {tuple(k): v for k, v in data["merges"]}
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in tok.merges.items():
            vocab[idx] = vocab[a] + vocab[b]
        tok.special = {k: int(v) for k, v in data["special"].items()}
        for tok_str, tid in tok.special.items():
            vocab[tid] = tok_str.encode("utf-8")
        tok.vocab = vocab
        tok._special_pat = re.compile(
            "(" + "|".join(re.escape(s) for s in tok.special) + ")"
        ) if tok.special else None
        return tok
