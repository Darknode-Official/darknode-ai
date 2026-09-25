"""Build a clean-provenance SFT dataset for the Darknode foundation model.

Produces chat-format JSONL (messages: system/user/assistant) suitable for LoRA
fine-tuning with TRL/transformers. Sources:

  1. Darknode authored corpus (data/authored/*.txt) -> domain-brief pairs.
  2. Darknode synthetic triage cases (control-token format) -> parsed to chat.
  3. Optional external JSONL passed with --extra (e.g. a local AttackQA export),
     mapped via a small field spec.

Guarantees:
  - Every record passes secret/PII redaction (reuses data.redact).
  - Records are deduplicated by content hash.
  - Sets distilled from a proprietary model are REFUSED (provenance guard):
    filenames matching the excluded list raise, so they cannot slip in.
  - A provenance manifest is written alongside the JSONL.

Nothing here needs a GPU or network; it runs on the checked-in corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from darknode_ai.data.redact import redact
from darknode_ai.foundation.persona import DARKNODE_SYSTEM

# Provenance guard: refuse anything that is distilled from another vendor model.
_REFUSED_SOURCES = ("primus-instruct", "primus-reasoning", "fable-5-traces",
                    "fable_5_traces", "claude-fable", "gpt5.5-terminal")

# Built-in field maps for known offensive/defensive datasets. Point --dataset at
# a locally-downloaded JSONL; this resolves the fields, license, and provenance.
#   commercial: False       -> non-commercial license, warns loudly
#   review: True            -> provenance not yet verified; needs --allow-review
KNOWN_DATASETS = {
    "pentest-v2": {"user_field": "instruction", "asst_field": "output",
                   "provenance": "gewsefa/pentest-v2",
                   "license": "derived (HackTricks CC-BY-NC etc.)",
                   "commercial": False, "review": False},
    "whiterabbitneo": {"user_field": "instruction", "asst_field": "response",
                       "provenance": "WhiteRabbitNeo-dataset",
                       "license": "WhiteRabbitNeo terms",
                       "commercial": True, "review": False},
    "nyu-ctf": {"user_field": "prompt", "asst_field": "solution",
                "provenance": "NYU-CTF-Bench",
                "license": "CSAW CTF (see repo)",
                "commercial": True, "review": False},
    "redsage-conv": {"user_field": "user", "asst_field": "assistant",
                     "provenance": "RISys-Lab/RedSage-Conv",
                     "license": "ICLR2026 release",
                     # synthetic multi-turn: verify it was not distilled from a
                     # proprietary model before trusting provenance.
                     "commercial": True, "review": True},
}


def known_spec(name: str, path: str, allow_review: bool = False) -> dict:
    key = name.lower()
    if key not in KNOWN_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Known: {', '.join(KNOWN_DATASETS)}. "
                         "For anything else pass an explicit --extra field spec.")
    d = KNOWN_DATASETS[key]
    if d["review"] and not allow_review:
        raise ValueError(
            f"'{name}' provenance is unverified (synthetic; possibly distilled). "
            "Verify how it was generated; re-run with --allow-review to include it "
            "once you have confirmed it is not distilled from a proprietary model.")
    if not d["commercial"]:
        print(f"WARNING: '{name}' is NON-COMMERCIAL ({d['license']}). Fine for "
              "research/lab use; do not ship it commercially without clearing the "
              "upstream license.")
    return {"path": path, "user_field": d["user_field"], "asst_field": d["asst_field"],
            "provenance": d["provenance"], "license": d["license"]}

_DOMAIN_LABELS = {
    "appsec": "application security",
    "cloud": "cloud security",
    "detection": "detection engineering",
    "ir": "incident response and DFIR",
    "linux": "Linux security",
    "logs": "log analysis",
    "malware_defense": "malware analysis and defense",
    "networking": "network security",
    "threatintel": "threat intelligence",
    "windows_ad": "Windows and Active Directory security",
}


@dataclass
class PrepStats:
    records: int = 0
    deduped: int = 0
    redactions: int = 0
    sources: list = field(default_factory=list)


def _guard_source(name: str) -> None:
    low = name.lower()
    for bad in _REFUSED_SOURCES:
        if bad in low:
            raise ValueError(
                f"Refusing source '{name}': matches excluded distilled-provenance "
                f"set '{bad}'. The foundation track uses clean-provenance data only "
                "(see darknode_ai/foundation/notice.py EXCLUDED).")


def _chunk_doc(text: str, target: int = 1400) -> list[str]:
    """Split a prose doc into ~target-sized chunks on paragraph boundaries."""
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


def _authored_records(authored_dir: Path) -> list[dict]:
    out = []
    for path in sorted(authored_dir.glob("*.txt")):
        _guard_source(path.name)
        stem = path.stem.replace("agent_", "")
        label = _DOMAIN_LABELS.get(stem, stem.replace("_", " "))
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, chunk in enumerate(_chunk_doc(text)):
            first = chunk.splitlines()[0].strip().rstrip(":.")
            prompt = (f"Brief me on {label}: {first}." if i == 0
                      else f"Continue the Darknode brief on {label}. Next: {first}.")
            out.append({"user": prompt, "assistant": chunk,
                        "provenance": "darknode-authored", "license": "CC0-1.0"})
    return out


def _parse_control_cases(text: str) -> list[dict]:
    """Parse `<|user|> ... <|assistant|> ...` blocks into chat records."""
    out = []
    blocks = re.split(r"<\|endoftext\|>", text)
    for b in blocks:
        m = re.search(r"<\|user\|>(.*?)<\|assistant\|>(.*)", b, re.S)
        if not m:
            continue
        user = m.group(1).strip()
        asst = m.group(2).strip()
        if user and asst:
            out.append({"user": user, "assistant": asst,
                        "provenance": "darknode-synthetic", "license": "CC0-1.0"})
    return out


def _extra_records(path: Path, user_field: str, asst_field: str,
                   provenance: str, license_: str) -> list[dict]:
    _guard_source(path.name)
    _guard_source(provenance)
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        u, a = obj.get(user_field), obj.get(asst_field)
        if u and a:
            out.append({"user": str(u), "assistant": str(a),
                        "provenance": provenance, "license": license_})
    return out


def _to_chat(rec: dict) -> dict:
    return {"messages": [
        {"role": "system", "content": DARKNODE_SYSTEM},
        {"role": "user", "content": rec["user"]},
        {"role": "assistant", "content": rec["assistant"]},
    ], "provenance": rec["provenance"], "license": rec["license"]}


def build_sft(out_dir: str | Path, authored_dir: str | Path = "data/authored",
              synthetic_file: str | Path | None = "data/corpus/synthetic_cases.txt",
              extra: list[dict] | None = None, val_fraction: float = 0.05,
              seed: int = 0) -> PrepStats:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = PrepStats()
    raw: list[dict] = []

    a_dir = Path(authored_dir)
    if a_dir.is_dir():
        recs = _authored_records(a_dir)
        raw += recs
        stats.sources.append({"source": str(a_dir), "records": len(recs),
                              "provenance": "darknode-authored"})

    if synthetic_file and Path(synthetic_file).is_file():
        recs = _parse_control_cases(Path(synthetic_file).read_text(
            encoding="utf-8", errors="replace"))
        raw += recs
        stats.sources.append({"source": str(synthetic_file), "records": len(recs),
                              "provenance": "darknode-synthetic"})

    for spec in (extra or []):
        recs = _extra_records(Path(spec["path"]), spec["user_field"],
                              spec["asst_field"], spec["provenance"], spec["license"])
        raw += recs
        stats.sources.append({"source": spec["path"], "records": len(recs),
                              "provenance": spec["provenance"]})

    # redact + dedup
    seen: set[str] = set()
    clean: list[dict] = []
    for rec in raw:
        u, ru = redact(rec["user"])
        a, ra = redact(rec["assistant"])
        stats.redactions += ru.total + ra.total
        rec = {**rec, "user": u, "assistant": a}
        h = hashlib.sha256((u + "\x00" + a).encode("utf-8")).hexdigest()
        if h in seen:
            stats.deduped += 1
            continue
        seen.add(h)
        clean.append(rec)

    rng = random.Random(seed)
    rng.shuffle(clean)
    n_val = int(len(clean) * val_fraction)
    val, train = clean[:n_val], clean[n_val:]
    stats.records = len(clean)

    for name, split in (("train.jsonl", train), ("val.jsonl", val)):
        with (out / name).open("w", encoding="utf-8") as f:
            for rec in split:
                f.write(json.dumps(_to_chat(rec), ensure_ascii=False) + "\n")

    version = hashlib.sha256(
        json.dumps([r["user"] + r["assistant"] for r in clean],
                   ensure_ascii=False).encode()).hexdigest()[:16]
    (out / "provenance.json").write_text(json.dumps({
        "version": version, "records": stats.records, "train": len(train),
        "val": len(val), "deduped": stats.deduped, "redactions": stats.redactions,
        "sources": stats.sources, "excluded_by_guard": list(_REFUSED_SOURCES),
    }, indent=2), encoding="utf-8")
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build clean-provenance SFT data")
    ap.add_argument("--out", default="data/sft")
    ap.add_argument("--authored", default="data/authored")
    ap.add_argument("--synthetic", default="data/corpus/synthetic_cases.txt")
    ap.add_argument("--dataset", action="append", default=[],
                    metavar="NAME=PATH",
                    help="add a known dataset from a local JSONL, e.g. "
                         "pentest-v2=./pentest.jsonl (see KNOWN_DATASETS)")
    ap.add_argument("--allow-review", action="store_true",
                    help="include datasets whose provenance is flagged for review")
    args = ap.parse_args(argv)

    extra = []
    for item in args.dataset:
        if "=" not in item:
            raise SystemExit(f"--dataset expects NAME=PATH, got '{item}'")
        name, path = item.split("=", 1)
        extra.append(known_spec(name.strip(), path.strip(),
                                allow_review=args.allow_review))

    stats = build_sft(args.out, authored_dir=args.authored,
                      synthetic_file=args.synthetic, extra=extra)
    print(f"sft: records={stats.records} deduped={stats.deduped} "
          f"redactions={stats.redactions} sources={len(stats.sources)} -> {args.out}")


if __name__ == "__main__":
    main()
