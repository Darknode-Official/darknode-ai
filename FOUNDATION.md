# Darknode AI — Foundation (open-base) track

Two tracks ship in this repo:

- **from-scratch** (`darknode_ai/model`, `train`) — a transformer trained from
  random init by Darknode. Honest "we built the architecture ourselves" story;
  capability capped by a small corpus. Truthfully "built by Darknode."
- **foundation** (`darknode_ai/foundation`) — **this track** — specializes a
  strong, openly-licensed **security** base model into Darknode with LoRA on
  clean-provenance data, and serves it as Darknode AI via Ollama. This is the
  path to a genuinely capable local Darknode.

## Why this base

**Default: WhiteRabbitNeo-13B-v1** (`WhiteRabbitNeo/WhiteRabbitNeo-13B-v1`) — an
open base built specifically for **offensive and defensive** security (red/blue
team), trained on ~1.7M offensive/defensive samples, single-GPU QLoRA-able, and
on Ollama (`jimscard/whiterabbit-neo`). It is the closest open base to Darknode's
"expert operator" goal. It is a dual-use, less-restrictive base; Darknode keeps
the authorized-engagement frame in the persona regardless.

Alternatives (`--preset` or `--base`):
- **`--preset 33b`** WhiteRabbitNeo-33B-v1 — more capability, ~1x A100/H100.
- **`--preset 100b`** Qwen3.5-122B (MoE, ~122B total / ~10B active) — a genuine
  100B+ Darknode on a rented A100/H100 80GB. See **[RUNBOOK_100B.md](RUNBOOK_100B.md)**.
- **`--preset 500b`** Nemotron-3-Ultra-550B (MoE, ~550B / ~55B active) — cluster
  scale (8x H100 node + DeepSpeed). See **[RUNBOOK_500B.md](RUNBOOK_500B.md)**.
- **CyberPal-2.0-20B** — 20B security expert, defense-leaning.
- **Foundation-Sec-8B** (Cisco, Llama-3.1-8B, ~5.1B security tokens) — smaller,
  defense-leaning, most permissive license.

WhiteRabbitNeo carries its own usage terms (retained in `NOTICE`); verify them
before commercial distribution.

## Identity, honestly

Darknode is the identity the operator sees: the persona says "Darknode AI," the
model tag is `darknode`, and no base-model plumbing is exposed at runtime. What
we do **not** do is forge the origin — the base model and datasets are
Llama-derived and carry attribution requirements, so their notice lives in the
repo `NOTICE` file (regenerate with `python -m darknode_ai.foundation.notice`).
Branding the assistant is fine; stripping a required license notice is a license
violation and a liability for a security product, so we keep it in-repo.

## Clean-provenance data only

`dataprep.py` builds SFT data from Darknode's own authored + synthetic corpus,
plus optional offensive/defensive datasets via `--dataset NAME=PATH` (known field
maps in `KNOWN_DATASETS`): `pentest-v2`, `whiterabbitneo`, `nyu-ctf`,
`redsage-conv`. Each carries its license/provenance; non-commercial sets
(e.g. `pentest-v2`, which draws on CC-BY-NC HackTricks) warn loudly, and
provenance-unverified synthetic sets (`redsage-conv`) require `--allow-review`.

It **refuses** sets distilled from another vendor's proprietary model —
`Primus-Instruct` (GPT-4o), `Primus-Reasoning` (o1), and any `Fable-5-traces`
(captured Claude output). That guard is enforced in code (`_REFUSED_SOURCES`) and
documented in `NOTICE`.

## Scope

The persona is an expert offensive **and** defensive operator inside an
authorized-engagement frame (pentest, red team, CTF, research, lab, defense). It
gives real technical depth for authorized work. It does not assist real-world
unauthorized intrusion, indiscriminate/destructive malware, or credential theft
against real victims. That boundary is in the system prompt, stated once as an
operating principle — not a per-turn disclaimer.

## Build it

```bash
# 1. clean SFT data (offline corpus + offensive datasets you downloaded locally)
python -m darknode_ai.foundation.dataprep --out data/sft \
    --dataset pentest-v2=./pentest-v2.jsonl \
    --dataset nyu-ctf=./nyu-ctf.jsonl

# 2. LoRA fine-tune on a GPU box or Colab/again GPU runtime
pip install -r requirements-foundation.txt
python -m darknode_ai.foundation.finetune \
    --base WhiteRabbitNeo/WhiteRabbitNeo-13B-v1 --data data/sft \
    --out runs/darknode-foundation --i-have-a-gpu

# 3. merge adapter + convert to GGUF (llama.cpp), then package for Ollama
python -m darknode_ai.foundation.modelfile \
    --mode gguf --gguf ./darknode.gguf \
    --out runs/darknode-foundation/Modelfile
ollama create darknode -f runs/darknode-foundation/Modelfile

# 4. serve Darknode (foundation backend)
DARKNODE_BACKEND=ollama DARKNODE_MODEL=darknode python -m darknode_ai.serve.app
```

Step 3's adapter→GGUF merge uses PEFT `merge_and_unload` + llama.cpp
`convert_hf_to_gguf.py`; on a base already in Ollama you can instead use
`--mode pull --base <ollama-tag> --adapter <dir>`.

## Evaluate before promoting

Gate every version with security benchmarks (CTI-Bench, AttackQA held-out) plus
the existing behavioural probes, and only then register/deploy it. RAG (CVE/NVD,
your docs) supplies live facts at inference — keep facts in retrieval, not baked
into weights.
