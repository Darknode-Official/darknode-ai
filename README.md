# Darknode AI

A cybersecurity language model with **two tracks** and a shared retrieval layer.

1. **From-scratch** (`darknode_ai/model`, `train`) — its own tokenizer,
   transformer, and training loop. No base model, no pretrained weights. Runs
   end-to-end in **Google Colab** and, small, on CPU. This is the "we built the
   architecture ourselves" track; it is honestly small.
2. **Foundation** (`darknode_ai/foundation`) — specializes a strong, openly
   licensed **security** base (default WhiteRabbitNeo-13B, an offensive+defensive
   red/blue-team model) into Darknode with LoRA on clean-provenance data, served
   under Darknode's own identity via Ollama. This is the **capable** track. See
   [FOUNDATION.md](FOUNDATION.md).
3. **Retrieval / RAG** (`darknode_ai/retrieval`) — a dependency-free BM25
   knowledge store that grounds either model in RETRIEVED, source-traced evidence
   (CVEs, your docs) instead of parametric guesswork.

> **Honest scope.** The from-scratch model is *small* — coherent domain text,
> not large-model reasoning. The foundation model is a genuinely capable
> specialized assistant, but bounded by its base size, not frontier-tier. Facts
> live in retrieval, not weights.

## What's in the box

| Stage | Module | From scratch? |
|-------|--------|---------------|
| Tokenizer | `darknode_ai/tokenizer/bpe.py` | Byte-level BPE, pure Python |
| Model | `darknode_ai/model/gpt.py` | Decoder-only transformer (RoPE, RMSNorm, SwiGLU) in PyTorch |
| Data governance | `darknode_ai/data/redact.py`, `pipeline.py` | Secret redaction, dedup, provenance manifest, packed `.bin` |
| Corpus | `darknode_ai/data/corpus.py` | Self-authored defensive knowledge + synthetic evidence-typed cases (CC0) |
| Training | `darknode_ai/train/trainer.py` | AdamW + cosine warmup + grad-accum + AMP + checkpoint/resume |
| Evaluation | `darknode_ai/eval/` | Perplexity + behavioural probes (evidence-typing, hedging, no-action-without-approval) |
| Registry | `darknode_ai/registry.py` | Versioned model + dataset + metrics, deploy/rollback |
| Inference | `darknode_ai/sample.py` | Temperature / top-k / top-p sampler |
| Foundation | `darknode_ai/foundation/` | Persona, Ollama provider, clean SFT data prep, LoRA recipe, Modelfile, NOTICE |
| Retrieval | `darknode_ai/retrieval/` | BM25 store (`index`) + RAG grounding (RETRIEVED evidence) |
| Agent | `darknode_ai/agent/` | Tool-calling ReAct loop (`agent`) — plan/act/observe; approval-gated tools |
| Serving | `darknode_ai/serve/` | FastAPI panel; `DARKNODE_BACKEND=scratch\|ollama`, optional `DARKNODE_KNOWLEDGE` RAG |

## The house style it is trained toward

Every synthetic case teaches the model to separate **observation from
speculation** using evidence tags — a core requirement for security work:

```
OBSERVED     — in the data / logs
RETRIEVED    — from a knowledge source
INFERRED     — a pattern-based deduction
HYPOTHESIS   — a candidate explanation, not confirmed
RECOMMENDATION — a suggested next step (never an executed action)
UNKNOWN      — insufficient evidence
```

## Quick start (local, CPU, tiny — a real end-to-end run)

```bash
pip install -e .
darknode-ai pipeline --preset tiny --vocab-size 2048
```

That builds the corpus, trains the tokenizer, redacts + packs the dataset, and
trains a tiny model for a few steps — the whole pipeline, verified.

## Full run (steps)

```bash
darknode-ai corpus                       # data/corpus/*.txt + manifest.json
darknode-ai tokenizer --vocab-size 8192  # runs/tokenizer.json
darknode-ai prepare                      # data/prepared/{train,val}.bin + meta.json
darknode-ai train  --preset small        # runs/darknode-small/ckpt.pt
darknode-ai eval   --ckpt runs/darknode-small/best.pt
darknode-ai sample --ckpt runs/darknode-small/best.pt \
  --prompt $'<|user|> Triage: many failed logins then one success.\n<|assistant|>\n'
darknode-ai register --ckpt runs/darknode-small/best.pt --version v0.1.0
```

## Retrieval (RAG)

```bash
darknode-ai index                                  # runs/knowledge.json (BM25)
darknode-ai index --jsonl cve.jsonl:description:nvd # + external facts
# serve the foundation model grounded in the store:
DARKNODE_BACKEND=ollama DARKNODE_MODEL=darknode \
  DARKNODE_KNOWLEDGE=runs/knowledge.json python -m darknode_ai.serve.app
```

Every retrieved chunk is returned tagged `RETRIEVED [source | provenance]`, so
answers are traceable to a source and gaps surface as `UNKNOWN`.

## Agentic mode (like Claude Code)

The `agent/` harness turns the foundation model into an agent: a ReAct-style
tool loop that plans, calls tools, observes, and iterates. Parameter count makes
each step smarter; the harness is what makes it an agent — so it works from 13B
to 500B, and a bigger base simply reasons better per step.

```bash
darknode-ai index                          # build the knowledge base first
darknode-ai agent "Audit ./config for exposed secrets and summarize findings."
darknode-ai agent "..." --autonomous       # approve write/shell tools (sandbox only)
```

Tools: `rag_search` (grounded facts), `read_file`, `list_dir` (read-only, run
freely) and `write_file`, `run_shell` (mutating — **denied by default**, gated
through an approval callback; `--autonomous` approves them for authorized/lab
use). In **Nexus**, pick it with `/engine darknode` — it runs through Nexus's
local tool loop with full read/write/edit/run access, private and Darknode-branded.

## Foundation model (capable track)

See [FOUNDATION.md](FOUNDATION.md): build clean SFT data (`dataprep`), LoRA
fine-tune a security base on a GPU (`finetune`), package for Ollama
(`modelfile` -> `ollama create darknode`), and serve. Base and datasets are
attributed in `NOTICE`; distilled-from-a-vendor sets are refused by the
provenance guard.

## Colab

Open `notebooks/darknode_ai_colab.ipynb` in Google Colab, pick a GPU runtime,
Run All. It trains the `small` from-scratch preset and saves checkpoints to
Google Drive. See `notebooks/` for details.

## Tests

```bash
pytest -q          # tokenizer, redaction, data pipeline, registry (no GPU)
                   # model/training tests run automatically if torch is installed
```

## Safety posture

- **From-scratch track:** defensive by construction — corpus, probes and style
  push evidence-based, hedged, human-in-the-loop analysis.
- **Foundation track:** an expert **offensive + defensive** operator inside an
  **authorized-engagement frame** (pentest, red team, CTF, research, lab). It
  gives real technical depth for authorized work; it does not assist real-world
  unauthorized intrusion, indiscriminate/destructive malware, or credential theft
  against real victims (persona operating principle).
- **Provenance:** secret/PII redaction is an enforced gate before any text is
  used; training data distilled from another vendor's proprietary model is
  refused (`foundation.dataprep` guard); base/dataset attribution is retained in
  `NOTICE`. The server never executes actions — it only generates text.

See `DATA_GOVERNANCE.md` and `FOUNDATION.md`.

## Scaling up

The same code trains a bigger model when you add data and compute:
- grow `--vocab-size` and the `small`/custom `DarknodeGPTConfig`,
- add more **rights-clean** sources to `data/manifest.json` (each with license +
  provenance) — the pipeline redacts, dedups and versions them,
- track every run in the registry and compare perplexity + probe rates before
  promoting a version.
