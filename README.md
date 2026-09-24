# Darknode AI

A cybersecurity language model **trained from scratch** — its own tokenizer, its
own transformer, its own training loop. No base model, no API wrapper, no
pretrained weights. Built to run end-to-end in a **Google Colab** notebook and,
at small scale, on CPU.

> **Honest scope.** A model trained from scratch on Colab-scale compute is a
> *small* model. It learns the vocabulary, structure and house-style of
> defensive-security text and generates coherent domain language. It is **not**
> a replacement for a large model's reasoning. The value here is a fully owned,
> inspectable, reproducible stack that scales up as you add corpus and compute —
> every component is the real thing, sized down.

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

## Colab

Open `notebooks/darknode_ai_colab.ipynb` in Google Colab, pick a GPU runtime,
Run All. It trains the `small` preset and saves checkpoints to Google Drive. See
`notebooks/` for details.

## Tests

```bash
pytest -q          # tokenizer, redaction, data pipeline, registry (no GPU)
                   # model/training tests run automatically if torch is installed
```

## Safety posture

Defensive by construction. The corpus, probes and generation style push toward
evidence-based, hedged, human-in-the-loop analysis. Secret redaction is an
enforced gate before any text is tokenized. See `DATA_GOVERNANCE.md`.

## Scaling up

The same code trains a bigger model when you add data and compute:
- grow `--vocab-size` and the `small`/custom `DarknodeGPTConfig`,
- add more **rights-clean** sources to `data/manifest.json` (each with license +
  provenance) — the pipeline redacts, dedups and versions them,
- track every run in the registry and compare perplexity + probe rates before
  promoting a version.
