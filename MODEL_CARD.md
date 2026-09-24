# Model Card — Darknode AI (from-scratch)

## Overview
Darknode AI is a small decoder-only transformer trained from random
initialization (no pretrained base) on a curated, rights-clean corpus of
defensive-security text. It is a foundation/prototype in a versioned stack, not
a production reasoning assistant.

## Architecture
- Decoder-only transformer: RoPE positions, RMSNorm (pre-norm), SwiGLU MLP,
  weight-tied embeddings, causal self-attention.
- Presets: `tiny` (~0.1–1M params, CPU/test), `small` (~15–35M params, Colab T4).
- Byte-level BPE tokenizer trained on the same corpus (default vocab 8192),
  lossless on arbitrary bytes (IPs, hashes, base64, JSON, logs).

## Training data
- `defensive-knowledge` — hand-authored explainers (networking, Linux, IR,
  detection engineering, log analysis, secure coding, vuln management, ATT&CK
  defensive framing). License CC0, provenance `darknode-authored`.
- `synthetic-triage` — generated evidence-typed triage cases teaching the
  OBSERVED/INFERRED/HYPOTHESIS/RECOMMENDATION/UNKNOWN style. License CC0.
- All text passes secret/PII redaction before tokenization. No scraped data.

## Intended use
- Generating and completing defensive-security domain text in Darknode's style.
- A research/education substrate: study tokenization, training dynamics, eval.
- A base for further fine-tuning as corpus and compute grow.

## Out of scope / not intended
- Authoritative security judgments or autonomous actions.
- Offensive tooling, exploitation, or any unauthorized activity.
- Factual lookups (CVE details, live intel) — that belongs in a retrieval layer,
  not model weights.

## Known limitations
- Small scale ⇒ limited world knowledge and reasoning; can produce fluent but
  incorrect text. Treat all output as a draft requiring analyst verification.
- Trained on a compact corpus ⇒ prone to memorization/overfit at small data
  sizes; expand the corpus before drawing conclusions from samples.

## Evaluation
Tracked per version in `runs/registry.json`:
- held-out perplexity,
- behavioural probe hit-rate (uses evidence tags, hedges, defers actions to
  approval),
- unsafe-phrase count (asserting certainty like "definitely compromised").

## Safety
Defensive-only corpus and objectives; enforced secret redaction; evaluation
penalizes overconfident/unsafe phrasing. Human-in-the-loop is assumed for any
downstream use.
