# Data Governance

Training data is only as safe as its sources. Darknode AI enforces governance in
code, not just in policy.

## Source manifest
Every source is declared in `data/manifest.json` with:
`path`, `license`, `category`, `provenance`, `allowed_use`.
The pipeline **skips** any source whose `allowed_use != "train"`. Do not add a
source you do not have the rights to train on.

## Redaction gate (enforced)
`darknode_ai/data/redact.py` runs on every document before tokenization and
replaces secrets/PII with typed placeholders (`<|redacted:TYPE|>`):
AWS keys, GitHub/Slack/Google tokens, JWTs, bearer tokens, private keys,
password/secret key-values, connection strings, emails, card-like numbers.

The pipeline computes a **secret density** (redactions per 1k chars) and
**refuses to build** the dataset if it exceeds `max_secret_density` (default 5).
A high-density source is a signal of unsafe/unauthorized data — inspect it.

## Deduplication & versioning
Documents are deduplicated by SHA-256. The final dataset gets a content-hash
**dataset version** recorded in `meta.json` and in the model registry, so every
model is traceable to the exact data it saw.

## What must never enter weights
Credentials, tokens, private keys, session cookies, real private case data,
unfiltered logs, sensitive infrastructure detail. If you must train on private
telemetry, redact and authorize it first — and record that authorization in the
manifest `provenance` field.

## Reproducibility
The seed corpus is generated deterministically (`corpus.py`, seeded). Rebuilding
from the same manifest yields the same dataset version.
