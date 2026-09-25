"""Darknode AI — foundation (open-base) track.

The from-scratch track (`darknode_ai.model`) builds a transformer from random
init: honest provenance, but capped by a tiny corpus. This track instead
*specialises* a strong, openly-licensed security base model (Cisco
Foundation-Sec-8B, a Llama-3.1-8B pretrained on cybersecurity data) into
Darknode with LoRA fine-tuning on clean-provenance data, then serves it under
Darknode's own identity via Ollama.

Design rules for this track:
  - Darknode is the product identity the user sees; the base model is plumbing.
  - Base-model and dataset attribution is preserved in NOTICE (license
    compliance) -- we brand the persona, we do not forge the origin.
  - Training data is clean-provenance only. Sets distilled from another
    vendor's proprietary model (GPT-4o / o1 / Claude traces) are refused here.
  - The persona operates in an authorized-engagement frame (pentest, red team,
    CTF, research, lab, defense). Expert offense and defense; not a tool for
    real-world unauthorized intrusion.
"""
