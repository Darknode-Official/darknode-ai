# Darknode 100B — cloud runbook

Making Darknode a 100B-parameter model means **LoRA-specializing a ~100B open
base**, not gathering more data (params come from the base's architecture, not
the corpus). This box has no GPU/network, so the training runs on **rented
cloud GPUs**. Everything below is reproducible; only step 3 needs the cloud.

## The base

Default 100B target: **Qwen3.5-122B** — a Mixture-of-Experts model, ~122B total
params / ~10B active per token. MoE is the reason 100B is feasible: you get the
parameter count but only a few billion fire per token, so it QLoRA-tunes and
serves on a single 80GB GPU instead of a cluster.

> Verify the exact Hugging Face repo id before running (`Qwen/Qwen3.5-122B-A10B`
> is the assumed tag). Dense 100B alternatives: `mistralai/Mistral-Large-*`
> (~123B, heavier — needs 2+ GPUs), `CohereForAI/c4ai-command-r-plus` (~104B).

## Hardware & cost (be realistic)

| Step | Needs | Rough cost |
|------|-------|-----------|
| QLoRA fine-tune (122B MoE, 4-bit) | 1x A100/H100 80GB | ~$30–80 per run |
| QLoRA fine-tune (dense ~120B) | 2x A100/H100 80GB | ~$100–200 per run |
| Serving (4-bit) | 64–80GB VRAM (or 64GB Mac for Qwen) | rent per hour |

Providers: RunPod, io.net, Lambda, Vast.ai. Storage for weights (~60–250GB) adds
a little. QLoRA keeps the base frozen (4-bit) and trains a ~30–60M-param adapter.

## Steps

```bash
# 0. (local, this repo) build clean SFT data + offensive datasets
python -m darknode_ai.foundation.dataprep --out data/sft \
    --dataset pentest-v2=./pentest-v2.jsonl --dataset nyu-ctf=./nyu-ctf.jsonl
#    scp data/sft to the rented box, or rebuild it there from this repo.

# 1. (rented A100/H100 box) install
pip install -r requirements-foundation.txt

# 2. confirm the base tag exists, then QLoRA fine-tune the 100B preset
python -m darknode_ai.foundation.finetune --preset 100b --data data/sft \
    --i-have-a-gpu
#    -> adapter in runs/darknode-100b/ . device_map="auto" shards across all
#       visible GPUs; gradient checkpointing is on.

# 3. merge adapter -> GGUF (llama.cpp) OR keep adapter for vLLM serving
#    then package for Ollama (needs an 80GB serving host):
python -m darknode_ai.foundation.modelfile --mode gguf \
    --gguf ./darknode-100b.gguf --out runs/darknode-100b/Modelfile
ollama create darknode -f runs/darknode-100b/Modelfile

# 4. serve, grounded in retrieval
DARKNODE_BACKEND=ollama DARKNODE_MODEL=darknode \
  DARKNODE_KNOWLEDGE=runs/knowledge.json python -m darknode_ai.serve.app
```

## Honest expectations

- This is a genuine 100B+ parameter Darknode, and it will be materially more
  capable than the 13B. It is still not frontier-lab tier — the base's training,
  not just its size, sets the ceiling.
- The corpus we have is enough to *specialize* it; you do not need 100B params'
  worth of new data to fine-tune (that scale of data is only for pretraining from
  scratch, which we are not doing).
- Provenance/scope rules are unchanged: distilled-from-a-vendor data stays
  refused, attribution stays in `NOTICE`, and the persona keeps the
  authorized-engagement frame.
