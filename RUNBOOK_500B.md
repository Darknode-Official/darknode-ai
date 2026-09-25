# Darknode 500B — cluster runbook

500B is a different class of problem from 100B. It is **cluster-scale**: you are
LoRA-tuning a ~550B MoE base across a multi-GPU node with a distributed training
stack. The Darknode pipeline provides the config, data, persona, and entrypoint;
the actual run needs rented cluster infrastructure and a DeepSpeed/expert-parallel
launch that you configure on the cluster.

## The base

`--preset 500b` targets **NVIDIA Nemotron 3 Ultra** — MoE, ~550B total / ~55B
active, 1M context. Alternatives (swap `--base`):
- `deepseek-ai/DeepSeek-V3` — 671B / ~37B active (well-documented LoRA path).
- Thinking Machines **Inkling** — 975B / ~41B active, Apache-2.0.

> Verify the exact HF repo id and license before running. The assumed tag
> `nvidia/Nemotron-3-Ultra-550B` is a placeholder.

## Hardware, cost, and why plain QLoRA isn't enough

A 550B MoE in FP8 is ~560GB+ of weights — more than a single 8xH100 node's 640GB
once activations and optimizer state are added. Reference points from the
comparable DeepSeek-671B LoRA work:

| Approach | Hardware | Rough cost |
|----------|----------|-----------|
| LoRA, 4-bit (AWQ) + CPU offload | **8x H100 80GB** (640GB) minimum | node ~$23/hr; a run = **hundreds** of $ |
| LoRA, mixed parallelism (ZeRO + PP + EP) | **~24x H100** for real throughput | **low thousands** of $ per run |
| Full fine-tune | many nodes | up to ~$240k (don't) |

Plain `device_map="auto"` (what the 13b/100b path uses) can shard the model
across one node, but at 500B you need a distributed strategy for it to be
tractable:
- **DeepSpeed ZeRO-3** (optimizer/param sharding) + **CPU/NVMe offload**,
- **expert parallelism (EP)** and **pipeline parallelism (PP)** for the MoE layers,
- 4-bit / AWQ quantization of the frozen base.

## Steps (on a rented 8x H100 node)

```bash
# 0. (local) build clean SFT data; copy data/sft to the cluster
python -m darknode_ai.foundation.dataprep --out data/sft \
    --dataset pentest-v2=./pentest-v2.jsonl --dataset nyu-ctf=./nyu-ctf.jsonl

# 1. (cluster) env + a DeepSpeed ZeRO-3 config (ds_zero3.json) with
#    offload_optimizer/offload_param = cpu, plus an accelerate config declaring
#    the GPU count and DeepSpeed plugin.
pip install -r requirements-foundation.txt deepspeed

# 2. verify the base tag exists, then launch distributed QLoRA
accelerate launch --config_file accelerate_ds_zero3.yaml \
    -m darknode_ai.foundation.finetune --preset 500b --data data/sft --i-have-a-gpu
#    -> adapter in runs/darknode-500b/

# 3. serve: a 550B MoE needs a multi-GPU inference server (vLLM/SGLang with
#    tensor+expert parallel), not a laptop Ollama. Point the Darknode provider at
#    that server's OpenAI-compatible endpoint, or run Ollama on the 8xH100 host.
```

The finetune entrypoint stays the same; the difference at 500B is the launcher
(`accelerate launch` + DeepSpeed) and the cluster, which live outside this repo
because they are environment-specific.

## Honest expectations

- This is real infrastructure and real money (hundreds to low-thousands per run,
  plus a serving cluster). It is not a hobby setup.
- Each size jump (13B -> 100B -> 500B) multiplies cost far faster than it adds
  Darknode-specific capability, because we are LoRA-specializing — the base's
  pretraining sets the ceiling, and our security data specializes the top layer
  either way. A 100B Darknode already gives most of the practical benefit at a
  tiny fraction of this cost.
- Scope/provenance rules are unchanged: distilled-vendor data refused,
  attribution in NOTICE, authorized-engagement persona intact.
