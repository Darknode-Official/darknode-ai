"""LoRA fine-tune recipe: specialize a security base model into Darknode.

This is the training entrypoint for the foundation track. It fine-tunes an
openly-licensed security base (default: Cisco Foundation-Sec-8B) with LoRA on the
clean-provenance SFT data built by foundation/dataprep.py, then saves an adapter.

Requirements (NOT installed by the core package -- this needs a GPU box or a
Colab/again GPU runtime): torch, transformers, peft, trl, datasets, accelerate,
bitsandbytes. See requirements-foundation.txt. The script is guarded so it does
not silently try to allocate an 8B model on a CPU laptop.

Typical flow:
    python -m darknode_ai.foundation.dataprep --out data/sft
    python -m darknode_ai.foundation.finetune --data data/sft \\
        --base fdtn-ai/Foundation-Sec-8B --out runs/darknode-foundation \\
        --i-have-a-gpu
    python -m darknode_ai.foundation.modelfile --adapter runs/darknode-foundation \\
        --base fdtn-ai/Foundation-Sec-8B --out runs/darknode-foundation/Modelfile
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    # Default: WhiteRabbitNeo-13B-v1 -- an offensive+defensive security base,
    # single-GPU QLoRA-able. Use LoRAConfig.preset("33b"/"100b") to scale up, or
    # --base for any HF model. See RUNBOOK_100B.md for the 100B path.
    base_model: str = "WhiteRabbitNeo/WhiteRabbitNeo-13B-v1"
    data_dir: str = "data/sft"
    out_dir: str = "runs/darknode-foundation"
    epochs: float = 3.0
    lr: float = 2e-4
    batch_size: int = 4
    grad_accum: int = 8
    max_seq_len: int = 4096
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = True
    gradient_checkpointing: bool = True
    max_steps: int | None = None   # cap steps (fast trial); overrides epochs when set
    seed: int = 0

    @classmethod
    def preset(cls, name: str) -> "LoRAConfig":
        name = name.lower()
        if name in ("13b", "default"):
            return cls()
        if name == "33b":
            # WhiteRabbitNeo-33B: bigger offensive base; ~1x A100/H100 80GB QLoRA.
            return cls(base_model="WhiteRabbitNeo/WhiteRabbitNeo-33B-v1",
                       out_dir="runs/darknode-33b", batch_size=2, grad_accum=16)
        if name == "100b":
            # 100B+ MoE base: 122B total / ~10B active. QLoRA on 1x A100/H100 80GB
            # (device_map shards across more GPUs if present). VERIFY the exact HF
            # repo id on Hugging Face before running -- Qwen3.5 122B MoE tag.
            return cls(base_model="Qwen/Qwen3.5-122B-A10B",
                       out_dir="runs/darknode-100b", batch_size=1, grad_accum=32,
                       max_seq_len=4096, lora_r=16, lora_alpha=32)
        if name == "500b":
            # 500B+ MoE base: Nemotron 3 Ultra, ~550B total / ~55B active. This is
            # CLUSTER scale: needs an 8x H100 80GB node minimum (4-bit + offload),
            # and DeepSpeed ZeRO-3 + expert parallelism for real throughput -- plain
            # device_map is not enough. See RUNBOOK_500B.md. VERIFY the HF repo id.
            return cls(base_model="nvidia/Nemotron-3-Ultra-550B",
                       out_dir="runs/darknode-500b", batch_size=1, grad_accum=64,
                       max_seq_len=4096, lora_r=8, lora_alpha=16)
        raise ValueError(f"unknown preset '{name}' (13b|33b|100b|500b)")


def run(cfg: LoRAConfig):
    # Heavy deps imported lazily so `import`/tests do not require them.
    import torch
    from datasets import load_dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU detected. Fine-tuning an 8B base needs a GPU (a free "
            "Colab/again T4 works for LoRA-4bit). Aborting rather than thrashing "
            "a CPU. Run on a GPU box or pass a smaller --base for a smoke test.")

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        # Many security bases (WhiteRabbitNeo / Llama-2 lineage) ship no chat
        # template, so apply_chat_template() would crash. Set a simple, consistent
        # instruction format so SFT can render our {system,user,assistant} messages.
        tok.chat_template = (
            "{% for m in messages %}"
            "{% if m['role'] == 'system' %}{{ m['content'] + '\n\n' }}"
            "{% elif m['role'] == 'user' %}{{ '### Instruction:\n' + m['content'] + '\n\n' }}"
            "{% elif m['role'] == 'assistant' %}{{ '### Response:\n' + m['content'] + eos_token + '\n\n' }}"
            "{% endif %}{% endfor %}"
        )

    quant = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    ) if cfg.load_in_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, quantization_config=quant, torch_dtype=torch.bfloat16,
        device_map="auto")  # shards a 100B base across all visible GPUs
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.gradient_checkpointing)

    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files={
        "train": f"{cfg.data_dir}/train.jsonl",
        "validation": f"{cfg.data_dir}/val.jsonl"})

    def format_chat(batch):
        return {"text": [tok.apply_chat_template(m, tokenize=False,
                                                 add_generation_prompt=False)
                         for m in batch["messages"]]}

    ds = ds.map(format_chat, batched=True, remove_columns=ds["train"].column_names)

    # TRL's SFTConfig/SFTTrainer signatures drift between versions (e.g. trl 0.9
    # vs 1.13): max_seq_length -> max_length, warmup_ratio dropped, tokenizer ->
    # processing_class. Build the kwargs, adapt the renames, then filter each dict
    # to what the *installed* signature actually accepts so one file runs on any TRL.
    import inspect
    sft_params = set(inspect.signature(SFTConfig.__init__).parameters)
    trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters)

    sft_kw = dict(
        output_dir=cfg.out_dir, num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps if cfg.max_steps else -1,  # -1 = use epochs
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True, logging_steps=10, eval_strategy="epoch",
        save_strategy="epoch", max_seq_length=cfg.max_seq_len,
        dataset_text_field="text",
        gradient_checkpointing=cfg.gradient_checkpointing,
        seed=cfg.seed, report_to=[])
    if "max_seq_length" not in sft_params and "max_length" in sft_params:
        sft_kw["max_length"] = sft_kw.pop("max_seq_length")  # renamed in trl 1.x
    sft_kw = {k: v for k, v in sft_kw.items() if k in sft_params}

    trainer_kw = dict(
        model=model, train_dataset=ds["train"], eval_dataset=ds["validation"],
        args=SFTConfig(**sft_kw))
    if "processing_class" in trainer_params:      # trl 1.x
        trainer_kw["processing_class"] = tok
    elif "tokenizer" in trainer_params:           # trl 0.x
        trainer_kw["tokenizer"] = tok
    trainer = SFTTrainer(**trainer_kw)
    trainer.train()
    trainer.save_model(cfg.out_dir)
    tok.save_pretrained(cfg.out_dir)
    print(f"Darknode LoRA adapter saved -> {cfg.out_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="LoRA fine-tune Darknode foundation")
    ap.add_argument("--preset", choices=["13b", "33b", "100b", "500b"], default=None,
                    help="size preset (sets base + hyperparams). 100b: 122B MoE on "
                         "1x A100/H100 (RUNBOOK_100B.md). 500b: 550B MoE, needs an "
                         "8x H100 node + DeepSpeed (RUNBOOK_500B.md).")
    ap.add_argument("--base", default=None, help="override the base model repo id")
    ap.add_argument("--data", default=LoRAConfig.data_dir)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None,
                    help="per-device batch size (use 1 to fit a 13B on a free T4)")
    ap.add_argument("--grad-accum", type=int, default=None,
                    help="gradient accumulation steps (raise to keep effective batch)")
    ap.add_argument("--seq-len", type=int, default=None,
                    help="max sequence length (lower, e.g. 1024, to fit small GPUs)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap training steps for a fast/cheap trial (overrides epochs)")
    ap.add_argument("--lora-r", type=int, default=None, help="LoRA rank override")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--i-have-a-gpu", action="store_true",
                    help="acknowledge this allocates a large model on GPU(s)")
    args = ap.parse_args(argv)
    if not args.i_have_a_gpu:
        raise SystemExit(
            "Fine-tuning the foundation model needs a GPU (the 100b preset needs a "
            "rented A100/H100 80GB). Re-run with --i-have-a-gpu on a GPU box. On "
            "this machine, use the from-scratch track (darknode-ai train) instead.")
    cfg = LoRAConfig.preset(args.preset) if args.preset else LoRAConfig()
    if args.base:
        cfg.base_model = args.base
    cfg.data_dir = args.data
    if args.out:
        cfg.out_dir = args.out
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch is not None:
        cfg.batch_size = args.batch
    if args.grad_accum is not None:
        cfg.grad_accum = args.grad_accum
    if args.seq_len is not None:
        cfg.max_seq_len = args.seq_len
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.lora_r is not None:
        cfg.lora_r = args.lora_r
    cfg.load_in_4bit = not args.no_4bit
    run(cfg)


if __name__ == "__main__":
    main()
