"""From-scratch training loop: AdamW, cosine LR with warmup, grad accumulation,
mixed precision, gradient clipping, checkpoint/resume, JSONL metric logging.

No trainer library — the loop is written out so every step is inspectable.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from darknode_ai.model.gpt import DarknodeGPT, DarknodeGPTConfig
from darknode_ai.train.config import TrainConfig


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(name: str, device: str):
    if name == "auto":
        if device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if device == "cuda" else torch.float32
    return {"float32": torch.float32, "bfloat16": torch.bfloat16,
            "float16": torch.float16}[name]


class DataLoader:
    """Samples random contiguous windows from a packed uint16 .bin memmap."""

    def __init__(self, data_dir: str, split: str, context_len: int,
                 batch_size: int, device: str):
        path = Path(data_dir) / f"{split}.bin"
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.context_len = context_len
        self.batch_size = batch_size
        self.device = device
        if len(self.data) <= context_len + 1:
            raise ValueError(f"{split}.bin too small ({len(self.data)} tokens) "
                             f"for context {context_len}")

    def batch(self):
        ix = torch.randint(len(self.data) - self.context_len - 1, (self.batch_size,))
        x = torch.stack([torch.from_numpy(
            self.data[i:i + self.context_len].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(
            self.data[i + 1:i + 1 + self.context_len].astype(np.int64)) for i in ix])
        if self.device == "cuda":
            return x.pin_memory().to("cuda", non_blocking=True), \
                   y.pin_memory().to("cuda", non_blocking=True)
        return x.to(self.device), y.to(self.device)


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if step > cfg.max_steps:
        return cfg.min_lr
    ratio = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, loaders, iters):
    model.eval()
    out = {}
    for split, loader in loaders.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = loader.batch()
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = float(losses.mean())
    model.train()
    return out


def train(model_cfg: DarknodeGPTConfig, cfg: TrainConfig, resume: bool = False):
    torch.manual_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    dtype = _resolve_dtype(cfg.dtype, device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_fp = (out_dir / "metrics.jsonl").open("a", encoding="utf-8")

    model = DarknodeGPT(model_cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                              betas=(cfg.beta1, cfg.beta2),
                              weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    start_step = 0
    best_val = float("inf")

    ckpt_path = out_dir / "ckpt.pt"
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        start_step = ck["step"] + 1
        best_val = ck.get("best_val", best_val)
        print(f"resumed from step {start_step}")

    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    loaders = {"train": DataLoader(cfg.data_dir, "train", model_cfg.context_len,
                                   cfg.batch_size, device)}
    if (Path(cfg.data_dir) / "val.bin").exists():
        loaders["val"] = DataLoader(cfg.data_dir, "val", model_cfg.context_len,
                                    cfg.batch_size, device)

    print(f"device={device} dtype={dtype} params={model.num_params()/1e6:.2f}M "
          f"steps={cfg.max_steps}")
    autocast = torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                              dtype=dtype, enabled=(device == "cuda"))

    t0 = time.time()
    model.train()
    for step in range(start_step, cfg.max_steps + 1):
        lr = cosine_lr(step, cfg)
        for g in optim.param_groups:
            g["lr"] = lr

        optim.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(cfg.grad_accum):
            x, y = loaders["train"].batch()
            with autocast:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum
            scaler.scale(loss).backward()
            loss_accum += loss.item()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optim)
        scaler.update()

        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            rec = {"step": step, "loss": round(loss_accum, 4), "lr": round(lr, 6),
                   "sec": round(dt, 1)}
            print(f"step {step:5d} | loss {loss_accum:.4f} | lr {lr:.2e} | {dt:.0f}s")
            log_fp.write(json.dumps(rec) + "\n"); log_fp.flush()

        if step % cfg.eval_interval == 0 and step > 0:
            metrics = estimate_loss(model, loaders, cfg.eval_iters)
            val = metrics.get("val", metrics["train"])
            ppl = math.exp(min(20, val))
            rec = {"step": step, "eval": metrics, "val_ppl": round(ppl, 3)}
            print(f"  eval step {step}: {metrics} ppl={ppl:.2f}")
            log_fp.write(json.dumps(rec) + "\n"); log_fp.flush()
            if val < best_val:
                best_val = val
                _save(out_dir / "best.pt", model, optim, step, model_cfg, cfg, best_val)

        if step % cfg.ckpt_interval == 0 and step > 0:
            _save(ckpt_path, model, optim, step, model_cfg, cfg, best_val)

    _save(ckpt_path, model, optim, cfg.max_steps, model_cfg, cfg, best_val)
    log_fp.close()
    return {"best_val": best_val, "out_dir": str(out_dir)}


def _save(path, model, optim, step, model_cfg, cfg, best_val):
    sd = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    torch.save({
        "model": sd,
        "optim": optim.state_dict(),
        "step": step,
        "best_val": best_val,
        "model_config": model_cfg.__dict__,
        "train_config": cfg.dict(),
    }, path)
