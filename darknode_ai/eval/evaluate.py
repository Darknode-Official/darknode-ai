"""Evaluate a trained checkpoint: perplexity + behavioural probes + samples."""
from __future__ import annotations

import math
from pathlib import Path

import torch

from darknode_ai.eval.security_suite import HELDOUT_TEXT, PROBES, score_continuation
from darknode_ai.model.gpt import DarknodeGPT, DarknodeGPTConfig
from darknode_ai.tokenizer.bpe import BPETokenizer


def load_checkpoint(ckpt_path: str, device: str = "cpu"):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = DarknodeGPTConfig(**ck["model_config"])
    model = DarknodeGPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def perplexity(model, tok: BPETokenizer, text: str, device="cpu") -> float:
    ids = tok.encode(text, allowed_special=False)
    ctx = model.cfg.context_len
    ids = ids[: ctx + 1] if len(ids) > ctx + 1 else ids
    if len(ids) < 2:
        return float("nan")
    x = torch.tensor([ids[:-1]], device=device)
    y = torch.tensor([ids[1:]], device=device)
    _, loss = model(x, y)
    return math.exp(min(20, loss.item()))


def evaluate(ckpt_path: str, tokenizer_path: str, device: str = "cpu",
             max_new_tokens: int = 80) -> dict:
    model, _ = load_checkpoint(ckpt_path, device)
    tok = BPETokenizer.load(tokenizer_path)
    eos = tok.special.get("<|endoftext|>")

    ppl = perplexity(model, tok, HELDOUT_TEXT, device)
    results = []
    for probe in PROBES:
        ids = torch.tensor([tok.encode(probe["prompt"])], device=device)
        out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.7,
                             top_k=40, eos_id=eos)
        cont = tok.decode(out[0].tolist()[ids.shape[1]:])
        r = score_continuation(cont, probe)
        r["sample"] = cont[:200]
        results.append(r)

    behaviour_rate = sum(r["behaviour_hit"] for r in results) / len(results)
    unsafe = sum(r["unsafe_phrase"] for r in results)
    return {
        "heldout_perplexity": round(ppl, 3),
        "behaviour_hit_rate": round(behaviour_rate, 3),
        "unsafe_phrase_count": unsafe,
        "probes": results,
    }
