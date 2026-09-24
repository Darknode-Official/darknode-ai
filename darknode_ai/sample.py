"""Inference — load a checkpoint + tokenizer and generate text."""
from __future__ import annotations

import torch

from darknode_ai.eval.evaluate import load_checkpoint
from darknode_ai.tokenizer.bpe import BPETokenizer


class DarknodeInference:
    def __init__(self, ckpt_path: str, tokenizer_path: str, device: str = "auto"):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") \
            if device == "auto" else device
        self.model, self.cfg = load_checkpoint(ckpt_path, self.device)
        self.tok = BPETokenizer.load(tokenizer_path)
        self.eos = self.tok.special.get("<|endoftext|>")

    def generate(self, prompt: str, max_new_tokens: int = 160, temperature: float = 0.8,
                 top_k: int | None = 40, top_p: float | None = 0.95,
                 allowed_special: bool = True) -> str:
        # untrusted content should be passed with allowed_special=False so it
        # cannot inject control tokens into the stream (injection boundary).
        ids = self.tok.encode(prompt, allowed_special=allowed_special)
        x = torch.tensor([ids], device=self.device)
        out = self.model.generate(x, max_new_tokens=max_new_tokens,
                                  temperature=temperature, top_k=top_k,
                                  top_p=top_p, eos_id=self.eos)
        return self.tok.decode(out[0].tolist()[len(ids):])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default="<|user|> Triage a suspicious login.\n<|assistant|>\n")
    ap.add_argument("--tokens", type=int, default=160)
    a = ap.parse_args()
    inf = DarknodeInference(a.ckpt, a.tokenizer)
    print(inf.generate(a.prompt, max_new_tokens=a.tokens))
