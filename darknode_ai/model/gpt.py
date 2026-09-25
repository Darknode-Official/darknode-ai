"""Darknode AI — decoder-only transformer, implemented from scratch.

A compact GPT-style language model written directly in PyTorch. No pretrained
weights, no imported model architectures. Modern-but-minimal choices that train
stably on modest (Colab) hardware:

  - token embeddings + weight-tied LM head
  - pre-norm blocks with RMSNorm
  - rotary position embeddings (RoPE)
  - multi-head causal self-attention (uses scaled_dot_product_attention when
    available for a memory-efficient kernel; falls back to an explicit path)
  - SwiGLU feed-forward

Sized by DarknodeGPTConfig. The `tiny` preset runs on CPU for tests; `small`
is tuned for a free Colab T4.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DarknodeGPTConfig:
    vocab_size: int = 8192
    context_len: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False

    @staticmethod
    def tiny(vocab_size: int = 512) -> "DarknodeGPTConfig":
        return DarknodeGPTConfig(vocab_size=vocab_size, context_len=64,
                                 n_layer=2, n_head=2, n_embd=64)

    @staticmethod
    def small(vocab_size: int = 8192) -> "DarknodeGPTConfig":
        return DarknodeGPTConfig(vocab_size=vocab_size, context_len=512,
                                 n_layer=8, n_head=8, n_embd=512, dropout=0.1)

    @staticmethod
    def medium(vocab_size: int = 16384) -> "DarknodeGPTConfig":
        # ~85M params; a real GPU (or patient CPU) target
        return DarknodeGPTConfig(vocab_size=vocab_size, context_len=512,
                                 n_layer=12, n_head=12, n_embd=768, dropout=0.1)

    @staticmethod
    def large(vocab_size: int = 32768) -> "DarknodeGPTConfig":
        # ~300M params; needs a proper GPU
        return DarknodeGPTConfig(vocab_size=vocab_size, context_len=1024,
                                 n_layer=24, n_head=16, n_embd=1024, dropout=0.1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def build_rope_cache(seq_len: int, head_dim: int, device, base: float = 10000.0):
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)              # (T, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D)
    B, H, T, D = x.shape
    x = x.view(B, H, T, D // 2, 2)
    x1, x2 = x[..., 0], x[..., 1]
    cos = cos[:T].view(1, 1, T, D // 2)
    sin = sin[:T].view(1, 1, T, D // 2)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack((out1, out2), dim=-1).view(B, H, T, D)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: DarknodeGPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout
        self.attn_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float("-inf"))
            att = self.attn_drop(F.softmax(att, dim=-1))
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: DarknodeGPTConfig) -> None:
        super().__init__()
        hidden = int(8 / 3 * cfg.n_embd)
        hidden = 32 * ((hidden + 31) // 32)  # round for kernel friendliness
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w2 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.silu(self.w1(x)) * self.w2(x)))


class Block(nn.Module):
    def __init__(self, cfg: DarknodeGPTConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class DarknodeGPT(nn.Module):
    def __init__(self, cfg: DarknodeGPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight  # weight tying
        self._rope = None
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        # exclude tied head (shares tok_emb) from the count
        return sum(p.numel() for p in self.parameters()) - self.lm_head.weight.numel()

    def _rope_cache(self, T, device):
        if self._rope is None or self._rope[0].size(0) < T or self._rope[0].device != device:
            self._rope = build_rope_cache(max(T, self.cfg.context_len),
                                          self.cfg.n_embd // self.cfg.n_head, device)
        cos, sin = self._rope
        return cos[:T], sin[:T]

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.context_len, f"seq {T} > context {self.cfg.context_len}"
        cos, sin = self._rope_cache(T, idx.device)
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None,
                 top_p=None, eos_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                if top_p is not None:
                    sp, si = torch.sort(probs, descending=True, dim=-1)
                    cum = torch.cumsum(sp, dim=-1)
                    mask = cum - sp > top_p
                    sp[mask] = 0.0
                    sp = sp / sp.sum(dim=-1, keepdim=True)
                    choice = torch.multinomial(sp, 1)
                    next_id = si.gather(-1, choice)
                else:
                    next_id = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_id), dim=1)
            if eos_id is not None and int(next_id) == eos_id:
                break
        return idx

    def config_dict(self) -> dict:
        return asdict(self.cfg)
