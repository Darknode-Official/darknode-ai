"""Training configuration presets."""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class TrainConfig:
    data_dir: str = "data/prepared"
    out_dir: str = "runs/darknode-small"
    # optimisation
    batch_size: int = 32
    grad_accum: int = 4
    max_steps: int = 5000
    warmup_steps: int = 200
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # schedule / io
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 20
    ckpt_interval: int = 500
    seed: int = 1337
    device: str = "auto"      # auto|cpu|cuda
    dtype: str = "auto"       # auto|float32|bfloat16|float16
    compile: bool = False

    def dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def tiny() -> "TrainConfig":
        return TrainConfig(batch_size=8, grad_accum=1, max_steps=20,
                           warmup_steps=2, eval_interval=10, eval_iters=5,
                           log_interval=5, ckpt_interval=20, device="cpu",
                           dtype="float32")

    @staticmethod
    def colab_t4() -> "TrainConfig":
        return TrainConfig(batch_size=24, grad_accum=8, max_steps=6000,
                           warmup_steps=300, eval_interval=300,
                           dtype="float16", compile=False)
