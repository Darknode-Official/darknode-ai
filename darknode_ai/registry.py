"""Model registry — versioned record of every trained Darknode AI checkpoint.

Tracks what the spec asks for: model version, dataset version, config, metrics,
checkpoint path, and a written limitations note. Backed by a single JSON file so
it is diffable and needs no service. This is how a model gets promoted from a
training run to a deployable version, and how a bad version is rolled back.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class ModelVersion:
    version: str
    created: float = field(default_factory=time.time)
    dataset_version: str = ""
    tokenizer_vocab: int = 0
    params_m: float = 0.0
    model_config: dict = field(default_factory=dict)
    train_config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    checkpoint: str = ""
    tokenizer: str = ""
    limitations: str = ""
    state: str = "trained"   # trained | evaluated | deployed | rolled_back


class Registry:
    def __init__(self, path: str | Path = "runs/registry.json"):
        self.path = Path(path)
        self.versions: list[ModelVersion] = []
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.versions = [ModelVersion(**v) for v in data.get("versions", [])]

    def add(self, mv: ModelVersion) -> None:
        self.versions = [v for v in self.versions if v.version != mv.version]
        self.versions.append(mv)
        self._flush()

    def get(self, version: str) -> ModelVersion | None:
        return next((v for v in self.versions if v.version == version), None)

    def set_state(self, version: str, state: str) -> None:
        mv = self.get(version)
        if not mv:
            raise KeyError(version)
        if state == "deployed":
            for v in self.versions:
                if v.state == "deployed":
                    v.state = "trained"
        mv.state = state
        self._flush()

    def deployed(self) -> ModelVersion | None:
        return next((v for v in self.versions if v.state == "deployed"), None)

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"versions": [asdict(v) for v in self.versions]}, indent=2),
            encoding="utf-8")
