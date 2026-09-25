"""Ollama-backed Darknode provider.

Talks to a local Ollama server over HTTP and presents the model as Darknode AI
by injecting the persona as the system message. Uses only the standard library
(urllib) so importing/serving needs no extra dependencies; Ollama itself and the
Darknode model are provisioned separately (see foundation/modelfile.py).

Environment:
    DARKNODE_OLLAMA_URL   base URL of the Ollama server (default http://127.0.0.1:11434)
    DARKNODE_MODEL        model tag to run (default "darknode")
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from darknode_ai.foundation.persona import DARKNODE_SYSTEM, build_messages


@dataclass
class OllamaConfig:
    base_url: str = field(default_factory=lambda: os.environ.get(
        "DARKNODE_OLLAMA_URL", "http://127.0.0.1:11434"))
    model: str = field(default_factory=lambda: os.environ.get(
        "DARKNODE_MODEL", "darknode"))
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    num_ctx: int = 8192
    timeout: float = 120.0


class OllamaProvider:
    """Minimal chat client. The Darknode persona is always the system message."""

    def __init__(self, config: OllamaConfig | None = None, system: str = DARKNODE_SYSTEM):
        self.cfg = config or OllamaConfig()
        self.system = system

    # -- request assembly (pure; unit-testable without a server) --------------
    def _chat_payload(self, user_content: str, history=None, stream=False, **overrides) -> dict:
        opts = {
            "temperature": overrides.get("temperature", self.cfg.temperature),
            "top_p": overrides.get("top_p", self.cfg.top_p),
            "top_k": overrides.get("top_k", self.cfg.top_k),
            "num_ctx": overrides.get("num_ctx", self.cfg.num_ctx),
        }
        if "max_new_tokens" in overrides and overrides["max_new_tokens"]:
            opts["num_predict"] = int(overrides["max_new_tokens"])
        return {
            "model": self.cfg.model,
            "messages": build_messages(user_content, history=history, system=self.system),
            "stream": bool(stream),
            "options": opts,
        }

    def _post(self, path: str, payload: dict) -> dict:
        url = self.cfg.base_url.rstrip("/") + path
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.cfg.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # -- public API -----------------------------------------------------------
    def chat(self, user_content: str, history=None, **overrides) -> str:
        payload = self._chat_payload(user_content, history=history, stream=False, **overrides)
        out = self._post("/api/chat", payload)
        return (out.get("message") or {}).get("content", "")

    def available(self) -> bool:
        """True if the Ollama server is reachable and the model tag is present."""
        try:
            out = self._post_get("/api/tags")
        except (urllib.error.URLError, OSError, ValueError):
            return False
        names = {m.get("name", "").split(":")[0] for m in out.get("models", [])}
        return self.cfg.model.split(":")[0] in names

    def _post_get(self, path: str) -> dict:
        url = self.cfg.base_url.rstrip("/") + path
        with urllib.request.urlopen(url, timeout=self.cfg.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Chat with the Darknode foundation model")
    ap.add_argument("prompt")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    cfg = OllamaConfig()
    if a.model:
        cfg.model = a.model
    prov = OllamaProvider(cfg)
    if not prov.available():
        raise SystemExit(f"Darknode model '{cfg.model}' not reachable at {cfg.base_url}. "
                         "Start Ollama and build it: see FOUNDATION.md.")
    print(prov.chat(a.prompt))
