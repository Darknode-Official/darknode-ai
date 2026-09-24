"""Darknode AI serving layer — a small FastAPI app that exposes the trained
from-scratch model over HTTP, plus a self-contained security-workspace panel.

Endpoints
    GET  /                -> the Darknode AI panel (static HTML)
    GET  /api/health      -> model + registry status
    GET  /api/info        -> deployed version, config, metrics
    POST /api/generate    -> { prompt, max_new_tokens, temperature, top_k, top_p }

Model selection order:
    1. explicit env DARKNODE_CKPT + DARKNODE_TOKENIZER
    2. the version marked `deployed` in the registry
    3. runs/darknode-*/best.pt with runs/tokenizer.json (dev fallback)

The server never executes actions. It only generates text. Prompts from the
panel are treated as untrusted and encoded without control tokens.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from darknode_ai.registry import Registry

_STATIC = Path(__file__).parent / "static"


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    max_new_tokens: int = Field(160, ge=1, le=1024)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int | None = Field(40, ge=1, le=1000)
    top_p: float | None = Field(0.95, ge=0.0, le=1.0)


def _resolve_model_paths() -> tuple[str, str, str]:
    ckpt = os.environ.get("DARKNODE_CKPT")
    tok = os.environ.get("DARKNODE_TOKENIZER")
    if ckpt and tok:
        return ckpt, tok, "env"
    reg = Registry(os.environ.get("DARKNODE_REGISTRY", "runs/registry.json"))
    dep = reg.deployed()
    if dep and Path(dep.checkpoint).exists():
        return dep.checkpoint, dep.tokenizer, dep.version
    for best in sorted(Path("runs").glob("*/best.pt")) if Path("runs").exists() else []:
        tok_path = "runs/tokenizer.json"
        if Path(tok_path).exists():
            return str(best), tok_path, "dev-fallback"
    raise RuntimeError("No model available. Train one or set DARKNODE_CKPT/TOKENIZER.")


def create_app() -> FastAPI:
    app = FastAPI(title="Darknode AI", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])
    state: dict = {"inference": None, "version": None, "error": None}

    def _ensure_model():
        if state["inference"] is not None:
            return state["inference"]
        from darknode_ai.sample import DarknodeInference  # torch import deferred
        ckpt, tok, version = _resolve_model_paths()
        state["inference"] = DarknodeInference(ckpt, tok, device="auto")
        state["version"] = version
        return state["inference"]

    @app.get("/api/health")
    def health():
        try:
            _ensure_model()
            return {"status": "ok", "model_loaded": True, "version": state["version"]}
        except Exception as e:  # model not trained yet is a normal state
            state["error"] = str(e)
            return {"status": "degraded", "model_loaded": False, "detail": str(e)}

    @app.get("/api/info")
    def info():
        reg = Registry(os.environ.get("DARKNODE_REGISTRY", "runs/registry.json"))
        return {
            "loaded_version": state["version"],
            "deployed": (reg.deployed().__dict__ if reg.deployed() else None),
            "versions": [v.version for v in reg.versions],
        }

    @app.post("/api/generate")
    def generate(req: GenerateRequest):
        try:
            inf = _ensure_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))
        text = inf.generate(req.prompt, max_new_tokens=req.max_new_tokens,
                            temperature=req.temperature, top_k=req.top_k,
                            top_p=req.top_p, allowed_special=False)
        return {"completion": text, "version": state["version"],
                "note": "Generated text. Draft only; verify against evidence. No action executed."}

    @app.get("/")
    def index():
        idx = _STATIC / "index.html"
        if idx.exists():
            return FileResponse(idx)
        raise HTTPException(status_code=404, detail="panel not built")

    return app


app = create_app()


def main():
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8799")))


if __name__ == "__main__":
    main()
