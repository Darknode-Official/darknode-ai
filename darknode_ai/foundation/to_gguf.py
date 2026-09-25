"""Convert a trained Darknode LoRA adapter into a GGUF that Ollama can load.

The light, recommended path: keep the base as an Ollama tag and attach the LoRA
as a GGUF adapter -- no 26GB fp16 merge, so it runs on a modest box (and on the
operator's own device where the base is already pulled).

    python -m darknode_ai.foundation.to_gguf \
        --adapter runs/darknode-13b \
        --out runs/darknode-13b/darknode-lora.gguf \
        --emit-modelfile --ollama-base jimscard/whiterabbit-neo

That runs llama.cpp's convert_lora_to_gguf.py (cloning llama.cpp if needed),
writes darknode-lora.gguf, and -- with --emit-modelfile -- a Modelfile of the
form:  FROM <ollama base> / ADAPTER ./darknode-lora.gguf / SYSTEM <persona>.
Then, on any box with Ollama:  ollama create darknode -f Modelfile.

Heavy alternative (self-contained single GGUF with the LoRA baked in) is the
merge path documented in FOUNDATION.md: peft merge_and_unload -> convert_hf_to_gguf
-> llama-quantize. This module focuses on the adapter path.

Needs: git, and the training env's python deps (torch, transformers, safetensors,
gguf). Nothing here needs a GPU.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp"


def _base_model_id(adapter_dir: Path, override: str | None) -> str:
    """The base the adapter was trained on -- from adapter_config.json unless
    overridden. convert_lora_to_gguf needs it to resolve tensor dimensions."""
    if override:
        return override
    cfg = adapter_dir / "adapter_config.json"
    if cfg.is_file():
        data = json.loads(cfg.read_text(encoding="utf-8"))
        base = data.get("base_model_name_or_path")
        if base:
            return base
    raise SystemExit(
        f"Could not determine the base model from {cfg}; pass --base-model-id "
        "(e.g. WhiteRabbitNeo/WhiteRabbitNeo-13B-v1).")


def build_convert_cmd(converter: str, adapter_dir: str, out_gguf: str,
                      base_model_id: str, outtype: str = "f16") -> list[str]:
    """The llama.cpp convert_lora_to_gguf.py invocation, as a pure list so it can
    be unit-tested without llama.cpp present."""
    return [
        "python", converter,
        "--outfile", out_gguf,
        "--outtype", outtype,
        "--base-model-id", base_model_id,
        adapter_dir,
    ]


def ensure_llama_cpp(path: Path) -> Path:
    """Return the path to convert_lora_to_gguf.py, cloning llama.cpp if absent."""
    if not path.exists():
        print(f"cloning llama.cpp -> {path}")
        subprocess.run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(path)],
                       check=True)
    converter = path / "convert_lora_to_gguf.py"
    if not converter.is_file():
        raise SystemExit(
            f"{converter} not found -- update llama.cpp (git -C {path} pull) or "
            "point --llama-cpp at a checkout that has convert_lora_to_gguf.py.")
    return converter


def convert(adapter_dir: str | Path, out_gguf: str | Path,
            llama_cpp: str | Path = "llama.cpp",
            base_model_id: str | None = None, outtype: str = "f16") -> Path:
    adapter_dir = Path(adapter_dir)
    if not (adapter_dir / "adapter_model.safetensors").is_file() and \
       not (adapter_dir / "adapter_model.bin").is_file():
        raise SystemExit(
            f"No adapter weights in {adapter_dir} (expected adapter_model.safetensors). "
            "Point --adapter at the finetune --out directory.")
    base = _base_model_id(adapter_dir, base_model_id)
    converter = ensure_llama_cpp(Path(llama_cpp))
    out_gguf = Path(out_gguf)
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_convert_cmd(str(converter), str(adapter_dir), str(out_gguf),
                            base, outtype)
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not out_gguf.is_file():
        raise SystemExit(f"conversion reported success but {out_gguf} is missing")
    print(f"GGUF LoRA adapter -> {out_gguf}")
    return out_gguf


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Convert a Darknode LoRA adapter to a GGUF for Ollama")
    ap.add_argument("--adapter", required=True,
                    help="finetune --out dir (has adapter_model.safetensors)")
    ap.add_argument("--out", default=None,
                    help="output .gguf path (default: <adapter>/darknode-lora.gguf)")
    ap.add_argument("--llama-cpp", default="llama.cpp",
                    help="llama.cpp checkout dir (cloned here if absent)")
    ap.add_argument("--base-model-id", default=None,
                    help="override the base model id (else read from adapter_config.json)")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16", "q8_0"])
    ap.add_argument("--emit-modelfile", action="store_true",
                    help="also write an Ollama Modelfile (FROM base + ADAPTER gguf)")
    ap.add_argument("--ollama-base", default="jimscard/whiterabbit-neo",
                    help="Ollama base tag to put in FROM (with --emit-modelfile)")
    args = ap.parse_args(argv)

    adapter = Path(args.adapter)
    out_gguf = Path(args.out) if args.out else adapter / "darknode-lora.gguf"
    out_gguf = convert(adapter, out_gguf, llama_cpp=args.llama_cpp,
                       base_model_id=args.base_model_id, outtype=args.outtype)

    if args.emit_modelfile:
        from darknode_ai.foundation.modelfile import render_modelfile
        mf = adapter / "Modelfile"
        text = render_modelfile(args.ollama_base, adapter=f"./{out_gguf.name}")
        mf.write_text(text, encoding="utf-8")
        print(f"Modelfile -> {mf}")
    print("\nNext, on any box with Ollama (and the base pulled):")
    print(f"  ollama create darknode -f {adapter / 'Modelfile'}")
    print("  ollama run darknode")


if __name__ == "__main__":
    main()
