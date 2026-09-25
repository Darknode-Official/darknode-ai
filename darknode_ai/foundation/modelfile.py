"""Generate an Ollama Modelfile that packages Darknode as its own model.

`ollama create darknode -f Modelfile` builds a model that:
  - starts from the security base (or a merged GGUF of base+adapter),
  - carries the Darknode persona as its SYSTEM prompt, and
  - runs under the name `darknode` -- so `ollama run darknode` is Darknode AI
    end to end, with no base-model plumbing shown to the operator.

Two build modes:
  * gguf  -> FROM ./darknode.gguf   (a merged+converted adapter; recommended for
             a self-contained model with the LoRA baked in)
  * pull  -> FROM <ollama base tag> + ADAPTER (when the base is an Ollama tag and
             you attach the adapter separately)

Note: the LoRA adapter from finetune.py is a PEFT adapter. To bake it into a
single GGUF, merge it (peft merge_and_unload) and convert with llama.cpp; see
FOUNDATION.md. This module only emits the Modelfile text.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from darknode_ai.foundation.persona import DARKNODE_SYSTEM, modelfile_system_block

_DEFAULT_PARAMS = {
    "temperature": "0.7",
    "top_p": "0.95",
    "top_k": "40",
    "num_ctx": "8192",
}


def render_modelfile(from_ref: str, adapter: str | None = None,
                     system: str = DARKNODE_SYSTEM,
                     params: dict | None = None) -> str:
    lines = [
        "# Darknode AI -- generated Modelfile",
        "# Base attribution is retained in the repo NOTICE (license compliance).",
        f"FROM {from_ref}",
        "",
    ]
    if adapter:
        lines += [f"ADAPTER {adapter}", ""]
    lines.append(modelfile_system_block(system))
    lines.append("")
    for k, v in (params or _DEFAULT_PARAMS).items():
        lines.append(f"PARAMETER {k} {v}")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Emit a Darknode Ollama Modelfile")
    ap.add_argument("--mode", choices=["gguf", "pull"], default="gguf")
    ap.add_argument("--gguf", default="./darknode.gguf",
                    help="path to merged GGUF (mode=gguf)")
    ap.add_argument("--base", default="jimscard/whiterabbit-neo",
                    help="Ollama base tag (mode=pull)")
    ap.add_argument("--adapter", default=None,
                    help="adapter path/dir to attach (mode=pull)")
    ap.add_argument("--out", default="runs/darknode-foundation/Modelfile")
    args = ap.parse_args(argv)

    from_ref = args.gguf if args.mode == "gguf" else args.base
    text = render_modelfile(from_ref, adapter=args.adapter if args.mode == "pull" else None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Modelfile -> {out}\nBuild it with:  ollama create darknode -f {out}")


if __name__ == "__main__":
    main()
