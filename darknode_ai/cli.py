"""darknode-ai CLI — orchestrates the full from-scratch pipeline.

    darknode-ai corpus       build the seed security corpus + manifest
    darknode-ai tokenizer    train the BPE tokenizer on the corpus
    darknode-ai prepare      redact + dedup + pack the dataset (.bin)
    darknode-ai train        train the model from scratch
    darknode-ai eval         evaluate a checkpoint (perplexity + probes)
    darknode-ai sample       generate text from a checkpoint
    darknode-ai register     record a trained version in the registry
    darknode-ai pipeline     corpus -> tokenizer -> prepare -> train -> eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _train_tokenizer(corpus_dir: str, out: str, vocab_size: int, verbose=True):
    from darknode_ai.tokenizer.bpe import BPETokenizer
    text = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                     for p in Path(corpus_dir).glob("*.txt"))
    tok = BPETokenizer()
    tok.train(text, vocab_size=vocab_size, verbose=verbose)
    tok.save(out)
    print(f"tokenizer: {tok.vocab_size} tokens -> {out}")
    return tok


def main(argv=None):
    ap = argparse.ArgumentParser(prog="darknode-ai")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("corpus"); p.add_argument("--out", default="data/corpus")
    p.add_argument("--synthetic", type=int, default=1200)

    p = sub.add_parser("tokenizer")
    p.add_argument("--corpus", default="data/corpus")
    p.add_argument("--out", default="runs/tokenizer.json")
    p.add_argument("--vocab-size", type=int, default=8192)

    p = sub.add_parser("manifest")
    p.add_argument("--corpus", default="data/corpus")
    p.add_argument("--out", default="data/manifest.json")

    p = sub.add_parser("prepare")
    p.add_argument("--manifest", default="data/manifest.json")
    p.add_argument("--tokenizer", default="runs/tokenizer.json")
    p.add_argument("--out", default="data/prepared")

    p = sub.add_parser("train")
    p.add_argument("--preset", choices=["tiny", "small", "medium", "large", "colab_t4"], default="small")
    p.add_argument("--data-dir", default="data/prepared")
    p.add_argument("--out-dir", default="runs/darknode-small")
    p.add_argument("--tokenizer", default="runs/tokenizer.json")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--resume", action="store_true")

    p = sub.add_parser("eval")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", default="runs/tokenizer.json")

    p = sub.add_parser("sample")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", default="runs/tokenizer.json")
    p.add_argument("--prompt", default="<|user|> Triage a suspicious login.\n<|assistant|>\n")

    p = sub.add_parser("register")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", default="runs/tokenizer.json")
    p.add_argument("--version", required=True)
    p.add_argument("--data-dir", default="data/prepared")
    p.add_argument("--limitations", default="Small from-scratch model; domain text generation only.")

    p = sub.add_parser("pipeline")
    p.add_argument("--preset", choices=["tiny", "small", "colab_t4"], default="tiny")
    p.add_argument("--vocab-size", type=int, default=2048)

    args = ap.parse_args(argv)

    if args.cmd == "corpus":
        from darknode_ai.data.corpus import build_corpus
        print(build_corpus(args.out, n_synthetic=args.synthetic))

    elif args.cmd == "tokenizer":
        _train_tokenizer(args.corpus, args.out, args.vocab_size)

    elif args.cmd == "manifest":
        from darknode_ai.data.corpus import build_manifest
        print(build_manifest(args.corpus, args.out))

    elif args.cmd == "prepare":
        from darknode_ai.tokenizer.bpe import BPETokenizer
        from darknode_ai.data.pipeline import build_dataset
        tok = BPETokenizer.load(args.tokenizer)
        stats = build_dataset(args.manifest, tok, args.out)
        print(f"prepared: train={stats.train_tokens} val={stats.val_tokens} "
              f"kept={stats.docs_kept} deduped={stats.docs_deduped} "
              f"version={stats.version} redacted={stats.redaction.total}")

    elif args.cmd == "train":
        from darknode_ai.tokenizer.bpe import BPETokenizer
        from darknode_ai.model.gpt import DarknodeGPTConfig
        from darknode_ai.train.config import TrainConfig
        from darknode_ai.train.trainer import train as run_train
        tok = BPETokenizer.load(args.tokenizer)
        mcfg = {"tiny": DarknodeGPTConfig.tiny,
                "small": DarknodeGPTConfig.small,
                "medium": DarknodeGPTConfig.medium,
                "large": DarknodeGPTConfig.large,
                "colab_t4": DarknodeGPTConfig.small}[args.preset](tok.vocab_size)
        tcfg = {"tiny": TrainConfig.tiny, "small": TrainConfig,
                "medium": TrainConfig.colab_t4, "large": TrainConfig.colab_t4,
                "colab_t4": TrainConfig.colab_t4}[args.preset]()
        tcfg.data_dir, tcfg.out_dir = args.data_dir, args.out_dir
        if args.max_steps:
            tcfg.max_steps = args.max_steps
        print(run_train(mcfg, tcfg, resume=args.resume))

    elif args.cmd == "eval":
        from darknode_ai.eval.evaluate import evaluate
        print(json.dumps(evaluate(args.ckpt, args.tokenizer), indent=2))

    elif args.cmd == "sample":
        from darknode_ai.sample import DarknodeInference
        inf = DarknodeInference(args.ckpt, args.tokenizer)
        print(inf.generate(args.prompt))

    elif args.cmd == "register":
        from darknode_ai.eval.evaluate import evaluate, load_checkpoint
        from darknode_ai.registry import Registry, ModelVersion
        _, mcfg = load_checkpoint(args.ckpt)
        meta_path = Path(args.data_dir) / "meta.json"
        dv = json.loads(meta_path.read_text())["dataset_version"] if meta_path.exists() else ""
        metrics = evaluate(args.ckpt, args.tokenizer)
        reg = Registry()
        reg.add(ModelVersion(
            version=args.version, dataset_version=dv,
            tokenizer_vocab=mcfg.vocab_size,
            model_config=mcfg.__dict__, metrics=metrics,
            checkpoint=args.ckpt, tokenizer=args.tokenizer,
            limitations=args.limitations, state="evaluated"))
        print(f"registered {args.version}: {metrics['heldout_perplexity']} ppl")

    elif args.cmd == "pipeline":
        # end-to-end smoke pipeline (small/tiny), for CI and local verification
        from darknode_ai.data.corpus import build_corpus
        from darknode_ai.tokenizer.bpe import BPETokenizer
        from darknode_ai.data.pipeline import build_dataset
        build_corpus("data/corpus", n_synthetic=200)
        tok = _train_tokenizer("data/corpus", "runs/tokenizer.json",
                               args.vocab_size, verbose=False)
        stats = build_dataset("data/manifest.json", tok, "data/prepared")
        print(f"data ready: {stats.train_tokens} train tokens, version {stats.version}")
        from darknode_ai.model.gpt import DarknodeGPTConfig
        from darknode_ai.train.config import TrainConfig
        from darknode_ai.train.trainer import train as run_train
        mcfg = DarknodeGPTConfig.tiny(tok.vocab_size)
        tcfg = TrainConfig.tiny()
        print(run_train(mcfg, tcfg))


if __name__ == "__main__":
    main()
