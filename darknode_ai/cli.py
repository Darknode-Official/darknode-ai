"""darknode-ai CLI — orchestrates the full from-scratch pipeline.

    darknode-ai corpus       build the seed security corpus + manifest
    darknode-ai tokenizer    train the BPE tokenizer on the corpus
    darknode-ai prepare      redact + dedup + pack the dataset (.bin)
    darknode-ai train        train the model from scratch
    darknode-ai eval         evaluate a checkpoint (perplexity + probes)
    darknode-ai sample       generate text from a checkpoint
    darknode-ai register     record a trained version in the registry
    darknode-ai index        build the BM25 knowledge store for RAG
    darknode-ai pipeline     corpus -> tokenizer -> prepare -> train -> eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _train_tokenizer(corpus_dir: str, out: str, vocab_size: int, verbose=True):
    from darknode_ai.tokenizer.bpe import BPETokenizer
    dirs = [Path(corpus_dir), Path(corpus_dir).parent / "authored"]
    text = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                     for d in dirs if d.is_dir() for p in sorted(d.glob("*.txt")))
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
    p.add_argument("--preset", choices=["tiny", "small", "medium", "large", "b1", "colab_t4"], default="small")
    p.add_argument("--i-have-a-big-gpu", action="store_true",
                   help="required to build large/b1 presets (guards against OOM on CPU/small GPU)")
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

    p = sub.add_parser("index")
    p.add_argument("--authored", default="data/authored")
    p.add_argument("--corpus", default="data/corpus")
    p.add_argument("--jsonl", action="append", default=[],
                   metavar="PATH:FIELD:SOURCE", help="add external JSONL, e.g. "
                   "cve.jsonl:description:nvd (repeatable)")
    p.add_argument("--out", default="runs/knowledge.json")

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
        if args.preset in ("large", "b1") and not getattr(args, "i_have_a_big_gpu", False):
            raise SystemExit(
                f"Preset '{args.preset}' allocates a very large model "
                f"({'~0.3B' if args.preset=='large' else '~1B'} params) and will OOM "
                "without a big GPU. Also note: the current corpus is far too small to "
                "train it usefully (it would overfit). Re-run with --i-have-a-big-gpu "
                "only if you have the GPU and a much larger dataset.")
        mcfg = {"tiny": DarknodeGPTConfig.tiny,
                "small": DarknodeGPTConfig.small,
                "medium": DarknodeGPTConfig.medium,
                "large": DarknodeGPTConfig.large,
                "b1": DarknodeGPTConfig.b1,
                "colab_t4": DarknodeGPTConfig.small}[args.preset](tok.vocab_size)
        tcfg = {"tiny": TrainConfig.tiny, "small": TrainConfig,
                "medium": TrainConfig.colab_t4, "large": TrainConfig.colab_t4,
                "b1": TrainConfig.colab_t4,
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

    elif args.cmd == "index":
        from darknode_ai.retrieval.store import KnowledgeStore
        store = KnowledgeStore()
        n = 0
        if Path(args.authored).is_dir():
            n += store.ingest_dir(args.authored, provenance="darknode-authored")
        kdir = Path(args.corpus)
        if kdir.is_dir():
            for kf in sorted(kdir.glob("*.txt")):
                n += store.ingest_text(kf.read_text(encoding="utf-8", errors="replace"),
                                       source=kf.name, provenance="darknode-corpus")
        for spec in args.jsonl:
            parts = spec.split(":")
            if len(parts) < 2:
                raise SystemExit(f"--jsonl expects PATH:FIELD[:SOURCE], got '{spec}'")
            path, field = parts[0], parts[1]
            source = parts[2] if len(parts) > 2 else "external"
            n += store.ingest_jsonl(path, text_field=field, source=source,
                                    provenance=source)
        store.build().save(args.out)
        print(f"indexed {len(store.docs)} chunks ({n} added) -> {args.out}")

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
