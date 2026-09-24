import json
import numpy as np

from darknode_ai.data.corpus import build_corpus
from darknode_ai.data.pipeline import build_dataset, normalise
from darknode_ai.tokenizer.bpe import BPETokenizer


def _tok(text):
    t = BPETokenizer()
    t.train(text, vocab_size=512)
    return t


def test_normalise():
    assert normalise("a\r\n\r\n\r\nb   c\t\td") == "a\n\nb c d"


def test_build_dataset_end_to_end(tmp_path):
    counts = build_corpus(tmp_path / "data" / "corpus", n_synthetic=80)
    assert counts["synthetic_docs"] == 80
    corpus_text = "\n".join(
        (tmp_path / "data" / "corpus" / f).read_text()
        for f in ["knowledge.txt", "synthetic_cases.txt"])
    tok = _tok(corpus_text)

    stats = build_dataset(tmp_path / "data" / "manifest.json", tok,
                          tmp_path / "prepared", val_fraction=0.1)
    assert stats.train_tokens > 0
    assert stats.docs_kept > 0
    assert stats.version  # dataset version hash present

    meta = json.loads((tmp_path / "prepared" / "meta.json").read_text())
    assert meta["vocab_size"] == tok.vocab_size
    arr = np.fromfile(tmp_path / "prepared" / "train.bin", dtype=np.uint16)
    assert arr.max() < tok.vocab_size  # every id in range


def test_dedup(tmp_path):
    d = tmp_path / "c"
    d.mkdir()
    (d / "a.txt").write_text("identical document about SMB port 445 detection logic here.\n<<<DOC>>>\nidentical document about SMB port 445 detection logic here.")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sources": [
        {"path": "c/a.txt", "license": "CC0-1.0", "category": "t",
         "provenance": "test", "allowed_use": "train"}]}))
    tok = _tok("identical document about SMB port 445 detection logic here " * 20)
    stats = build_dataset(manifest, tok, tmp_path / "out", val_fraction=0.0)
    assert stats.docs_deduped == 1
