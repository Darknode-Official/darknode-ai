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


def test_build_manifest_scans_corpus_and_authored(tmp_path):
    from darknode_ai.data.corpus import build_manifest
    data = tmp_path / "data"
    (data / "corpus").mkdir(parents=True)
    (data / "authored").mkdir(parents=True)
    (data / "corpus" / "knowledge.txt").write_text("seed knowledge doc")
    (data / "authored" / "agent_linux.txt").write_text("authored linux doc")
    out = build_manifest(data / "corpus", data / "manifest.json")
    assert out["sources"] == 2
    manifest = json.loads((data / "manifest.json").read_text())
    cats = {s["category"] for s in manifest["sources"]}
    assert "linux-security" in cats and "defensive-knowledge" in cats
    provs = {s["provenance"] for s in manifest["sources"]}
    assert "darknode-agent-authored" in provs
