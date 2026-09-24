import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from darknode_ai.serve.app import create_app


def _train_tiny(tmp_path):
    """Train a real tiny checkpoint so the server has something to load."""
    import numpy as np
    from darknode_ai.data.corpus import build_corpus
    from darknode_ai.tokenizer.bpe import BPETokenizer
    from darknode_ai.data.pipeline import build_dataset
    from darknode_ai.model.gpt import DarknodeGPTConfig
    from darknode_ai.train.config import TrainConfig
    from darknode_ai.train.trainer import train

    build_corpus(tmp_path / "data" / "corpus", n_synthetic=60)
    text = "\n".join((tmp_path / "data" / "corpus" / f).read_text()
                     for f in ["knowledge.txt", "synthetic_cases.txt"])
    tok = BPETokenizer(); tok.train(text, vocab_size=512)
    tok_path = tmp_path / "tokenizer.json"; tok.save(tok_path)
    build_dataset(tmp_path / "data" / "manifest.json", tok, tmp_path / "prepared",
                  val_fraction=0.1)
    mcfg = DarknodeGPTConfig.tiny(tok.vocab_size)
    tcfg = TrainConfig.tiny(); tcfg.max_steps = 10
    tcfg.data_dir = str(tmp_path / "prepared")
    tcfg.out_dir = str(tmp_path / "run")
    train(mcfg, tcfg)
    return str(tmp_path / "run" / "best.pt"), str(tok_path)


def test_server_health_and_generate(tmp_path, monkeypatch):
    ckpt, tok = _train_tiny(tmp_path)
    monkeypatch.setenv("DARKNODE_CKPT", ckpt)
    monkeypatch.setenv("DARKNODE_TOKENIZER", tok)
    client = TestClient(create_app())

    h = client.get("/api/health").json()
    assert h["model_loaded"] is True

    r = client.post("/api/generate", json={"prompt": "<|user|> triage\n<|assistant|>\n",
                                           "max_new_tokens": 16, "temperature": 0.7})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["completion"], str)
    assert "no action" in body["note"].lower()

    # panel is served
    assert client.get("/").status_code == 200


def test_generate_validation():
    app = create_app()
    client = TestClient(app)
    # empty prompt rejected by schema
    assert client.post("/api/generate", json={"prompt": ""}).status_code == 422
