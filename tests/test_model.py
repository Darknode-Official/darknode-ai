import pytest

torch = pytest.importorskip("torch")

from darknode_ai.model.gpt import DarknodeGPT, DarknodeGPTConfig


def test_forward_shapes_and_loss():
    cfg = DarknodeGPTConfig.tiny(vocab_size=128)
    model = DarknodeGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss.item() > 0
    assert model.num_params() > 0


def test_weight_tying():
    cfg = DarknodeGPTConfig.tiny(vocab_size=64)
    model = DarknodeGPT(cfg)
    assert model.tok_emb.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_generate_runs_and_respects_context():
    cfg = DarknodeGPTConfig.tiny(vocab_size=64)
    model = DarknodeGPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(idx, max_new_tokens=10, temperature=0.8, top_k=5)
    assert out.shape[1] == 14


def test_one_training_step_reduces_loss_on_fixed_batch():
    torch.manual_seed(0)
    cfg = DarknodeGPTConfig.tiny(vocab_size=64)
    model = DarknodeGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randint(0, cfg.vocab_size, (4, 16))
    y = torch.randint(0, cfg.vocab_size, (4, 16))
    _, first = model(x, y)
    for _ in range(30):
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        opt.step()
    assert loss.item() < first.item()  # memorises a fixed batch -> loss drops
