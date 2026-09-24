from darknode_ai.data.redact import redact


def test_redacts_common_secrets():
    text = (
        "aws key AKIAIOSFODNN7EXAMPLE here\n"
        "token=ghp_1234567890abcdefghijklmnopqrstuvwxyz\n"
        "password: hunter2secret\n"
        "contact admin@example.com about it\n"
        "db postgres://user:sekritpw@10.0.0.5:5432/prod\n"
    )
    out, rep = redact(text)
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "ghp_1234567890" not in out
    assert "hunter2secret" not in out
    assert "sekritpw" not in out
    assert "<|redacted:" in out
    assert rep.total >= 5


def test_density_metric():
    out, rep = redact("password: abcdef " * 5)
    assert rep.total == 5
    assert rep.density > 0


def test_clean_text_untouched():
    clean = "The analyst correlated firewall and endpoint logs by timezone."
    out, rep = redact(clean)
    assert out == clean
    assert rep.total == 0
