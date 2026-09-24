from darknode_ai.tokenizer.bpe import BPETokenizer, SPECIAL_TOKENS


CORPUS = ("The SOC analyst reviewed the firewall logs. "
          "10.0.0.1 connected to 10.0.0.2 over port 445 (SMB). "
          "CVE-2021-44228 affects log4j. sha256:abcdef0123456789. "
          "Detection: many failed logons then a success. ") * 40


def test_train_and_roundtrip():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=400, verbose=False)
    # BPE may stop early once no pair repeats (>=2); vocab never exceeds target
    # and always includes the 256 byte base + specials.
    assert 256 + len(SPECIAL_TOKENS) <= tok.vocab_size <= 400
    for s in ["10.0.0.1 over port 445", "CVE-2021-44228", "failed logons then a success"]:
        assert tok.decode(tok.encode(s)) == s


def test_vocab_grows_with_diversity_never_exceeds_target():
    small = BPETokenizer(); small.train(CORPUS, vocab_size=1000)
    diverse = BPETokenizer()
    extra = " ".join(f"host-{i} user{i} 10.0.{i}.{i} CVE-2020-{1000+i}" for i in range(300))
    diverse.train(CORPUS + " " + extra, vocab_size=1000)
    assert diverse.vocab_size > small.vocab_size      # more distinct text -> more merges
    assert diverse.vocab_size <= 1000                 # never past the target


def test_byte_level_lossless_on_unseen():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=400)
    weird = "🔒 base64: YWRtaW46cGFzcw== \t {json:true}\néè"
    assert tok.decode(tok.encode(weird)) == weird


def test_special_tokens_registered_and_optional():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=400)
    for s in SPECIAL_TOKENS:
        assert s in tok.special
    ids_special = tok.encode("<|user|> hi", allowed_special=True)
    assert tok.special["<|user|>"] in ids_special
    # untrusted mode must NOT emit the control-token id
    ids_plain = tok.encode("<|user|> hi", allowed_special=False)
    assert tok.special["<|user|>"] not in ids_plain


def test_save_load(tmp_path):
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=400)
    p = tmp_path / "tok.json"
    tok.save(p)
    tok2 = BPETokenizer.load(p)
    s = "port 445 SMB CVE-2021-44228"
    assert tok2.encode(s) == tok.encode(s)
    assert tok2.vocab_size == tok.vocab_size
