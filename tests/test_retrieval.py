"""Retrieval (RAG) tests. Pure stdlib, no torch/network."""
import json

from darknode_ai.retrieval.store import KnowledgeStore, tokenize, chunk_text
from darknode_ai.retrieval.rag import (format_evidence, build_grounded_prompt,
                                       DarknodeRAG)


def test_tokenize_keeps_dotted_terms():
    toks = tokenize("SMB on port 445 and CVE-2021-44228 log4j")
    assert "445" in toks and "cve-2021-44228" in toks and "log4j" in toks


def test_chunking_splits_on_paragraphs():
    text = "a" * 600 + "\n\n" + "b" * 600 + "\n\n" + "c" * 600
    chunks = chunk_text(text, target=1000)
    assert len(chunks) >= 2


def test_bm25_ranks_relevant_chunk_first():
    s = KnowledgeStore()
    s.add("Kerberoasting abuses SPN accounts to crack service tickets offline.",
          "ad.txt", "authored")
    s.add("Nginx reverse proxy configuration and TLS termination basics.",
          "web.txt", "authored")
    s.add("Phishing awareness and email header analysis for defenders.",
          "email.txt", "authored")
    hits = s.build().search("how to detect kerberoasting service tickets", k=2)
    assert hits and hits[0].doc.source == "ad.txt"
    assert hits[0].score > 0


def test_search_empty_store():
    assert KnowledgeStore().search("anything") == []


def test_save_load_roundtrip(tmp_path):
    s = KnowledgeStore()
    s.ingest_text("Buffer overflow on the stack overwrites the return address.",
                  "pwn.txt", "authored")
    s.build().save(tmp_path / "k.json")
    loaded = KnowledgeStore.load(tmp_path / "k.json")
    assert len(loaded.docs) == len(s.docs)
    assert loaded.search("return address overwrite")[0].doc.source == "pwn.txt"


def test_ingest_jsonl_cve(tmp_path):
    p = tmp_path / "cve.jsonl"
    p.write_text(json.dumps({"id": "CVE-2021-44228",
                             "description": "Log4Shell JNDI remote code execution."}) + "\n")
    s = KnowledgeStore()
    n = s.ingest_jsonl(p, text_field="description", source="nvd",
                       provenance="NVD", id_field="id")
    assert n == 1
    hit = s.build().search("log4shell rce")[0]
    assert "CVE-2021-44228" in hit.doc.source


def test_format_evidence_tags_retrieved():
    s = KnowledgeStore()
    s.add("SPN kerberoasting detail.", "ad.txt", "authored")
    hits = s.build().search("kerberoasting")
    block = format_evidence(hits)
    assert block.startswith("RETRIEVED [ad.txt")


def test_format_evidence_empty_is_unknown():
    assert "UNKNOWN" in format_evidence([])


def test_build_grounded_prompt_includes_question_and_evidence():
    s = KnowledgeStore()
    s.add("SMB signing prevents relay.", "smb.txt", "authored")
    hits = s.build().search("smb relay")
    prompt = build_grounded_prompt("How do I stop SMB relay?", hits)
    assert "Operator question: How do I stop SMB relay?" in prompt
    assert "RETRIEVED" in prompt


def test_darknode_rag_answer_grounds_and_reports_evidence():
    s = KnowledgeStore()
    s.add("To detect kerberoasting, alert on TGS-REQ with RC4 encryption.",
          "ad.txt", "authored")
    s.build()

    class FakeProvider:
        def __init__(self):
            self.seen = None

        def chat(self, content, **kw):
            self.seen = content
            return "RETRIEVED [ad.txt]: alert on RC4 TGS-REQ."

    prov = FakeProvider()
    rag = DarknodeRAG(s, prov, k=2)
    out = rag.answer("detect kerberoasting")
    assert "RETRIEVED" in prov.seen  # evidence was injected into the prompt
    assert out["evidence"] and out["evidence"][0]["source"] == "ad.txt"
    assert "RC4" in out["completion"]
