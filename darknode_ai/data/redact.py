"""Secret & PII redaction — runs on every document BEFORE tokenization.

The spec is explicit: credentials, tokens, keys and private data must never
enter model weights. This module is the enforced gate. It is deliberately
conservative (over-redacts rather than under-redacts) and returns a report so
the data pipeline can refuse a corpus whose secret density is suspiciously high
(a sign of an unsafe source).

Redaction replaces matches with typed placeholders so the model still learns
the *shape* of security text ("an API key appears here") without memorising the
secret itself.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("AWS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("AWS_SECRET", re.compile(r"(?i)aws_secret_access_key\s*[=:]\s*[A-Za-z0-9/+=]{40}")),
    ("GITHUB_TOKEN", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b")),
    ("SLACK_TOKEN", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("GOOGLE_API_KEY", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("PRIVATE_KEY", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    ("BEARER", re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}")),
    ("PASSWORD_KV", re.compile(r"(?i)(password|passwd|pwd|secret|api[_-]?key|token)\s*[=:]\s*[\"']?[^\s\"',]{6,}[\"']?")),
    ("CONN_STRING", re.compile(r"\b[a-z]+://[^\s:@/]+:[^\s:@/]+@[^\s/]+")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
]


@dataclass
class RedactionReport:
    counts: dict[str, int] = field(default_factory=dict)
    total: int = 0
    chars_in: int = 0

    @property
    def density(self) -> float:
        return self.total / max(1, self.chars_in) * 1000  # secrets per 1k chars

    def merge(self, other: "RedactionReport") -> None:
        for k, v in other.counts.items():
            self.counts[k] = self.counts.get(k, 0) + v
        self.total += other.total
        self.chars_in += other.chars_in


def redact(text: str) -> tuple[str, RedactionReport]:
    report = RedactionReport(chars_in=len(text))
    for label, pat in _PATTERNS:
        def _sub(_m, _label=label):
            report.counts[_label] = report.counts.get(_label, 0) + 1
            report.total += 1
            return f"<|redacted:{_label}|>"
        text = pat.sub(_sub, text)
    return text, report
