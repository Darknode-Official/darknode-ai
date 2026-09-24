"""Seed corpus builder for Darknode AI.

Everything here is authored for Darknode (license: CC0 / internal) so the
training set is rights-clean by construction — no scraping. Two parts:

  1. KNOWLEDGE  — hand-written defensive-security explainers.
  2. SYNTHETIC  — a generator that emits many evidence-typed security examples
                  in Darknode's house style (OBSERVED / RETRIEVED / INFERRED /
                  HYPOTHESIS / RECOMMENDATION / UNKNOWN). This teaches the model
                  to separate observation from speculation — a core spec goal.

Running this module writes UTF-8 corpus files and a governance manifest.json.
The generator is seeded, so the corpus is reproducible (dataset versioning).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

DOC = "\n<<<DOC>>>\n"

KNOWLEDGE: list[str] = [
    # --- networking ---
    """TCP three-way handshake. A client opens a connection with SYN, the server
replies SYN-ACK, the client confirms with ACK. A half-open connection is one
where the handshake never completed; a flood of SYNs without ACKs is the shape
of a SYN-flood denial-of-service. Defensively, SYN cookies let a server avoid
allocating state until the handshake completes, which blunts the flood.""",
    """Common ports worth memorising for triage: 22 SSH, 23 Telnet (cleartext,
treat any use as a finding), 25/587 SMTP, 53 DNS, 80/443 HTTP/HTTPS, 139/445
SMB, 3389 RDP, 3306 MySQL, 5432 Postgres, 6379 Redis (often exposed with no
auth). An unexpected listener on one of these is a pivot point during an
investigation, not a conclusion.""",
    """DNS is a frequent exfiltration and command-and-control channel because it
is rarely blocked. Signals: long random-looking subdomains, high query volume
to one domain, TXT-record replies larger than typical. None of these alone
proves tunnelling; correlate with the process that made the query.""",
    # --- linux ---
    """Linux persistence lives in predictable places a defender checks first:
cron (/etc/cron*, crontab -l), systemd units (systemctl list-unit-files,
~/.config/systemd/user), shell rc files (.bashrc, .profile), SSH authorized_keys,
and LD_PRELOAD in the environment. Enumerating these is baseline host triage.""",
    """File integrity: a defender establishes a known-good baseline (hashes of
system binaries) and compares against it. A changed hash on /bin/ls or
/usr/sbin/sshd is high signal. Without a baseline the same observation is only
a hypothesis, because the binary may have changed for a legitimate update.""",
    # --- incident response ---
    """The incident-response lifecycle: Preparation, Identification, Containment,
Eradication, Recovery, Lessons Learned (PICERL). Containment is chosen before
eradication so evidence is preserved and the adversary is not tipped off. An
analyst who jumps to eradication can destroy the very logs needed to scope the
incident.""",
    """Scoping an incident means answering: what accounts, hosts and data are
involved, over what time window, and by what technique. Every claim in a scope
should be tagged with its evidence type. "The attacker used valid credentials"
is an inference unless an authentication log OBSERVES the successful login.""",
    # --- detection engineering ---
    """A good detection rule states a hypothesis about adversary behaviour, the
data source that would reveal it, the logic, and the expected false positives.
Detection engineering is measured by precision and recall on labelled data, not
by how clever the rule looks. A rule with no documented false-positive profile
is not finished.""",
    """Detection as code: rules live in version control, are tested against
recorded telemetry (both malicious and benign), and ship through review. This
mirrors software engineering and is why detections should be diffable text, not
clicks in a console.""",
    # --- log analysis ---
    """Authentication logs answer who, from where, when and whether it succeeded.
A burst of failures followed by one success from the same source is the shape
of a successful password-guessing attempt — but "shape of" is a hypothesis;
confirmation needs the session that followed. Absence of failures before a
success can indicate valid stolen credentials.""",
    """When reading logs, first fix the timezone. Correlating a firewall log in
UTC with an endpoint log in local time is the most common analyst error and
produces phantom timelines. State the timezone explicitly in every timeline.""",
    # --- secure coding ---
    """Injection flaws (SQL, command, LDAP) share one cause: untrusted input is
concatenated into an interpreter's syntax. The fix is not escaping but
separation — parameterised queries and argument arrays keep data out of the
code plane. Escaping is a fragile fallback.""",
    """Output encoding prevents cross-site scripting by neutralising markup in
the context where data is placed (HTML body, attribute, JavaScript, URL). The
same string is safe in one context and dangerous in another, so encoding is
context-specific, applied at output, not at input.""",
    # --- vuln management ---
    """CVSS is a severity score, not a risk score. A 9.8 on an asset with no
network exposure and no sensitive data may be lower real risk than a 6.5 on an
internet-facing crown-jewel system. Risk = severity in the context of exposure
and asset value. Report both.""",
    """Patch prioritisation uses known-exploited status (e.g. CISA KEV) over raw
CVSS: a vulnerability with confirmed in-the-wild exploitation and a low score
often outranks a theoretical critical. Evidence of exploitation changes the
priority.""",
    # --- MITRE ATT&CK, defensive framing ---
    """ATT&CK organises adversary behaviour into tactics (the why: Initial
Access, Persistence, Lateral Movement, Exfiltration) and techniques (the how).
A defender maps their detections to techniques to find coverage gaps. Mapping a
single alert to a technique is a hypothesis about intent, not proof of it.""",
    """Lateral movement over SMB or WMI shows as authentication from one internal
host to many others in a short window, often with the same account. The
detection hypothesis is "one credential, many destinations, short time"; the
false positives are vulnerability scanners and management tooling, which must be
allow-listed with justification.""",
]

# Building blocks for the synthetic evidence-typed generator.
_HOSTS = ["web-01", "db-03", "dc-01", "workstation-14", "vpn-gw", "mail-02", "jump-01"]
_USERS = ["svc_backup", "a.khan", "root", "administrator", "j.doe", "svc_deploy"]
_SRCIPS = ["10.0.4.11", "10.0.9.240", "192.168.1.50", "172.16.8.4", "10.0.2.99"]
_TECHS = [
    ("T1110", "brute force", "many auth failures then a success"),
    ("T1021", "remote services", "internal auth from one host to many"),
    ("T1048", "exfiltration over alternative protocol", "large outbound DNS volume"),
    ("T1053", "scheduled task/job", "a new cron entry appeared"),
    ("T1059", "command interpreter", "a shell spawned from a browser process"),
]


def _synthetic_case(rng: random.Random) -> str:
    host = rng.choice(_HOSTS)
    user = rng.choice(_USERS)
    ip = rng.choice(_SRCIPS)
    tid, tname, shape = rng.choice(_TECHS)
    fails = rng.randint(12, 400)
    lines = [
        f"<|system|> Darknode analyst assistant. Separate observation from inference. Cite evidence.",
        f"<|user|> Triage alert {rng.randint(1000,9999)} on {host}.",
        f"<|assistant|>",
        f"Summary: possible {tname} ({tid}) involving account {user} on {host}.",
        f"<|evidence|> OBSERVED: {fails} failed logons for {user} from {ip}, then 1 success, within 6 minutes (auth log).",
        f"<|evidence|> OBSERVED: source {ip} is internal and not on the scanner allow-list.",
        f"<|evidence|> RETRIEVED: {user} is a service account that normally authenticates only from the backup host.",
        f"<|evidence|> INFERRED: the failure-then-success pattern is consistent with {shape}.",
        f"<|evidence|> HYPOTHESIS: the credential for {user} may be compromised. Not yet confirmed.",
        f"<|evidence|> UNKNOWN: whether any action was taken after the successful logon (session logs not yet retrieved).",
        f"Recommended next steps: retrieve the post-logon session for {user} on {host}; "
        f"check for new persistence (cron, systemd, authorized_keys); "
        f"consider containment (disable {user}) pending approval. RECOMMENDATION only — no action taken.",
    ]
    return "\n".join(lines)


def build_corpus(out_dir: str | Path, n_synthetic: int = 1200, seed: int = 7) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    (out_dir / "knowledge.txt").write_text(DOC.join(KNOWLEDGE), encoding="utf-8")
    synth = [_synthetic_case(rng) for _ in range(n_synthetic)]
    (out_dir / "synthetic_cases.txt").write_text(DOC.join(synth), encoding="utf-8")

    manifest = {
        "sources": [
            {"path": "corpus/knowledge.txt", "license": "CC0-1.0",
             "category": "defensive-knowledge",
             "provenance": "darknode-authored", "allowed_use": "train"},
            {"path": "corpus/synthetic_cases.txt", "license": "CC0-1.0",
             "category": "synthetic-triage", "provenance": "darknode-generated",
             "allowed_use": "train"},
        ]
    }
    (out_dir.parent / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                                   encoding="utf-8")
    return {"knowledge_docs": len(KNOWLEDGE), "synthetic_docs": len(synth)}


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "data/corpus"
    print(build_corpus(d))
