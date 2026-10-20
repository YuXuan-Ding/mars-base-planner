"""Guardrails for Ask Claude, enforced in code around the model (not only in the prompt).

  screen(question)  refuses, before Claude is called, requests to reveal the system prompt or credentials,
                    prompt-injection phrasing, requests to harvest personal data, and requests for weapon
                    or explosive instructions
  clean(answer)     after Claude answers: redacts emails, phone numbers, SSNs, card and ID numbers, key-like
                    strings and the values of secret environment variables, and replaces any answer that
                    echoes the system prompt with a refusal

Found by the VEYZL audit of POST /construct/ask (prompt injection, secret leakage, PII leak, disallowed content;
EU AI Act Art. 15, GDPR Art. 5). The patterns are deliberately narrow so ordinary project questions ("how do I
set the API key?", "what does blast and clear cost?") still reach Claude.
"""
from __future__ import annotations

import os
import re

REFUSAL = ("I can't help with that. I only answer questions about the Mars Base Planner, and I don't share my "
           "instructions, configuration, credentials or anyone's personal data.")

_I = re.IGNORECASE | re.DOTALL
_SECRET_WORDS = r"(api[\s_-]?keys?|passwords?|pass\s?phrases?|tokens?|secrets?|credentials?|private\s+keys?)"
_PII_WORDS = (r"(e-?mail(\s+address(es)?)?s?|phone\s+numbers?|telephone|social\s+security|ssns?|passport(\s+numbers?)?|"
              r"(id|identification|account|card|licen[cs]e)\s+numbers?|personal\s+(details?|data|information|info)|"
              r"home\s+address(es)?|dates?\s+of\s+birth)")
_HARM_OBJECTS = (r"(explosive(\s+device)?s?|bombs?|ieds?|detonators?|pipe\s+bombs?|molotov|nerve\s+agents?|sarin|"
                 r"bio-?weapons?|chemical\s+weapons?|ricin|anthrax|ghost\s+guns?|untraceable\s+(guns?|firearms?))")

RULES = {
    "prompt_extraction": [
        r"\b(system|hidden|initial|original|developer|internal)\s+(prompt|instructions?|message|rules)\b",
        r"\b(reveal|print|show|repeat|output|dump|leak|recite|disclose)\b.{0,40}\b(your|the)\s+(instructions|prompt|rules|"
        r"guidelines|configuration|config)\b",
    ],
    "injection": [
        r"\bignore\s+(all\s+|any\s+|the\s+|your\s+)?(previous\s+|prior\s+|above\s+|earlier\s+|preceding\s+)?"
        r"(instructions|rules|directions|guidelines|prompts?)\b",
        r"\b(disregard|forget|override|bypass)\s+(all\s+|any\s+|your\s+|the\s+)?(previous\s+|prior\s+)?"
        r"(instructions|rules|guidelines|restrictions|safety)\b",
        r"\b(developer|god|jailbreak|dan)\s+mode\b",
        r"\byou\s+are\s+now\s+(an?\s+)?(unfiltered|unrestricted|uncensored|evil|different)\b",
    ],
    "secret_request": [
        rf"\b(reveal|leak|dump|exfiltrat\w*|expose|disclose)\b.{{0,60}}\b{_SECRET_WORDS}",
        rf"\b(print|show|give|list|tell|output|send)\b\s+(me\s+)?(your|the|any|all|every)\s+{_SECRET_WORDS}",
        rf"\b{_SECRET_WORDS}\b.{{0,40}}\b(in|from)\s+(your|the)\s+(context|prompt|memory|environment|env|config\w*|\.env|"
        r"system)\b",
    ],
    "pii_harvest": [
        rf"\b(print|list|show|give|repeat|extract|tell|output|dump|collect|transcribe)\b.{{0,60}}\b{_PII_WORDS}\b",
    ],
    "harmful": [
        rf"\b(how\s+(do|can|would|to)|step[\s-]+by[\s-]+step|instructions?|recipe|guide|teach)\b.{{0,60}}"
        rf"\b(build|make|assemble|synthesi[sz]e|create|construct|produce)\b.{{0,40}}\b{_HARM_OBJECTS}",
        rf"\b(build|make|assemble|synthesi[sz]e)\b\s+(an?\s+|some\s+)?{_HARM_OBJECTS}",
    ],
}
_COMPILED = {k: [re.compile(p, _I) for p in v] for k, v in RULES.items()}


def screen(question: str) -> str | None:
    """The rule a question breaks (prompt_extraction, injection, secret_request, pii_harvest, harmful), or None."""
    q = re.sub(r"\s+", " ", question or "")
    for name, pats in _COMPILED.items():
        if any(p.search(q) for p in pats):
            return name
    return None


REDACT = [
    ("[credential removed]", re.compile(r"\bsk-ant-[\w-]{10,}|\b(?:sk|pk|rk|xai)-[A-Za-z0-9_-]{20,}|\bgh[pousr]_[A-Za-z0-9]{20,}"
                                        r"|\bAKIA[0-9A-Z]{16}\b|\bBearer\s+[A-Za-z0-9._~+/=-]{20,}", re.I)),
    ("[email removed]", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("[ID number removed]", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),                         # US SSN
    ("[card number removed]", re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{1,4}\b")),
    ("[phone number removed]", re.compile(r"(?<![\w.,])(?:\+\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?![\w.,]*\d)")),
]
_SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)", re.I)


def _env_secrets() -> list[str]:
    return [v for k, v in os.environ.items() if _SECRET_ENV.search(k) and v and len(v) >= 8]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)) if len(s.strip()) >= 40]


def clean(answer: str, system_prompt: str = "") -> tuple[str, list[str]]:
    """The answer with personal data and secrets redacted, and the kinds of things removed. An answer that
    repeats a sentence of the system prompt is replaced by the refusal."""
    flat = re.sub(r"\s+", " ", answer)
    if system_prompt and any(s in flat for s in _sentences(system_prompt)):
        return REFUSAL, ["system_prompt"]
    out, hits = answer, []
    for secret in _env_secrets():
        if secret in out:
            out, hits = out.replace(secret, "[credential removed]"), hits + ["credential"]
    for label, pat in REDACT:
        out, n = pat.subn(label, out)
        if n:
            hits.append(label.strip("[]").replace(" removed", ""))
    return out, hits
