"""Shared redaction helpers for memory-derived text."""

from __future__ import annotations

import re

from agentos.redact import redact_sensitive_text

# A plain \b before the keyword only fires at a transition between a word
# character and a non-word character. That misses a common way a credential
# name is actually written: snake_case ("reset_token") glues the keyword to
# its qualifier with an underscore, which is itself a word character, so no
# \b exists there at all. A value sitting behind one of these -- and
# carrying no recognisable vendor prefix for the shape-based pass in
# agentos.redact to catch by itself -- would otherwise reach durable,
# searchable memory unmasked.
#
# camelCase ("resetToken") is deliberately NOT extended the same way: it is
# syntactically identical to "sellToken", which this module's own test
# suite explicitly protects from redaction (web3 asset names, same concern
# documented in agentos.redact). There is no boundary- or shape-based way
# to tell "resetToken" from "sellToken" -- that needs the qualifier
# allowlist agentos.redact already maintains -- so widening the boundary
# for camelCase here would just trade one false negative for a regression
# on an already-decided case.
_KEYWORD_PATTERN = re.compile(
    r"(?i)(?:\b|(?<=_))(api[_-]?key|secret|token|password)(\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|\S+)"
)

def _replace_keyword(match: re.Match[str]) -> str:
    val = match.group(3)
    stripped_val = val.strip("\"'")
    if "***" in stripped_val or "«redacted" in stripped_val or "[REDACTED]" in stripped_val:
        return match.group(0)
    return f"{match.group(1)}{match.group(2)}[REDACTED]"


def redact_memory_text(text: str) -> str:
    # force=True: AGENTOS_REDACT_SECRETS=0 is an *egress* escape hatch and must
    # not unmask what gets written to durable memory.
    redacted = redact_sensitive_text(text, force=True) or text
    return _KEYWORD_PATTERN.sub(_replace_keyword, redacted)
