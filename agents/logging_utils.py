"""PII/financial data scrubbing before logging (goals doc: "PII/financial
data scrubbing before logging"). The raw sensitive value is redacted before
it's ever handed to the logger, so nothing sensitive touches disk — this
isn't a log post-processing step.
"""

import logging
import re
from pathlib import Path

LOG_PATH = Path(__file__).parent.parent / "data" / "tool_calls.log"

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
_AMOUNT_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")

# Tool-arg keys whose value is redacted outright, regardless of shape —
# cheaper and more reliable than regex for things like amount=150.0, which
# has no "$" to pattern-match on.
_SENSITIVE_KEYS = {"amount", "email", "phone", "body", "subject"}


def _scrub_text(value: str) -> str:
    value = _EMAIL_RE.sub("[REDACTED_EMAIL]", value)
    value = _PHONE_RE.sub("[REDACTED_PHONE]", value)
    value = _AMOUNT_RE.sub("[REDACTED_AMOUNT]", value)
    return value


def scrub_args(args: dict) -> dict:
    """Returns a redacted copy of a tool call's args, safe to log."""
    scrubbed = {}
    for key, value in args.items():
        if key.lower() in _SENSITIVE_KEYS:
            scrubbed[key] = "[REDACTED]"
        elif isinstance(value, str):
            scrubbed[key] = _scrub_text(value)
        else:
            scrubbed[key] = value
    return scrubbed


tool_call_logger = logging.getLogger("business_ops.tool_calls")
tool_call_logger.setLevel(logging.INFO)
if not tool_call_logger.handlers:
    _handler = logging.FileHandler(LOG_PATH)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    tool_call_logger.addHandler(_handler)
    tool_call_logger.propagate = False
