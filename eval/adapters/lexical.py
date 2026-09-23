"""Baseline adapter: token overlap between (goal + last previous action) and each
candidate's text/attrs, softmax'd into a distribution. Operation is a tag/role
heuristic on the top-scoring candidate (never looks at the gold label)."""
import math
import re
import time

OPERATIONS = ("CLICK", "TYPE", "SELECT")
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text):
    if text is None:
        return set()
    if not isinstance(text, str):
        text = str(text)
    return set(_WORD_RE.findall(text.lower()))


def _candidate_tokens(c):
    toks = _tokens(c.get("text"))
    toks |= _tokens(c.get("tag"))
    toks |= _tokens(c.get("role"))
    attrs = c.get("attrs")
    if isinstance(attrs, dict):
        for k, v in attrs.items():
            toks |= _tokens(k)
            toks |= _tokens(v)
    else:
        toks |= _tokens(attrs)
    return toks


def _softmax(scores):
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(scores)] * len(scores)
    return [e / total for e in exps]


def predict(row):
    t0 = time.perf_counter()

    query_tokens = _tokens(row.get("goal"))
    prev = row.get("previous_actions") or []
    if prev:
        last = prev[-1]
        query_tokens |= _tokens(last if isinstance(last, str) else str(last))

    scores = [float(len(query_tokens & _candidate_tokens(c))) for c in row["candidates"]]
    element_probs = _softmax(scores)

    top_idx = max(range(len(scores)), key=lambda i: (scores[i], -i))
    top = row["candidates"][top_idx]
    tag = (top.get("tag") or "").lower()
    role = (top.get("role") or "").lower()
    if tag == "select" or role in ("combobox", "listbox"):
        op = "SELECT"
    elif tag in ("input", "textarea") and role not in ("button", "checkbox", "radio", "link"):
        op = "TYPE"
    else:
        op = "CLICK"
    operation_probs = {o: (0.7 if o == op else 0.15) for o in OPERATIONS}

    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "element_probs": element_probs,
        "operation_probs": operation_probs,
        "latency_ms": latency_ms,
    }
