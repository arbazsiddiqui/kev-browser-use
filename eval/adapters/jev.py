"""Adapter over the TypeSafe System One API (Jev), POST /v1/systemone.

Auth: Bearer TYPESAFE_API_KEY (export it before running). One request per
row with two `choice` questions -- element (10 candidates) and operation (3 ops)
-- sharing one state string. Retries with exponential backoff on 429/529.
"""
import json
import os
import time

from eval.render import element_instructions, render_candidate, render_state
import urllib.error
import urllib.request

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
_OPERATIONS = ("CLICK", "TYPE", "SELECT")
_MAX_RETRIES = 5
_TIMEOUT_S = 30


def _api_key():
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY not set -- export it before running the jev adapter")
    return key


def _request(body):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    delay = 1.0
    last_err = None
    for attempt in range(_MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 529) and attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
            body_text = e.read().decode("utf-8", "replace") if hasattr(e, "read") else ""
            raise RuntimeError(f"TypeSafe API HTTP {e.code}: {body_text[:500]}") from e
        except urllib.error.URLError as e:
            last_err = e
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise last_err


def predict(row):
    element_criteria = {str(c["idx"]): render_candidate(c) for c in row["candidates"]}
    body = {
        "state": render_state(row),
        "model": MODEL,
        "questions": {
            "element": {
                "type": "choice",
                "instructions": element_instructions(),
                "criteria": element_criteria,
            },
            "operation": {
                "type": "choice",
                "instructions": "What operation should be performed on the target element?",
                "criteria": {
                    "CLICK": "Click the element",
                    "TYPE": "Type text into the element",
                    "SELECT": "Select an option from the element",
                },
            },
        },
    }

    t0 = time.perf_counter()
    result = _request(body)
    latency_ms = (time.perf_counter() - t0) * 1000

    probs_by_key = result["answers"]["element"]["probabilities"]
    element_probs = [float(probs_by_key.get(str(c["idx"]), 0.0)) for c in row["candidates"]]
    s = sum(element_probs)
    if s > 0:
        element_probs = [p / s for p in element_probs]

    op_probs_raw = result["answers"]["operation"]["probabilities"]
    operation_probs = {op: float(op_probs_raw.get(op, 0.0)) for op in _OPERATIONS}
    s2 = sum(operation_probs.values())
    if s2 > 0:
        operation_probs = {k: v / s2 for k, v in operation_probs.items()}

    return {
        "element_probs": element_probs,
        "operation_probs": operation_probs,
        "latency_ms": latency_ms,
        "usage": result.get("usage") or {},
    }
