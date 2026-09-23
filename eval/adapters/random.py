"""Baseline adapter: uniform-random element and operation probabilities.

Seeded per-row by the row id so a re-run is reproducible, while the distribution
over a whole dataset is still uniform-random (no signal from goal/candidates).
"""
import random as _random
import time

OPERATIONS = ("CLICK", "TYPE", "SELECT")


def predict(row):
    t0 = time.perf_counter()
    rng = _random.Random(str(row["id"]))

    n = len(row["candidates"])
    raw = [rng.random() for _ in range(n)]
    total = sum(raw)
    element_probs = [x / total for x in raw]

    op_raw = [rng.random() for _ in OPERATIONS]
    op_total = sum(op_raw)
    operation_probs = {op: p / op_total for op, p in zip(OPERATIONS, op_raw)}

    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "element_probs": element_probs,
        "operation_probs": operation_probs,
        "latency_ms": latency_ms,
    }
