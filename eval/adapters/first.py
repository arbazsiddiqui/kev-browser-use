"""Baseline adapter: always pick candidate idx 0 and operation CLICK."""
import time

OPERATIONS = ("CLICK", "TYPE", "SELECT")


def predict(row):
    t0 = time.perf_counter()
    n = len(row["candidates"])
    element_probs = [1.0 if i == 0 else 0.0 for i in range(n)]
    operation_probs = {op: (1.0 if op == "CLICK" else 0.0) for op in OPERATIONS}
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "element_probs": element_probs,
        "operation_probs": operation_probs,
        "latency_ms": latency_ms,
    }
