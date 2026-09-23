"""Adapter over decider-2b (Mapika/decider-2b), Qwen3.5-2B-Base + shared decoder.

The HF weights repo bundles the `decider` package source alongside the weights
(decider/infer.py, decider/systemone.py, ...), so one snapshot_download gets both. We
add that local snapshot dir to sys.path and import `decider.infer.Decider`, whose
`system_one(state, questions)` is already the Jev wire contract (same {"answers": {...}}
shape as jev.py/laya.py). Needs a CUDA GPU.

    DECIDER_MODEL=Mapika/decider-2b DECIDER_LOCAL_DIR=vendor/decider-2b \
        python3 eval/harness.py run --adapter decider --data <path> --dataset-name <name>
"""
import os
import sys
import time

from eval.render import render_candidate, render_state

_OPERATIONS = ("CLICK", "TYPE", "SELECT")
_DECIDER = None


def _decider():
    global _DECIDER
    if _DECIDER is None:
        model_id = os.environ.get("DECIDER_MODEL", "Mapika/decider-2b")
        local_dir = os.environ.get("DECIDER_LOCAL_DIR")
        if not local_dir:
            from huggingface_hub import snapshot_download

            local_dir = snapshot_download(model_id)
        if local_dir not in sys.path:
            sys.path.insert(0, local_dir)
        from decider.infer import Decider

        # use_graphs defaults to True on CUDA (shape-bucketed CUDA graphs via decider.engine);
        # DECIDER_USE_GRAPHS=0 falls back to eager DecisionModel if the graphs path errors.
        use_graphs_env = os.environ.get("DECIDER_USE_GRAPHS")
        use_graphs = None if use_graphs_env is None else use_graphs_env not in ("0", "false", "False")
        device = os.environ.get("DECIDER_DEVICE", "cuda")
        _DECIDER = Decider(local_dir, device=device, use_graphs=use_graphs)
    return _DECIDER


def predict(row):
    d = _decider()
    element_criteria = {str(c["idx"]): render_candidate(c) for c in row["candidates"]}
    questions = {
        "element": {
            "type": "choice",
            "instructions": "Given the goal and previous actions, which candidate element should be acted on next?",
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
    }

    t0 = time.perf_counter()
    result = d.system_one(render_state(row), questions)
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
