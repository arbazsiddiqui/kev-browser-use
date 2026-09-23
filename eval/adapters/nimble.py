"""Adapter over Bespoke Nimble-9B (bespokelabsai/nimble, weights
bespokelabs/Bespoke-Nimble-9B), Qwen3.5-9B LoRA, softmax over allowed answer-token
logits from one generation step.

Unlike the other Jev-family clones here, Nimble has no `system_one`-shaped wire
contract -- its `inference.py::NimbleModel.score(context, schema)` takes a JSON-Schema-
style field spec (`{"type": "enum"|"boolean", "description", "choices", "choice_descriptions"}`)
and constrains generation to one allowed answer token per field (A/B/C/... over up to 26
choices). `inference.py` checks a SHA-256 of its sibling `parallel_schema.py` against
`schema_config.json`, so both files must be downloaded from the HF weights repo itself
(not reimplemented here) and kept side by side -- hence the whole-repo snapshot_download
plus a sys.path insert, rather than importing a pip package. Requires CUDA+bf16
(NimbleModel raises otherwise).

    NIMBLE_MODEL_DIR=vendor/Bespoke-Nimble-9B \
        python3 eval/harness.py run --adapter nimble --data <path> --dataset-name <name>
"""
import os
import sys
import time

from eval.render import render_candidate, render_state

_OPERATIONS = ("CLICK", "TYPE", "SELECT")
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        model_dir = os.environ.get("NIMBLE_MODEL_DIR")
        if not model_dir:
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download("bespokelabs/Bespoke-Nimble-9B")
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from inference import NimbleModel

        _MODEL = NimbleModel(model_dir)
    return _MODEL


def predict(row):
    model = _model()
    element_choices = [str(c["idx"]) for c in row["candidates"]]
    element_descriptions = {str(c["idx"]): render_candidate(c) for c in row["candidates"]}
    schema = {
        "element": {
            "type": "enum",
            "description": "Given the goal and previous actions, which candidate element should be acted on next?",
            "choices": element_choices,
            "choice_descriptions": element_descriptions,
        },
        "operation": {
            "type": "enum",
            "description": "What operation should be performed on the target element?",
            "choices": list(_OPERATIONS),
            "choice_descriptions": {
                "CLICK": "Click the element",
                "TYPE": "Type text into the element",
                "SELECT": "Select an option from the element",
            },
        },
    }

    t0 = time.perf_counter()
    result = model.score(render_state(row), schema)
    latency_ms = (time.perf_counter() - t0) * 1000

    fields = result["fields"]
    probs_by_key = fields["element"]["probabilities"]
    element_probs = [float(probs_by_key.get(str(c["idx"]), 0.0)) for c in row["candidates"]]
    s = sum(element_probs)
    if s > 0:
        element_probs = [p / s for p in element_probs]

    op_probs_raw = fields["operation"]["probabilities"]
    operation_probs = {op: float(op_probs_raw.get(op, 0.0)) for op in _OPERATIONS}
    s2 = sum(operation_probs.values())
    if s2 > 0:
        operation_probs = {k: v / s2 for k, v in operation_probs.items()}

    return {
        "element_probs": element_probs,
        "operation_probs": operation_probs,
        "latency_ms": latency_ms,
    }
