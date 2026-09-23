"""Adapter over NanoJev (TianyuCodings/NanoJev, weights C-Tianyu/NanoJev), Qwen3-0.6B +
parallel decision heads (standard attention, no DeltaNet).

Trained on ViZDoom/Maze/Snake, not on browser DOM data -- this evaluation is genuinely
zero-shot/out-of-domain for it, same as for every other competitor here. No pip package;
the repo ships a `DecisionPredictor` under source/scripts/predict_toy_decisions.py inside
the HF weight snapshot itself (downloaded with allow_patterns=["source/*", ...] per the
model card). Request shape is a batch of states: {"states": [{"id", "state", "questions"}]}
-- one state per call here, batch size 1, matching every other adapter's per-row latency
measurement. The model card's own quickstart never prints the actual response shape
(`print(answer)` with output elided), so `_answers_for` below tries the plausible
response shapes defensively and raises with the raw payload on the first row if none of
them match, rather than silently mis-scoring every row. Needs a CUDA GPU.

    NANOJEV_CHECKPOINT=vendor/NanoJev-unified \
        python3 eval/harness.py run --adapter nanojev --data <path> --dataset-name <name>
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
        checkpoint_dir = os.environ.get("NANOJEV_CHECKPOINT")
        if not checkpoint_dir:
            from huggingface_hub import snapshot_download

            checkpoint_dir = snapshot_download(
                "C-Tianyu/NanoJev",
                revision="unified-games-v1",
                allow_patterns=["best.safetensors", "config.json", "backbone_config/*", "tokenizer/*", "source/*"],
            )
        scripts_dir = os.path.join(checkpoint_dir, "source", "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from predict_toy_decisions import DecisionPredictor

        device = os.environ.get("NANOJEV_DEVICE", "cuda:0")
        _MODEL = DecisionPredictor(checkpoint_dir, device_name=device, precision="bf16", disable_native_triton=True)
    return _MODEL


def _answers_for(payload, state_id, qid):
    """Try the plausible response shapes for one question's answer dict."""
    candidates = []
    if isinstance(payload, dict):
        if "answers" in payload:
            candidates.append(payload["answers"])
        if "states" in payload:
            for s in payload["states"]:
                if s.get("id") == state_id:
                    candidates.append(s.get("answers", s))
        if state_id in payload:
            candidates.append(payload[state_id])
    if isinstance(payload, list):
        for s in payload:
            if isinstance(s, dict) and s.get("id") == state_id:
                candidates.append(s.get("answers", s))
    for c in candidates:
        if isinstance(c, dict) and qid in c:
            return c[qid]
    raise RuntimeError(
        f"nanojev: couldn't find question {qid!r} for state {state_id!r} in response shape "
        f"{type(payload).__name__}; raw payload: {payload!r}"
    )


def predict(row):
    model = _model()
    element_criteria = {str(c["idx"]): render_candidate(c) for c in row["candidates"]}
    request = {
        "states": [
            {
                "id": row["id"],
                "state": render_state(row),
                "questions": {
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
                },
            }
        ]
    }

    t0 = time.perf_counter()
    payload = model.predict(request)
    latency_ms = (time.perf_counter() - t0) * 1000

    element_answer = _answers_for(payload, row["id"], "element")
    probs_by_key = element_answer.get("probabilities", element_answer) if isinstance(element_answer, dict) else {}
    element_probs = [float(probs_by_key.get(str(c["idx"]), 0.0)) for c in row["candidates"]]
    s = sum(element_probs)
    if s > 0:
        element_probs = [p / s for p in element_probs]

    operation_answer = _answers_for(payload, row["id"], "operation")
    op_probs_raw = operation_answer.get("probabilities", operation_answer) if isinstance(operation_answer, dict) else {}
    operation_probs = {op: float(op_probs_raw.get(op, 0.0)) for op in _OPERATIONS}
    s2 = sum(operation_probs.values())
    if s2 > 0:
        operation_probs = {k: v / s2 for k, v in operation_probs.items()}

    return {
        "element_probs": element_probs,
        "operation_probs": operation_probs,
        "latency_ms": latency_ms,
    }
