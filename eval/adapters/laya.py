"""Adapter over the local Laya System One model (convaiinnovations/laya).

Runs on a CUDA GPU, e.g.:

    LAYA_DEVICE=cuda python3 eval/harness.py run --adapter laya --data <path> \
        --dataset-name <name>

One agent.system_one() call per row: a `choice` question over the 10 candidates
(short strings) and a `choice` question over the 3 operations, sharing one state
string (goal + previous actions + candidate table).

Fine-tuned checkpoint loading: ShaunSpark/laya-mind2web-browser-agent (Hugging Face) is
convaiinnovations/laya fine-tuned on Mind2Web, distributed as a bare pytorch_model.bin
rather than a `laya.load()`-able repo. Its model card loads it as:

    agent = laya.load("convaiinnovations/laya")
    agent.model.load_state_dict(torch.load("pytorch_model.bin", map_location="cpu"))

which this adapter replicates via two env vars, applied to `agent.model` right after
`laya.load(...)`:
  - LAYA_STATE_DICT=<path to .bin/.safetensors>: load that local file.
  - LAYA_HF_FILE=<repo>:<filename>, e.g. LAYA_HF_FILE=ShaunSpark/laya-mind2web-browser-agent:pytorch_model.bin:
    downloads <filename> from <repo> at run time via huggingface_hub.hf_hub_download (into
    the HF cache) and loads that.
Either way the load is strict=True: any missing or unexpected key is printed to stderr and
then raised, so a bad checkpoint fails the run instead of silently running the base model.
Neither var set -> behaviour is unchanged (no state dict load, base laya.load() weights only).
"""
import os
import sys
import time

from eval.render import element_instructions, render_candidate, render_state

_AGENT = None
_OPERATIONS = ("CLICK", "TYPE", "SELECT")


def _resolve_state_dict_path():
    hf_file = os.environ.get("LAYA_HF_FILE")
    if hf_file:
        if ":" not in hf_file:
            raise RuntimeError(f"LAYA_HF_FILE must be '<repo>:<filename>', got {hf_file!r}")
        repo_id, filename = hf_file.split(":", 1)
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=repo_id, filename=filename)
    return os.environ.get("LAYA_STATE_DICT")


def _load_state_dict(model, path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(path)
    else:
        import torch

        state_dict = torch.load(path, map_location="cpu")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        print(f"[laya] strict load_state_dict failed for {path}: {exc}", file=sys.stderr)
        raise
    print(f"[laya] loaded state dict from {path} (strict, no missing/unexpected keys)", file=sys.stderr)


def _agent():
    global _AGENT
    if _AGENT is None:
        import laya
        device = os.environ.get("LAYA_DEVICE", "mps")
        model_id = os.environ.get("LAYA_MODEL", "convaiinnovations/laya")
        _AGENT = laya.load(model_id, device=device)
        state_dict_path = _resolve_state_dict_path()
        if state_dict_path:
            _load_state_dict(_AGENT.model, state_dict_path)
    return _AGENT


def predict(row):
    agent = _agent()
    element_criteria = {str(c["idx"]): render_candidate(c) for c in row["candidates"]}
    questions = {
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
    }

    t0 = time.perf_counter()
    result = agent.system_one(render_state(row), questions)
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
    }
