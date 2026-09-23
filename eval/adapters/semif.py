"""Adapter replicating SemIf's "direct" scorer (github.com/TheoLeeCJ/SemIf, MIT,
src/semif_phase1/core.py + direct.py): a frozen chat model gets a system prompt, a JSON
payload {evidence, criterion, options:[{letter, description}]} through the chat template
(no thinking, generation prompt appended), and the answer is read from the last-position
logits of the uppercase letter tokens A.. only, softmaxed. No training, no generation.

Two SemIf rows per eval step, sharing the state: the element choice (10 options, letters A-J)
and the operation choice (CLICK/TYPE/SELECT). SEMIF_MODEL defaults to Qwen/Qwen3.5-4B, the
model SemIf reports 0.845 agreement with Jev on; its browser tier is a 3 GB GGUF of the same.
Runs on the L4 (SEMIF_DEVICE, default cuda).
"""

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from eval.render import element_instructions, render_candidate, render_state  # noqa: E402

MODEL_ID = os.environ.get("SEMIF_MODEL", "Qwen/Qwen3.5-4B")
LETTERS = "ABCDEFGHIJKLMNOP"
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
OPERATIONS = ["CLICK", "TYPE", "SELECT"]
OPERATION_QUESTION = "Which operation should be performed on the target element?"

_STATE = None


def _state():
    global _STATE
    if _STATE is None:
        device = os.environ.get("SEMIF_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval()
        slots = []
        for letter in LETTERS:
            enc = tok.encode(letter, add_special_tokens=False)
            if len(enc) != 1 or tok.decode(enc) != letter:
                raise ValueError(f"answer slot {letter!r} is not one exact round-trip token")
            slots.append(enc[0])
        _STATE = {"tok": tok, "model": model, "device": device, "slots": slots}
    return _STATE


def messages(state, question, descriptions):
    payload = {
        "evidence": state,
        "criterion": question,
        "options": [{"letter": LETTERS[i], "description": d} for i, d in enumerate(descriptions)],
    }
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _score(s, state, question, descriptions):
    prompt = s["tok"].apply_chat_template(
        messages(state, question, descriptions), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = s["tok"].encode(prompt, add_special_tokens=False)
    inputs = torch.tensor([ids], dtype=torch.long, device=s["device"])
    with torch.inference_mode():
        logits = s["model"](input_ids=inputs).logits[0, -1].float()
    selected = logits[s["slots"][: len(descriptions)]]
    return torch.softmax(selected, dim=-1).tolist()


def predict(row):
    s = _state()
    state = render_state(row)
    descriptions = [render_candidate(c) for c in row["candidates"]]
    t0 = time.perf_counter()
    element_probs = _score(s, state, element_instructions(), descriptions)
    operation_probs = _score(s, state, OPERATION_QUESTION, OPERATIONS)
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "element_probs": element_probs,
        "operation_probs": dict(zip(OPERATIONS, operation_probs)),
        "latency_ms": latency_ms,
    }
