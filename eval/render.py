"""One candidate/state rendering shared by every adapter, so models are compared on the
same information. Mode comes from EVAL_RENDER: "default" (80 chars of text plus
attributes) or "compact" (60 chars, no attributes, two previous actions) for small contexts."""

import os

MODE = os.environ.get("EVAL_RENDER", "default")


def render_candidate(c, mode=None):
    mode = mode or MODE
    text = (c.get("text") or "").strip().replace("\n", " ")
    limit = 60 if mode == "compact" else 80
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    base = f'[{c["idx"]}] <{c.get("tag", "")}> role={c.get("role", "")} "{text}"'
    if mode != "compact":
        base = f"{base} attrs={c.get('attrs')}"
    return base


def render_state(row, mode=None):
    mode = mode or MODE
    lines = [f"Goal: {row['goal']}", f"Website: {row.get('website', '')} ({row.get('domain', '')})"]
    prev = row.get("previous_actions") or []
    if mode == "compact":
        prev = prev[-2:]
    if prev:
        lines.append("Previous actions:")
        lines.extend(f"- {a}" for a in prev)
    lines.append("Candidate elements:")
    lines.extend(render_candidate(c, mode) for c in row["candidates"])
    return "\n".join(lines)


PHRASINGS = {
    "a": "Given the goal and previous actions, which candidate element should be acted on next?",
    "b": "Which one of the listed elements is the right next target for this goal?",
    "c": "Pick the element to interact with next to make progress on the task.",
}


def element_instructions():
    """Instruction text for the element question; EVAL_PHRASING=a|b|c selects a variant
    so phrasing sensitivity can be measured with identical data."""
    return PHRASINGS[os.environ.get("EVAL_PHRASING", "a")]
