"""Converts eval-schema JSONL rows (data/README.md) into Kev's own training record
format, so github.com/jaredpalmer/kev's `kev/train.py --data <out>` can fine-tune
jaredpalmer/kev-0.6b directly on these rows.

Kev's `--data` loader (kev/data.py: load_records) wants one JSON object per line, shaped
like its /v1/systemone request plus a label per question:

    {"state": "...", "questions": {"team": {"type": "choice", "instructions": "...",
                                             "criteria": {"billing": "...", "other": null},
                                             "label": "billing"}}}

Two Choice questions per row, matching the instructions text eval/adapters/kev.py already
sends at inference time so training and serving see the same wording:

  "element":   10 options, one per candidate in idx order, criteria key = a short label
               that packs in the idx so it is unique even when tag/text collide (the smoke
               set has rows with 8 identical "Select" buttons); label = the gold candidate's.
  "operation": CLICK/TYPE/SELECT; label = the row's operation.

State text follows eval/render.py's compact candidate lines, as
compact mode (goal, last 2 previous actions, the candidate table via
eval.render.render_candidate), minus the [CAND] marker token -- Kev has its own option
delimiters (kev/model.py's encode() wraps each option in its own <opt>...</opt> span), so
the marker would just be noise here.

Usage:
    python3 train/data/to_kev_format.py --in data/train_m2w.jsonl --out train/data/kev_train.jsonl
    python3 train/data/to_kev_format.py --in data/val.jsonl --out train/data/kev_val.jsonl
    python3 train/data/to_kev_format.py --in data/eval_smoke.jsonl --out /tmp/kev_smoke.jsonl --limit 10
    python3 train/data/to_kev_format.py --selftest   # round-trip check on data/eval_smoke.jsonl, no model
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.harness import load_dataset  # noqa: E402  (idx-sorts, validates 10 candidates/row)
from eval.render import render_candidate  # noqa: E402

OPERATIONS = ("CLICK", "TYPE", "SELECT")
ELEMENT_INSTRUCTIONS = "Which element should be acted on?"
OPERATION_INSTRUCTIONS = "What operation should be performed on the target element?"


def candidate_label(c):
    """Short, unique-by-construction option label, e.g. '[3] <a> "Cars"'. The leading [idx]
    guarantees uniqueness across the 10 candidates even when tag/text collide; role/attrs
    are left out since they already sit in the state's candidate table (render_candidate)
    and the option only has to name which row of that table it points to."""
    text = (c.get("text") or "").strip().replace("\n", " ")
    if len(text) > 60:
        text = text[:57] + "..."
    return f'[{c["idx"]}] <{c.get("tag", "")}> "{text}"'


def render_state(row):
    """goal + last 2 previous actions + the candidate table in eval/render.py's compact mode."""
    lines = [f"Goal: {row['goal']}"]
    prev = (row.get("previous_actions") or [])[-2:]
    if prev:
        lines.append("Previous actions:")
        lines.extend(f"- {a}" for a in prev)
    lines.append("Candidate elements:")
    lines.extend(render_candidate(c, mode="compact") for c in row["candidates"])
    return "\n".join(lines)


def to_kev_record(row):
    labels = [candidate_label(c) for c in row["candidates"]]
    if len(set(labels)) != len(labels):
        raise ValueError(f"row {row.get('id')!r}: candidate labels are not unique: {labels}")
    operation = row["operation"]
    if operation not in OPERATIONS:
        raise ValueError(f"row {row.get('id')!r}: unknown operation {operation!r}")
    return {
        "state": render_state(row),
        "questions": {
            "element": {
                "type": "choice",
                "instructions": ELEMENT_INSTRUCTIONS,
                "criteria": {label: None for label in labels},
                "label": labels[row["gold_idx"]],
            },
            "operation": {
                "type": "choice",
                "instructions": OPERATION_INSTRUCTIONS,
                "criteria": {op: None for op in OPERATIONS},
                "label": operation,
            },
        },
        "_meta": {"id": row.get("id")},
    }


def convert(in_path, out_path, limit=None):
    rows = load_dataset(in_path)
    if limit:
        rows = rows[:limit]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as f:
        for row in rows:
            # ensure_ascii: page text can carry U+2028 and NEL characters that line-splitting loaders treat as newlines
            f.write(json.dumps(to_kev_record(row), ensure_ascii=True) + "\n")
            n += 1
    return n


def selftest(path=None):
    """No model, no Kev checkout needed: round-trips every row of `path` (default
    data/eval_smoke.jsonl) through to_kev_record() and checks what kev.data.load_records
    requires of a record (per kev/data.py's docstring, fetched from github.com/jaredpalmer/kev)
    -- non-empty string state, a label per question, and the gold candidate's label present
    among the element question's own options -- plus a plain JSON round-trip."""
    path = path or (REPO_ROOT / "data" / "eval_smoke.jsonl")
    rows = load_dataset(path)
    ok = True
    for row in rows:
        rec = to_kev_record(row)
        if not rec["state"] or not isinstance(rec["state"], str):
            print(f"FAIL {row['id']}: empty/non-string state"); ok = False
            continue
        element_q = rec["questions"]["element"]
        gold_label = candidate_label(row["candidates"][row["gold_idx"]])
        if gold_label not in element_q["criteria"]:
            print(f"FAIL {row['id']}: gold label {gold_label!r} missing from options"); ok = False
        if element_q["label"] != gold_label:
            print(f"FAIL {row['id']}: element label {element_q['label']!r} != gold {gold_label!r}"); ok = False
        if len(element_q["criteria"]) != 10:
            print(f"FAIL {row['id']}: expected 10 element options, got {len(element_q['criteria'])}"); ok = False
        op_q = rec["questions"]["operation"]
        if op_q["label"] != row["operation"] or op_q["label"] not in op_q["criteria"]:
            print(f"FAIL {row['id']}: operation label/criteria mismatch"); ok = False
        reloaded = json.loads(json.dumps(rec, ensure_ascii=True))
        if reloaded != rec:
            print(f"FAIL {row['id']}: JSON round-trip changed the record"); ok = False
    print(f"selftest: {len(rows)} rows from {path}, {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", help="JSONL rows in the data/README.md schema")
    ap.add_argument("--out", dest="out_path", help="Kev record JSONL (kev.data.load_records schema)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--selftest", action="store_true", help="round-trip check on data/eval_smoke.jsonl, no model")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    if not args.in_path or not args.out_path:
        ap.error("--in and --out are required unless --selftest")

    n = convert(args.in_path, args.out_path, args.limit)
    print(f"wrote {n} Kev records to {args.out_path}")


if __name__ == "__main__":
    main()
