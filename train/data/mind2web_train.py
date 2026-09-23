"""Build the training set from Mind2Web's TRAIN split.

Source: Hugging Face dataset osunlp/Mind2Web (CC BY 4.0), the 11 plain-JSON shards
under data/train/ (no password, unlike the test.zip the eval set uses). Streamed
directly over HTTP with ijson, nothing is written to disk unparsed.

Same row schema as eval/build_dataset.py: id, split, website, domain, goal,
previous_actions, candidates[10], gold_idx, operation, value. Rendering (visible
text extraction, attrs, negative ranking) is reused from eval.build_dataset so a
row means the same thing whether it came from train or test.

Every step with a pos_candidate becomes a "base" row (9 negatives = the hardest by
scores_all_data.pkl rank, seed 0) plus an "augmented" row (9 negatives = a seeded
random sample of the same pool, seed 1; independent gold-position shuffle, seed 1)
so a model can't learn to key off decoy identity or gold position for a given step.

10% of WEBSITES are held out as data/val.jsonl (every task and both variants of those sites); a held-out
task's steps go to val together, so no row from a val task ever reaches train.

Usage:
    python3 train/data/mind2web_train.py            # full build: raw pass + split
    python3 train/data/mind2web_train.py --phase-a  # only stream HF -> data/raw/train_m2w_raw.jsonl
    python3 train/data/mind2web_train.py --split     # only re-split data/raw/train_m2w_raw.jsonl
"""

import argparse
import json
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import ijson
import lxml.html
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.build_dataset import render_candidate, rank_negatives  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_PATH = RAW_DIR / "train_m2w_raw.jsonl"
SCORES_PATH = RAW_DIR / "scores_all_data.pkl"

SEED = 0
AUG_SEED = 1
VAL_SITE_FRACTION = 0.10
N_SHARDS = 11
BASE_URL = "https://huggingface.co/datasets/osunlp/Mind2Web/resolve/main/data/train/train_{}.json"


def stream_shard(i: int):
    url = BASE_URL.format(i)
    r = requests.get(url, stream=True)
    r.raise_for_status()
    try:
        for task in ijson.items(r.raw, "item"):
            yield task
    finally:
        r.close()


def load_rank_lookup():
    with open(SCORES_PATH, "rb") as f:
        d = pickle.load(f)
    return d["ranks"]


def random_negatives(neg_candidates, rng, k=9):
    pool = list(neg_candidates)
    rng.shuffle(pool)
    return pool[:k]


def shuffle_and_place(gold_render, neg_renders, key: str, seed: int):
    items = [gold_render] + neg_renders
    rng = random.Random(f"{seed}:{key}:shuffle")
    order = list(range(10))
    rng.shuffle(order)
    shuffled = [items[i] for i in order]
    gold_idx = order.index(0)
    for i, c in enumerate(shuffled):
        c["idx"] = i
    return shuffled, gold_idx


def render_negs(tree, cands, need=9):
    """Render candidates in order, skipping any not found in cleaned_html, until
    `need` are rendered or the pool runs out."""
    out = []
    for c in cands:
        r = render_candidate(tree, c)
        if r is not None:
            out.append(r)
        if len(out) >= need:
            break
    return out


def build_rows_for_step(task, step_idx, action, rank_map, counters):
    """Returns (base_row, aug_row) or (None, None) if the step is skipped."""
    pos = action["pos_candidates"]
    if not pos:
        counters["skip_no_pos"] += 1
        return None, None
    gold_cand = next((c for c in pos if c.get("is_original_target")), pos[0])

    neg_all = action["neg_candidates"]
    key = f"{task['annotation_id']}_{action['action_uid']}"

    rng0 = random.Random(f"{SEED}:{key}")
    top_negs, used_ranks = rank_negatives(neg_all, rank_map, rng0, k=9)
    counters["neg_by_rank" if used_ranks else "neg_by_fallback_random"] += 1
    if len(top_negs) < 9:
        counters["skip_insufficient_neg"] += 1
        return None, None

    rng1 = random.Random(f"{AUG_SEED}:{key}")
    aug_negs = random_negatives(neg_all, rng1, k=9)
    if len(aug_negs) < 9:
        counters["skip_insufficient_neg_aug"] += 1
        return None, None

    tree = lxml.html.fromstring(action["cleaned_html"])
    gold_render = render_candidate(tree, gold_cand)
    if gold_render is None:
        counters["skip_gold_not_found"] += 1
        return None, None

    # extra_pool backs both variants if a rendered candidate turns out to have
    # no matching node in cleaned_html (rare, matches eval/build_dataset.py).
    extra_pool = [c for c in neg_all if c not in top_negs and c not in aug_negs]

    def fill(primary, other_used):
        rendered = render_negs(tree, primary, need=9)
        pool = list(extra_pool)
        while len(rendered) < 9 and pool:
            r = render_candidate(tree, pool.pop(0))
            if r is not None:
                rendered.append(r)
        return rendered if len(rendered) == 9 else None

    base_neg_renders = fill(top_negs, aug_negs)
    if base_neg_renders is None:
        counters["skip_neg_not_found"] += 1
        return None, None
    aug_neg_renders = fill(aug_negs, top_negs)
    if aug_neg_renders is None:
        counters["skip_neg_not_found_aug"] += 1
        return None, None

    op = action["operation"]
    value = op.get("value") or None
    n_prev = 5
    previous_actions = task["action_reprs"][max(0, step_idx - n_prev):step_idx]

    base_cands, base_gold_idx = shuffle_and_place(dict(gold_render), base_neg_renders, key, SEED)
    aug_cands, aug_gold_idx = shuffle_and_place(dict(gold_render), aug_neg_renders, key, AUG_SEED)

    common = {
        "website": task["website"],
        "domain": task["domain"],
        "goal": task["confirmed_task"],
        "previous_actions": previous_actions,
        "operation": op["op"],
        "value": value,
        "_annotation_id": task["annotation_id"],
    }
    base_row = {"id": key, "split": None, "candidates": base_cands, "gold_idx": base_gold_idx, **common}
    aug_row = {"id": f"{key}_aug", "split": None, "candidates": aug_cands, "gold_idx": aug_gold_idx, **common}
    return base_row, aug_row


def phase_a():
    rank_lookup = load_rank_lookup()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    counters = Counter()
    n_tasks = 0
    n_steps_seen = 0
    n_written = 0
    t0 = time.time()
    with open(RAW_PATH, "w") as out_f:
        for i in range(N_SHARDS):
            shard_written = 0
            shard_tasks = 0
            for task in stream_shard(i):
                n_tasks += 1
                shard_tasks += 1
                for step_idx, action in enumerate(task["actions"]):
                    n_steps_seen += 1
                    key = f"{task['annotation_id']}_{action['action_uid']}"
                    rank_map = rank_lookup.get(key)
                    base_row, aug_row = build_rows_for_step(task, step_idx, action, rank_map, counters)
                    if base_row is None:
                        continue
                    out_f.write(json.dumps(base_row) + "\n")
                    out_f.write(json.dumps(aug_row) + "\n")
                    n_written += 2
            print(
                f"[train_{i}] tasks={shard_tasks} running_total_tasks={n_tasks} "
                f"rows={n_written} elapsed={time.time()-t0:.0f}s",
                file=sys.stderr,
            )
    print(
        f"[mind2web train] tasks={n_tasks} steps_seen={n_steps_seen} rows_written={n_written} "
        f"skips={dict(counters)} elapsed={time.time()-t0:.0f}s",
        file=sys.stderr,
    )
    return {"tasks": n_tasks, "steps_seen": n_steps_seen, "rows_written": n_written, "counters": dict(counters)}


def do_split():
    with open(RAW_PATH) as f:
        rows = [json.loads(line) for line in f]

    # Hold out whole WEBSITES, not tasks: the eval set is cross-website, and a same-site val
    # split scored 54-67% element while the same models scored 28-33% on unseen sites.
    websites = sorted({r["website"] for r in rows})
    rng = random.Random(f"{SEED}:val_holdout_sites")
    shuffled = list(websites)
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * VAL_SITE_FRACTION))
    val_sites = set(shuffled[:n_val])

    train_rows, val_rows = [], []
    for r in rows:
        r = dict(r)
        r.pop("_annotation_id")
        if r["website"] in val_sites:
            r["split"] = "val"
            val_rows.append(r)
        else:
            r["split"] = "train"
            train_rows.append(r)

    with open(DATA_DIR / "train_m2w.jsonl", "w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with open(DATA_DIR / "val.jsonl", "w") as f:
        for r in val_rows:
            f.write(json.dumps(r) + "\n")

    return train_rows, val_rows, set(websites) - val_sites, val_sites


def sanity_checks(rows, label):
    print(f"\n=== sanity: {label} ({len(rows)} rows) ===")
    ok = True

    ids = [r["id"] for r in rows]
    dup = len(ids) - len(set(ids))
    print(f"duplicate ids: {dup}")
    ok &= dup == 0

    bad_shape = 0
    for r in rows:
        idxs = [c["idx"] for c in r["candidates"]]
        if len(r["candidates"]) != 10 or sorted(idxs) != list(range(10)):
            bad_shape += 1
    print(f"rows without exactly 10 candidates w/ distinct idx 0-9: {bad_shape}")
    ok &= bad_shape == 0

    n = len(rows)
    gold_counts = Counter(r["gold_idx"] for r in rows)
    print("gold_idx distribution:", {k: gold_counts.get(k, 0) for k in range(10)})
    band_ok = all(0.06 <= gold_counts.get(k, 0) / n <= 0.14 for k in range(10)) if n else False
    print(f"gold_idx uniform-ish (6%-14% each): {band_ok}")
    ok &= band_ok

    non_empty = sum(1 for r in rows if r["candidates"][r["gold_idx"]]["text"].strip())
    frac = non_empty / n if n else 0
    print(f"gold text non-empty: {non_empty}/{n} ({frac:.1%}), threshold >=95%: {frac >= 0.95}")
    ok &= frac >= 0.95

    op_counts = Counter(r["operation"] for r in rows)
    print("operation distribution:", dict(op_counts))
    site_counts = Counter(r["website"] for r in rows)
    print(f"distinct websites: {len(site_counts)}")

    print(f"OVERALL PASS: {ok}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-a", action="store_true", help="only stream HF -> data/raw/train_m2w_raw.jsonl")
    ap.add_argument("--split", action="store_true", help="only re-split data/raw/train_m2w_raw.jsonl")
    args = ap.parse_args()

    do_phase_a = args.phase_a or not args.split
    do_do_split = args.split or not args.phase_a

    stats = {}
    if do_phase_a:
        stats = phase_a()

    if not do_do_split:
        return

    train_rows, val_rows, train_task_ids, val_task_ids = do_split()

    print("\n" + "=" * 60)
    print("BUILD SUMMARY")
    print("=" * 60)
    if stats:
        print(f"phase_a: {stats}")
    print(f"tasks: train={len(train_task_ids)} val={len(val_task_ids)} "
          f"(val fraction={len(val_task_ids)/(len(train_task_ids)+len(val_task_ids)):.3f})")
    overlap = train_task_ids & val_task_ids
    print(f"train/val task overlap: {len(overlap)} (must be 0)")

    train_ok = sanity_checks(train_rows, "train_m2w.jsonl")
    val_ok = sanity_checks(val_rows, "val.jsonl")

    if not (train_ok and val_ok and not overlap):
        print("\nSANITY CHECKS FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
