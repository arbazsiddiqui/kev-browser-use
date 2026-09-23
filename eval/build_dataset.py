"""Build the evaluation set from Mind2Web's OFFICIAL test splits.

Source: Hugging Face dataset osunlp/Mind2Web (CC BY 4.0), the same test.zip that
the project's GitHub repo (OSU-NLP-Group/Mind2Web) points to, password
"mind2web". Only test_website (-> split "cross_website") and test_domain
(-> split "cross_domain") are used; test_task ("cross_task", same websites as
train) is out of scope for this project.

Each Mind2Web step becomes one eval row: goal + previous actions + 10 candidate
elements (1 gold + 9 hardest negatives, ranked by scores_all_data.pkl, the
candidate-ranker scores the dataset authors publish alongside the data) ->
which candidate, which operation.

Usage:
    python3 eval/build_dataset.py            # full build: phase A + smoke + 1k
    python3 eval/build_dataset.py --phase-a  # only rebuild data/raw/all_*.jsonl
    python3 eval/build_dataset.py --sample   # only resample smoke/1k from data/raw/all_*.jsonl
"""

import argparse
import json
import pickle
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import ijson
import lxml.html

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ZIP_PATH = RAW_DIR / "test.zip"
SCORES_PATH = RAW_DIR / "scores_all_data.pkl"
PASSWORD = "mind2web"
SEED = 0

SPLITS = {
    "cross_website": "test_website",
    "cross_domain": "test_domain",
}
SHARD_COUNTS = {
    "cross_website": 2,
    "cross_domain": 10,
}

ATTR_KEYS = ["id", "aria-label", "placeholder", "name", "type"]
WS_RE = re.compile(r"\s+")


def list_shard_members(prefix: str, count: int):
    return [f"{prefix}/{prefix}_{i}.json" for i in range(count)]


def stream_tasks(member: str):
    """Yield task dicts from one shard member of test.zip, decrypted+streamed."""
    proc = subprocess.Popen(
        ["unzip", "-P", PASSWORD, "-p", str(ZIP_PATH), member],
        stdout=subprocess.PIPE,
    )
    try:
        for task in ijson.items(proc.stdout, "item"):
            yield task
    finally:
        proc.stdout.close()
        proc.wait()


def compact_attrs(el) -> dict:
    out = {}
    for key in ATTR_KEYS:
        val = el.get(key)
        if val:
            out[key] = val.strip()[:80]
    href = el.get("href")
    if href:
        try:
            host = urlparse(href).netloc
        except ValueError:
            host = ""
        if host:
            out["href"] = host
    return out


TEXT_FALLBACK_ATTRS = ["aria-label", "placeholder", "value", "title", "alt"]
CLASS_TOKEN_RE = re.compile(r"[-_]+")


def _humanize_class(cls: str) -> str:
    tokens = [t for t in CLASS_TOKEN_RE.sub(" ", cls).split() if len(t) > 2 and not t.isdigit()]
    return " ".join(tokens[:4])


def visible_text(el, limit=120) -> str:
    txt = WS_RE.sub(" ", " ".join(el.itertext())).strip()
    if not txt:
        # Elements like icon buttons, date inputs or search boxes often carry
        # no text node; their accessible label is the closest thing a user
        # sees, so fall back to it before giving up.
        for attr in TEXT_FALLBACK_ATTRS:
            val = el.get(attr)
            if val and val.strip():
                txt = val.strip()
                break
    if not txt:
        # Bare icon controls: no text anywhere and no accessible label. Two
        # last-resort, still-deterministic signals: an ancestor's label, or
        # the element's own CSS class name turned into words.
        ancestor = el.getparent()
        depth = 0
        while ancestor is not None and depth < 2 and not txt:
            for attr in ("aria-label", "title"):
                val = ancestor.get(attr)
                if val and val.strip():
                    txt = val.strip()
                    break
            ancestor = ancestor.getparent()
            depth += 1
    if not txt:
        cls = el.get("class")
        if cls:
            txt = _humanize_class(cls)
    return txt[:limit]


def find_by_backend_id(tree, backend_node_id: str):
    els = tree.xpath(f'//*[@backend_node_id="{backend_node_id}"]')
    return els[0] if els else None


def render_candidate(tree, cand: dict):
    el = find_by_backend_id(tree, cand["backend_node_id"])
    if el is None:
        return None
    return {
        "tag": el.tag,
        "role": el.get("role") or None,
        "text": visible_text(el),
        "attrs": compact_attrs(el),
    }


def rank_negatives(neg_candidates, rank_map, rng, k=9):
    """Pick the k hardest negatives. Uses candidate-ranker ranks when available
    (lower rank = more confusable with the gold element = harder), else falls
    back to a seeded random sample."""
    if rank_map:
        BIG = 10**9
        ordered = sorted(
            neg_candidates,
            key=lambda c: rank_map.get(c["backend_node_id"], BIG),
        )
        return ordered[:k], True
    else:
        pool = list(neg_candidates)
        rng.shuffle(pool)
        return pool[:k], False


def build_row(task, step_idx, action, rank_map, counters):
    pos = action["pos_candidates"]
    if not pos:
        counters["skip_no_pos"] += 1
        return None
    gold_cand = next((c for c in pos if c.get("is_original_target")), pos[0])

    neg_all = action["neg_candidates"]
    key = f"{task['annotation_id']}_{action['action_uid']}"
    rng = random.Random(f"{SEED}:{key}")
    top_negs, used_ranks = rank_negatives(neg_all, rank_map, rng, k=9)
    if used_ranks:
        counters["neg_by_rank"] += 1
    else:
        counters["neg_by_fallback_random"] += 1
    if len(top_negs) < 9:
        counters["skip_insufficient_neg"] += 1
        return None

    tree = lxml.html.fromstring(action["cleaned_html"])
    gold_render = render_candidate(tree, gold_cand)
    if gold_render is None:
        counters["skip_gold_not_found"] += 1
        return None

    neg_renders = []
    extra_pool = [c for c in neg_all if c not in top_negs]
    candidates_needed = list(top_negs)
    idx = 0
    while len(neg_renders) < 9:
        if idx >= len(candidates_needed):
            if not extra_pool:
                counters["skip_neg_not_found"] += 1
                return None
            candidates_needed.append(extra_pool.pop(0))
        r = render_candidate(tree, candidates_needed[idx])
        idx += 1
        if r is not None:
            neg_renders.append(r)
        else:
            counters["neg_element_not_found_repaired"] += 1

    items = [gold_render] + neg_renders
    rng_shuffle = random.Random(f"{SEED}:{key}:shuffle")
    order = list(range(10))
    rng_shuffle.shuffle(order)
    shuffled = [items[i] for i in order]
    gold_idx = order.index(0)
    for i, c in enumerate(shuffled):
        c["idx"] = i

    op = action["operation"]
    value = op.get("value") or None

    n_prev = 5
    previous_actions = task["action_reprs"][max(0, step_idx - n_prev):step_idx]

    return {
        "id": key,
        "split": None,  # filled by caller
        "website": task["website"],
        "domain": task["domain"],
        "goal": task["confirmed_task"],
        "previous_actions": previous_actions,
        "candidates": shuffled,
        "gold_idx": gold_idx,
        "operation": op["op"],
        "value": value,
    }


def phase_a(split_name: str, rank_lookup: dict):
    prefix = SPLITS[split_name]
    n_shards = SHARD_COUNTS[split_name]
    out_path = RAW_DIR / f"all_{split_name}.jsonl"
    counters = Counter()
    n_tasks = 0
    n_steps_seen = 0
    n_written = 0
    with open(out_path, "w") as out_f:
        for member in list_shard_members(prefix, n_shards):
            print(f"[{split_name}] streaming {member} ...", file=sys.stderr)
            shard_written = 0
            for task in stream_tasks(member):
                n_tasks += 1
                for step_idx, action in enumerate(task["actions"]):
                    n_steps_seen += 1
                    key = f"{task['annotation_id']}_{action['action_uid']}"
                    rank_map = rank_lookup.get(key)
                    row = build_row(task, step_idx, action, rank_map, counters)
                    if row is None:
                        continue
                    row["split"] = split_name
                    out_f.write(json.dumps(row) + "\n")
                    n_written += 1
                    shard_written += 1
            print(f"[{split_name}] {member}: +{shard_written} rows (running total {n_written})", file=sys.stderr)
    print(
        f"[{split_name}] tasks={n_tasks} steps_seen={n_steps_seen} written={n_written} "
        f"skips={dict(counters)}",
        file=sys.stderr,
    )
    return {
        "tasks": n_tasks,
        "steps_seen": n_steps_seen,
        "written": n_written,
        "counters": dict(counters),
    }


def load_rank_lookup():
    with open(SCORES_PATH, "rb") as f:
        d = pickle.load(f)
    return d["ranks"]


def sample_smoke(rows_by_split, n_per_split=25):
    """Round-robin across gold_idx buckets (0-9) so the 50-row smoke set
    stays comfortably inside the 6%-14% uniformity band despite its small N."""
    out = []
    for split, rows in rows_by_split.items():
        by_bucket = defaultdict(list)
        for r in rows:
            by_bucket[r["gold_idx"]].append(r)
        rng = random.Random(f"{SEED}:smoke:{split}")
        for b in by_bucket:
            rng.shuffle(by_bucket[b])
        buckets = list(range(10))
        rng.shuffle(buckets)
        picked = []
        cursor = {b: 0 for b in buckets}
        while len(picked) < n_per_split:
            progressed = False
            for b in buckets:
                if len(picked) >= n_per_split:
                    break
                i = cursor[b]
                if i < len(by_bucket[b]):
                    picked.append(by_bucket[b][i])
                    cursor[b] = i + 1
                    progressed = True
            if not progressed:
                break
        out.extend(picked)
    return out


def min_feasible_cap(counts, target):
    """Smallest per-group cap C such that sum(min(c, C) for c in counts) >=
    target. Used when the ideal 5%-of-target cap can't reach the target
    because a split simply doesn't have enough distinct websites."""
    if sum(counts) < target:
        return max(counts) if counts else 0
    lo, hi = 1, max(counts)
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(min(c, mid) for c in counts) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def sample_diverse(rows, target, seed_tag):
    """Round-robin across websites to maximize distinct-website count, capping
    each website at 5% of the target when that's enough to reach the target,
    otherwise at the smallest cap that still reaches it (see min_feasible_cap;
    Mind2Web's cross_website test only spans 10 websites, so 5% is infeasible
    there)."""
    by_site = defaultdict(list)
    for r in rows:
        by_site[r["website"]].append(r)

    ideal_cap = max(1, int(target * 0.05))
    cap = ideal_cap
    if sum(min(len(v), ideal_cap) for v in by_site.values()) < target:
        cap = min_feasible_cap([len(v) for v in by_site.values()], target)

    rng = random.Random(f"{SEED}:{seed_tag}:sites")
    sites = list(by_site.keys())
    rng.shuffle(sites)
    for s in sites:
        rng2 = random.Random(f"{SEED}:{seed_tag}:{s}")
        rng2.shuffle(by_site[s])

    picked = []
    counts = Counter()
    cursor = {s: 0 for s in sites}
    progressed = True
    while len(picked) < target and progressed:
        progressed = False
        for s in sites:
            if len(picked) >= target:
                break
            if counts[s] >= cap:
                continue
            i = cursor[s]
            if i >= len(by_site[s]):
                continue
            picked.append(by_site[s][i])
            cursor[s] = i + 1
            counts[s] += 1
            progressed = True
    return picked, cap


def sanity_checks(rows, label):
    print(f"\n=== sanity: {label} ({len(rows)} rows) ===")
    ok = True

    ids = [r["id"] for r in rows]
    dup = len(ids) - len(set(ids))
    print(f"duplicate ids: {dup}")
    ok &= dup == 0

    bad_cand_shape = 0
    for r in rows:
        idxs = [c["idx"] for c in r["candidates"]]
        if len(r["candidates"]) != 10 or sorted(idxs) != list(range(10)):
            bad_cand_shape += 1
    print(f"rows without exactly 10 candidates w/ distinct idx 0-9: {bad_cand_shape}")
    ok &= bad_cand_shape == 0

    gold_counts = Counter(r["gold_idx"] for r in rows)
    n = len(rows)
    print("gold_idx distribution:", {k: gold_counts.get(k, 0) for k in range(10)})
    band_ok = True
    for k in range(10):
        frac = gold_counts.get(k, 0) / n if n else 0
        if not (0.06 <= frac <= 0.14):
            band_ok = False
    print(f"gold_idx uniform-ish (6%-14% each): {band_ok}")
    ok &= band_ok

    non_empty = sum(
        1 for r in rows if r["candidates"][r["gold_idx"]]["text"].strip()
    )
    frac_non_empty = non_empty / n if n else 0
    print(f"gold text non-empty: {non_empty}/{n} ({frac_non_empty:.1%}), threshold >=95%: {frac_non_empty >= 0.95}")
    ok &= frac_non_empty >= 0.95

    op_counts = Counter(r["operation"] for r in rows)
    print("operation distribution:", dict(op_counts))

    site_counts = Counter(r["website"] for r in rows)
    print(f"distinct websites: {len(site_counts)}, top 5: {site_counts.most_common(5)}")

    print(f"OVERALL PASS: {ok}")
    return ok


def print_examples(rows, n=5):
    print(f"\n=== {n} example rows ===")
    rng = random.Random(f"{SEED}:examples")
    sample = rng.sample(rows, min(n, len(rows)))
    for r in sample:
        gold = r["candidates"][r["gold_idx"]]
        print(
            f"- id={r['id'][:8]}.. split={r['split']} site={r['website']} op={r['operation']} "
            f"value={r['value']!r} gold_idx={r['gold_idx']} "
            f"gold=({gold['tag']}, text={gold['text'][:40]!r}, attrs={gold['attrs']}) "
            f"goal={r['goal'][:60]!r}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-a", action="store_true", help="only rebuild data/raw/all_*.jsonl")
    ap.add_argument("--sample", action="store_true", help="only resample smoke/1k from data/raw/all_*.jsonl")
    args = ap.parse_args()

    do_phase_a = args.phase_a or not args.sample
    do_sample = args.sample or not args.phase_a

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    phase_a_stats = {}

    if do_phase_a:
        rank_lookup = load_rank_lookup()
        for split in SPLITS:
            phase_a_stats[split] = phase_a(split, rank_lookup)

    if not do_sample:
        return

    rows_by_split = {}
    for split in SPLITS:
        path = RAW_DIR / f"all_{split}.jsonl"
        with open(path) as f:
            rows_by_split[split] = [json.loads(line) for line in f]
        print(f"loaded {len(rows_by_split[split])} candidate rows for {split}", file=sys.stderr)

    smoke_rows = sample_smoke(rows_by_split, n_per_split=25)
    with open(DATA_DIR / "eval_smoke.jsonl", "w") as f:
        for r in smoke_rows:
            f.write(json.dumps(r) + "\n")

    thousand_rows = []
    caps = {}
    for split, rows in rows_by_split.items():
        picked, cap = sample_diverse(rows, target=500, seed_tag="eval1k")
        caps[split] = cap
        thousand_rows.extend(picked)
    with open(DATA_DIR / "eval_1k.jsonl", "w") as f:
        for r in thousand_rows:
            f.write(json.dumps(r) + "\n")

    print("\n" + "=" * 60)
    print("BUILD SUMMARY")
    print("=" * 60)
    for split, stats in phase_a_stats.items():
        print(f"{split}: {stats}")
    for split in SPLITS:
        print(f"eval_1k per-website cap for {split}: {caps.get(split)}")

    smoke_ok = sanity_checks(smoke_rows, "eval_smoke.jsonl")
    thousand_ok = sanity_checks(thousand_rows, "eval_1k.jsonl")
    print_examples(thousand_rows, n=5)

    if not (smoke_ok and thousand_ok):
        print("\nSANITY CHECKS FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
