"""Build a training slice from WebChain (arXiv 2603.05295, CC BY 4.0,
webagentlab/WebChain on Hugging Face).

WebChain logs each step as a "Triple Alignment" of visual, structural and action
data: per the paper, every step carries the target element's CSS selector, DOM
path and bounding box, plus (via `html_dom_url`/`ax_tree_url`) a full pre-action
DOM/AX-tree snapshot hosted at data.imean.tech. That snapshot is exactly the
structural channel our eval schema needs: fetch the DOM snapshot, locate the
target via its CSS selector (verified unique per step), and every other
interactive element on that same page becomes a candidate, rendered with the
same visible_text/compact_attrs logic eval/build_dataset.py uses for Mind2Web.

The dataset's own metadata (actions.parquet / traces.parquet, ~48 parts) is
small and downloaded whole; the DOM snapshots are not part of the release and
must be fetched one HTTP call per step, which is the throughput bottleneck.
Calibrated at ~11 rows/sec with 10 concurrent fetches, so this script runs to a
wall-clock deadline (default 30 min) rather than a fixed row count, and writes
what it produced. See train/data/README.md for the negative-sampling method
(no ranker exists for WebChain, so negatives favor goal/text overlap) and the
verdict on why the structural channel supports this (it does).

Usage:
    python3 train/data/webchain.py                  # 30-minute budget, 10 workers
    python3 train/data/webchain.py --minutes 5       # smoke test
"""

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import lxml.html
import pandas as pd
import requests
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.build_dataset import visible_text, compact_attrs  # noqa: E402
from train.data.mind2web_train import shuffle_and_place  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "raw" / "webchain_cache"
OUT_PATH = DATA_DIR / "train_webchain.jsonl"

REPO_ID = "webagentlab/WebChain"
N_PARTS = 48
SEED = 0

OP_MAP = {"click": "CLICK", "type": "TYPE", "select": "SELECT"}
INTERACTIVE_CSS = "a, button, input, select, option, textarea, [role]"
WORD_RE = re.compile(r"[a-z0-9]+")


def part_files(part: int):
    prefix = f"data/seed_sft/parts/part_{part:02d}/metadata"
    return f"{prefix}/actions.parquet", f"{prefix}/traces.parquet"


def load_part(part: int):
    actions_fn, traces_fn = part_files(part)
    actions_path = hf_hub_download(REPO_ID, repo_type="dataset", filename=actions_fn, local_dir=CACHE_DIR)
    traces_path = hf_hub_download(REPO_ID, repo_type="dataset", filename=traces_fn, local_dir=CACHE_DIR)
    actions = pd.read_parquet(actions_path)
    traces = pd.read_parquet(traces_path)
    return actions, traces


def eligible_rows(actions: pd.DataFrame):
    m = (
        actions["action_type"].isin(OP_MAP)
        & actions["included_in_sft"].fillna(False)
        & (actions["selector"].fillna("") != "")
        & (actions["html_dom_url"].fillna("") != "")
    )
    return actions[m]


def build_prev_actions(actions: pd.DataFrame):
    """Lightweight previous-action reprs built from fields already in the
    metadata table (no extra HTTP calls), so this stays synchronous and the
    threaded fetch pool below only ever fetches the CURRENT step's DOM."""
    prev_by_key = {}
    for trace_uid, g in actions.groupby("trace_uid"):
        g = g.sort_values("source_step_index")
        history = []
        for _, row in g.iterrows():
            key = (trace_uid, row["source_step_index"])
            prev_by_key[key] = list(history[-5:])
            op = OP_MAP.get(row["action_type"], row["action_type"])
            label = row.get("input_text") or ""
            if label in ("", "no input text"):
                path = row.get("dom_path") or ""
                label = path.split(">")[-1] if path else ""
            history.append(f"[{row['action_type']}] {label[:40]} -> {op}")
    return prev_by_key


def token_overlap_score(goal: str, cand: dict) -> int:
    goal_tokens = set(WORD_RE.findall(goal.lower()))
    cand_text = " ".join([cand["text"]] + [str(v) for v in cand["attrs"].values()])
    cand_tokens = set(WORD_RE.findall(cand_text.lower()))
    return len(goal_tokens & cand_tokens)


def pick_negatives(goal: str, pool: list, key: str, k=9):
    rng = random.Random(f"{SEED}:{key}:negs")
    scored = [(token_overlap_score(goal, c), rng.random(), c) for c in pool]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [c for _, _, c in scored[:k]]


def process_row(row, goal, website, domain, prev_actions, session):
    key = f"webchain_{row['trace_uid']}_{row['source_step_id']}"
    resp = session.get(row["html_dom_url"], timeout=20)
    resp.raise_for_status()
    # bytes, not .text: requests' charset detection (chardet) cost 1.2 s per 2 MB snapshot
    tree = lxml.html.fromstring(resp.content)

    targets = tree.cssselect(row["selector"])
    if len(targets) != 1:
        return None, "target_selector_no_unique_match"
    target_el = targets[0]

    pool_els = [el for el in tree.cssselect(INTERACTIVE_CSS) if el is not target_el]
    if len(pool_els) < 9:
        return None, "insufficient_candidates"

    gold_render = {
        "tag": target_el.tag,
        "role": target_el.get("role") or None,
        "text": visible_text(target_el),
        "attrs": compact_attrs(target_el),
    }
    pool_rendered = [
        {
            "tag": el.tag,
            "role": el.get("role") or None,
            "text": visible_text(el),
            "attrs": compact_attrs(el),
        }
        for el in pool_els
    ]

    negs = pick_negatives(goal, pool_rendered, key, k=9)
    if len(negs) < 9:
        return None, "insufficient_candidates"

    candidates, gold_idx = shuffle_and_place(gold_render, negs, key, SEED)

    op = OP_MAP[row["action_type"]]
    raw_value = row.get("input_text") or ""
    value = None if raw_value in ("", "no input text") else raw_value.strip()

    return {
        "id": key,
        "split": "train",
        "website": website,
        "domain": domain,
        "goal": goal,
        "previous_actions": prev_actions,
        "candidates": candidates,
        "gold_idx": gold_idx,
        "operation": op,
        "value": value,
    }, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-parts", type=int, default=N_PARTS)
    ap.add_argument("--start-part", type=int, default=0, help="first metadata part to process (resume)")
    ap.add_argument("--out", default=str(OUT_PATH), help="output jsonl (default data/train_webchain.jsonl)")
    args = ap.parse_args()

    deadline = time.time() + args.minutes * 60
    session = requests.Session()
    counters = Counter()
    n_written = 0
    websites = set()
    t0 = time.time()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as out_f:
        for part in range(args.start_part, min(args.max_parts, N_PARTS)):
            if time.time() >= deadline:
                print(f"[webchain] deadline reached before part {part}", file=sys.stderr)
                break
            actions, traces = load_part(part)
            trace_info = traces.set_index("uid")[["query", "web_type", "primary_host"]].to_dict("index")
            elig = eligible_rows(actions)
            counters["seen_total"] += len(actions)
            counters["eligible_total"] += len(elig)
            prev_by_key = build_prev_actions(actions)

            jobs = []
            for _, row in elig.iterrows():
                info = trace_info.get(row["trace_uid"])
                if not info or not info.get("query"):
                    counters["skip_no_goal"] += 1
                    continue
                goal = info["query"]
                website = (info.get("primary_host") or row.get("host") or "").removeprefix("www.")
                domain = info.get("web_type")
                prev = prev_by_key.get((row["trace_uid"], row["source_step_index"]), [])
                jobs.append((row, goal, website, domain, prev))

            print(f"[webchain] part {part}: {len(jobs)} eligible jobs queued, "
                  f"elapsed={time.time()-t0:.0f}s written_so_far={n_written}", file=sys.stderr)

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {
                    ex.submit(process_row, row, goal, website, domain, prev, session): row["source_step_id"]
                    for row, goal, website, domain, prev in jobs
                }
                for i, fut in enumerate(as_completed(futs)):
                    try:
                        result, skip_reason = fut.result()
                    except Exception as e:
                        result, skip_reason = None, f"exception:{type(e).__name__}"
                    if result is None:
                        counters[skip_reason or "unknown_skip"] += 1
                    else:
                        out_f.write(json.dumps(result) + "\n")
                        n_written += 1
                        websites.add(result["website"])
                    if i % 200 == 0:
                        elapsed = time.time() - t0
                        print(f"[webchain] part {part} {i}/{len(jobs)} elapsed={elapsed:.0f}s "
                              f"written={n_written} rate={n_written/max(elapsed,1):.1f}/s", file=sys.stderr)
                    if time.time() >= deadline:
                        print(f"[webchain] deadline reached mid-part {part} ({i}/{len(jobs)})", file=sys.stderr)
                        for f2 in futs:
                            f2.cancel()
                        break
            if time.time() >= deadline:
                break

    elapsed = time.time() - t0
    print(
        f"\n[webchain] DONE rows_written={n_written} distinct_websites={len(websites)} "
        f"elapsed={elapsed:.0f}s counters={dict(counters)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
