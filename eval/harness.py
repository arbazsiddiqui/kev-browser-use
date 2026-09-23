#!/usr/bin/env python3
"""Evaluation harness for kev-browser-use.

Runs an adapter over a dataset of steps (goal + previous actions + 10 candidate
elements -> which element, which operation) and scores it against the gold label.

Usage:
    python3 eval/harness.py run --adapter random --data data/eval_fake.jsonl --dataset-name fake
    python3 eval/harness.py run --adapter jev --data data/eval_fake.jsonl --dataset-name fake --limit 5
    python3 eval/harness.py table

Adapters live in eval/adapters/<name>.py and expose predict(row) -> {
    "element_probs": [10 floats summing to 1],
    "operation_probs": {"CLICK": p, "TYPE": p, "SELECT": p},
    "latency_ms": float,
}
"""
import argparse
import json
import math
import statistics
import sys
import time
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RESULTS_DIR = REPO_ROOT / "results"

sys.path.insert(0, str(HERE))
import importlib  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_REPO_ROOT = str(_Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:  # adapters import eval.render regardless of the cwd
    sys.path.insert(0, _REPO_ROOT)

OPERATIONS = ("CLICK", "TYPE", "SELECT")
ADAPTER_NAMES = (
    "random", "first", "lexical", "laya", "jev", "nanojev", "decider", "kev", "nimble", "semif",
)
LARGE_N = 200  # threshold above which the 5%-15% / 3pt sanity gates are enforced, not just noted


# --------------------------------------------------------------------------- data


def load_dataset(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cands = sorted(row["candidates"], key=lambda c: c["idx"])
            if len(cands) != 10:
                raise ValueError(f"row {row.get('id')!r} has {len(cands)} candidates, expected 10")
            row["candidates"] = cands
            rows.append(row)
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    return rows


def load_adapter(name):
    if name not in ADAPTER_NAMES:
        raise SystemExit(f"Unknown adapter {name!r}. Known: {ADAPTER_NAMES}")
    mod = importlib.import_module(f"adapters.{name}")
    return mod.predict


# ----------------------------------------------------------------------- running


def _predict_one(predict_fn, row):
    """Adapters never see the labels: gold_idx, operation and value are stripped first."""
    visible = {k: v for k, v in row.items() if k not in ("gold_idx", "operation", "value")}
    t0 = time.perf_counter()
    out = dict(predict_fn(visible))
    out.setdefault("latency_ms", (time.perf_counter() - t0) * 1000)
    return row["id"], out

def run_adapter(predict_fn, rows, concurrency=1):
    if concurrency <= 1:
        results = {}
        for row in rows:
            rid, out = _predict_one(predict_fn, row)
            results[rid] = out
    else:
        results = {}
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(_predict_one, predict_fn, row) for row in rows]
            for fut in as_completed(futs):
                rid, out = fut.result()
                results[rid] = out
    return [results[row["id"]] for row in rows]


# --------------------------------------------------------------------- metrics


def brier_score(element_probs, gold_idx):
    return sum((p - (1.0 if i == gold_idx else 0.0)) ** 2 for i, p in enumerate(element_probs))


def evaluate(rows, predictions):
    per_row = []
    for row, pred in zip(rows, predictions):
        element_probs = pred["element_probs"]
        gold_idx = row["gold_idx"]
        pred_idx = max(range(len(element_probs)), key=lambda i: element_probs[i])

        op_probs = pred["operation_probs"]
        pred_op = max(op_probs, key=op_probs.get)
        gold_op = row["operation"]

        element_correct = pred_idx == gold_idx
        operation_correct = pred_op == gold_op

        per_row.append({
            "id": row["id"],
            "split": row.get("split"),
            "pred_idx": pred_idx,
            "gold_idx": gold_idx,
            "pred_op": pred_op,
            "gold_op": gold_op,
            "element_correct": element_correct,
            "operation_correct": operation_correct,
            "step_success": element_correct and operation_correct,
            "brier": brier_score(element_probs, gold_idx),
            "top_prob": float(max(element_probs)) if element_probs else None,
            "latency_ms": pred.get("latency_ms", 0.0),
            "usage": pred.get("usage"),
        })
    return per_row


def _agg(subset):
    n = len(subset)
    if n == 0:
        return None
    lat = sorted(r["latency_ms"] for r in subset)
    return {
        "n": n,
        "element_acc": sum(r["element_correct"] for r in subset) / n,
        "operation_acc": sum(r["operation_correct"] for r in subset) / n,
        "step_success": sum(r["step_success"] for r in subset) / n,
        "brier": sum(r["brier"] for r in subset) / n,
        "latency_mean_ms": statistics.fmean(lat),
        "latency_p50_ms": statistics.median(lat),
    }


def aggregate(per_row):
    out = {"overall": _agg(per_row)}
    for s in sorted({r["split"] for r in per_row if r.get("split")}):
        out[s] = _agg([r for r in per_row if r.get("split") == s])
    return out


def sum_usage(per_row):
    total = {}
    for r in per_row:
        u = r.get("usage")
        if not u:
            continue
        for k, v in u.items():
            if isinstance(v, (int, float)):
                total[k] = total.get(k, 0) + v
    return total or None


# ------------------------------------------------------------------ sanity gates


def dump_suspect_rows(rows, per_row, k=5):
    by_id = {row["id"]: row for row in rows}
    print(f"  --- dumping {min(k, len(per_row))} rows for inspection ---")
    for r in per_row[:k]:
        row = by_id[r["id"]]
        print(f"  id={r['id']} goal={row['goal']!r}")
        print(f"    pred_idx={r['pred_idx']} gold_idx={r['gold_idx']} pred_op={r['pred_op']} gold_op={r['gold_op']}")
        for c in row["candidates"]:
            mark = " <-- GOLD" if c["idx"] == r["gold_idx"] else ""
            print(f"      [{c['idx']}] <{c.get('tag')}> {c.get('text', '')[:50]!r}{mark}")


def sanity_gates(adapter_name, dataset_name, rows, per_row, agg_overall, all_results):
    """Return [(gate_name, ok_or_None, message)]. ok=None means not enforced (e.g. small-n)."""
    gates = []
    n = agg_overall["n"]
    large = n >= LARGE_N

    if adapter_name == "random":
        acc = agg_overall["element_acc"]
        if large:
            ok = 0.05 <= acc <= 0.15
            gates.append(("random element acc in [5%,15%]", ok, f"acc={acc:.3f} n={n}"))
        else:
            gates.append(("random element acc in [5%,15%]", None, f"acc={acc:.3f} n={n} (small-n, not enforced)"))

    if adapter_name == "first":
        actual_rate = sum(1 for r in rows if r["gold_idx"] == 0) / len(rows)
        acc = agg_overall["element_acc"]
        ok = abs(acc - actual_rate) < 1e-9
        gates.append(("first element acc == gold_idx==0 rate", ok, f"acc={acc:.3f} actual_rate={actual_rate:.3f}"))

    if adapter_name == "lexical":
        random_res = all_results.get(("random", dataset_name))
        if random_res:
            r_acc = random_res["metrics"]["overall"]["element_acc"]
            l_acc = agg_overall["element_acc"]
            gates.append(("lexical beats random", l_acc > r_acc, f"lexical={l_acc:.3f} random={r_acc:.3f}"))
        else:
            gates.append(("lexical beats random", None, "no random results for this dataset yet"))

    if adapter_name in ("laya", "jev", "nanojev", "decider", "kev", "nimble", "semif"):
        random_res = all_results.get(("random", dataset_name))
        if random_res and large:
            r_acc = random_res["metrics"]["overall"]["element_acc"]
            m_acc = agg_overall["element_acc"]
            suspect = abs(m_acc - r_acc) <= 0.03
            gates.append((f"{adapter_name} beats random by >3pts", not suspect, f"{adapter_name}={m_acc:.3f} random={r_acc:.3f}"))
            if suspect:
                print(f"SUSPECT HARNESS: {adapter_name} is within 3 points of random on {dataset_name} (n={n})")
                dump_suspect_rows(rows, per_row, 5)
        elif random_res:
            gates.append((f"{adapter_name} vs random", None, f"n={n} (small-n, not enforced)"))
        else:
            gates.append((f"{adapter_name} vs random", None, "no random results for this dataset yet"))

    return gates


# ------------------------------------------------------------------------- I/O


def results_path(adapter, dataset_name):
    return RESULTS_DIR / f"{adapter}_{dataset_name}.json"


def load_all_results():
    idx = {}
    if not RESULTS_DIR.exists():
        return idx
    for p in RESULTS_DIR.glob("*.json"):
        if p.name == "table.md":
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        a, d = data.get("adapter"), data.get("dataset_name")
        if a and d:
            idx[(a, d)] = data
    return idx


# ------------------------------------------------------------------------- CLI


def cmd_run(args):
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path
    rows = load_dataset(data_path)
    if args.limit:
        rows = rows[: args.limit]

    predict_fn = load_adapter(args.adapter)
    default_concurrency = 4 if args.adapter == "jev" else 1
    concurrency = args.concurrency if args.concurrency is not None else default_concurrency

    t_start = time.time()
    predictions = run_adapter(predict_fn, rows, concurrency=concurrency)
    wall_s = time.time() - t_start

    per_row = evaluate(rows, predictions)
    agg = aggregate(per_row)
    usage_total = sum_usage(per_row)

    out_path = Path(args.out) if args.out else results_path(args.adapter, args.dataset_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "adapter": args.adapter,
        "dataset_name": args.dataset_name,
        "dataset_path": str(args.data),
        "n": len(rows),
        "render_mode": os.environ.get("EVAL_RENDER", "default"),
        "phrasing": os.environ.get("EVAL_PHRASING", "a"),
        "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_s": round(wall_s, 2),
        "metrics": agg,
        "usage_total": usage_total,
        "per_row": per_row,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    m = agg["overall"]
    print(
        f"[{args.adapter}/{args.dataset_name}] n={m['n']} element_acc={m['element_acc']:.3f} "
        f"operation_acc={m['operation_acc']:.3f} step_success={m['step_success']:.3f} "
        f"brier={m['brier']:.3f} latency_mean_ms={m['latency_mean_ms']:.2f} "
        f"latency_p50_ms={m['latency_p50_ms']:.2f} wall_s={wall_s:.1f}"
    )
    if usage_total:
        print(f"[{args.adapter}] usage total: {usage_total}")
    print(f"wrote {out_path}")

    all_results = load_all_results()
    all_results[(args.adapter, args.dataset_name)] = payload
    gates = sanity_gates(args.adapter, args.dataset_name, rows, per_row, m, all_results)
    print("--- sanity gates ---")
    for name, ok, msg in gates:
        status = "PASS" if ok is True else ("FAIL" if ok is False else "SKIP")
        print(f"  [{status}] {name}: {msg}")


def cmd_table(args):
    idx = load_all_results()
    if not idx:
        print("no results in results/ yet")
        return
    datasets = sorted({d for _, d in idx})
    lines = ["# kev-browser-use eval results", ""]
    for d in datasets:
        lines.append(f"## dataset: {d}")
        lines.append("")
        lines.append("| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for a in ADAPTER_NAMES:
            r = idx.get((a, d))
            if not r:
                continue
            m = r["metrics"]["overall"]
            lines.append(
                f"| {a} | {m['element_acc']:.3f} | {m['operation_acc']:.3f} | {m['step_success']:.3f} | "
                f"{m['brier']:.3f} | {m['latency_mean_ms']:.2f} | {m['latency_p50_ms']:.2f} | {m['n']} |"
            )
        lines.append("")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "table.md"
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run one adapter over one dataset")
    run_p.add_argument("--adapter", required=True, choices=ADAPTER_NAMES)
    run_p.add_argument("--data", required=True, help="path to a JSONL dataset")
    run_p.add_argument("--dataset-name", required=True, help="short label, e.g. fake/smoke/1k")
    run_p.add_argument("--limit", type=int, default=None)
    run_p.add_argument("--concurrency", type=int, default=None)
    run_p.add_argument("--out", default=None)
    run_p.set_defaults(func=cmd_run)

    table_p = sub.add_parser("table", help="aggregate results/*.json into results/table.md")
    table_p.set_defaults(func=cmd_table)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
