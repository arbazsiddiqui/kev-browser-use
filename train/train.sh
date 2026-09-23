#!/usr/bin/env bash
# Trains kev-0.6b-browser-use and scores it on the 1,000-step eval. Needs one 24 GB GPU and a
# checkout of Kev (https://github.com/jaredpalmer/kev) with `uv sync` done; point KEV_REPO_DIR at it.
#
#   python3 eval/build_dataset.py                        # data/eval_1k.jsonl
#   python3 train/data/mind2web_train.py                 # data/train_m2w.jsonl, data/val.jsonl
#   python3 train/data/webchain.py --max-parts 6         # data/train_webchain.jsonl
#   KEV_REPO_DIR=../kev bash train/train.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
say() { echo "[train] $(date -u +%T) $*"; }
KEV_REPO_DIR=$(cd "${KEV_REPO_DIR:-vendor/kev}" && pwd)
KEV_PY=$KEV_REPO_DIR/.venv/bin/python
export PYTHONPATH=$KEV_REPO_DIR
OUT=train/runs/kev-0.6b-browser-use
mkdir -p train/runs

say "1/4 mixture (Mind2Web twice + WebChain) in Kev's format"
python3 - <<'PYEOF'
import random
m2w = [l for l in open("data/train_m2w.jsonl") if l.strip()]
wc = [l for l in open("data/train_webchain.jsonl") if l.strip()]
rows = m2w * 2 + wc
random.Random(0).shuffle(rows)
open("data/train_mix.jsonl", "w").writelines(rows)
print(f"train_mix rows: {len(rows)}")
PYEOF
python3 train/data/to_kev_format.py --in data/train_mix.jsonl --out train/data/kev_train_mix.jsonl
python3 train/data/to_kev_format.py --in data/val.jsonl --out train/data/kev_val.jsonl

say "2/4 kev.train, 2 epochs, batch 8 -> $OUT"
rm -rf "$OUT"
"$KEV_PY" -m kev.train --data train/data/kev_train_mix.jsonl \
  --base Qwen/Qwen3-0.6B-Base --init_from jaredpalmer/kev-0.6b --lora 16 --head_dim 256 \
  --epochs 2 --lr 2e-5 --batch 8 --accum 1 --dtype bf16 --device cuda --p_none_pair 0 --seed 0 \
  --out "$OUT" > "$OUT.train.log" 2>&1
say "train exit=$? $(tail -1 "$OUT.train.log" | cut -c1-160)"

say "3/4 validation benchmark"
"$KEV_PY" -m kev.benchmark --run "$OUT" --data train/data/kev_val.jsonl --out "$OUT-val-benchmark" \
  --device cuda > "$OUT.bench.log" 2>&1
say "benchmark exit=$?"

say "4/4 1,000-step eval"
KEV_REPO_DIR=$KEV_REPO_DIR KEV_RUN=$PWD/$OUT KEV_MERGE=0 KEV_TIMEOUT_S=300 \
  python3 eval/harness.py run --adapter kev --data data/eval_1k.jsonl --dataset-name 1k > "$OUT.eval.log" 2>&1
grep -E "^\[kev/" "$OUT.eval.log"
python3 eval/harness.py table > /dev/null
say "done"
