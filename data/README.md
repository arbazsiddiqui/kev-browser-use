# Eval set

Built by `eval/build_dataset.py`. Not committed (see `.gitignore`); regenerate with:

```
python3 eval/build_dataset.py            # full build: raw pass + eval_smoke + eval_1k
python3 eval/build_dataset.py --phase-a  # only rebuild data/raw/all_*.jsonl from Mind2Web
python3 eval/build_dataset.py --sample   # only resample eval_smoke/eval_1k from data/raw/
```

## Source

[`osunlp/Mind2Web`](https://huggingface.co/datasets/osunlp/Mind2Web) on Hugging Face,
CC BY 4.0. Uses the dataset's **official test.zip** (password `mind2web`, per the dataset
card and the [Mind2Web GitHub repo](https://github.com/OSU-NLP-Group/Mind2Web)), the only
place the test splits are published; the HF `datasets` viewer only exposes `train`. Two of
its three splits are in scope:

- `test_website` -> our `cross_website` (websites unseen during Mind2Web training)
- `test_domain` -> our `cross_domain` (entire domains unseen during training)
- `test_task` (cross-task, same websites as train) is out of scope for this project.

Candidate-ranking scores come from `scores_all_data.pkl` (published alongside `test.zip` in
the same HF repo), the DeBERTa-v3 candidate ranker's per-element ranks from the Mind2Web
paper, keyed by `{annotation_id}_{action_uid} -> {backend_node_id: rank}`.

## Schema

One row per Mind2Web action step:

```
id                 "{annotation_id}_{action_uid}", globally unique
split              "cross_website" | "cross_domain"
website, domain    from the source task
goal               confirmed_task (the crowdworker's task description)
previous_actions   up to 5 prior action_reprs strings (human-readable, Mind2Web's own repr)
candidates         exactly 10 items, shuffled, {idx, tag, role, text, attrs}
gold_idx           index (0-9) of the correct candidate
operation          "CLICK" | "TYPE" | "SELECT"
value              typed text / selected option, else null
```

Each candidate is rendered by locating its `backend_node_id` in the step's `cleaned_html`
(via lxml) and extracting:
- `tag`: element tag name
- `role`: `role` attribute, else `null`
- `text`: visible text (`itertext()`, whitespace-collapsed), capped at 120 chars. When an
  element has no text node (icon buttons, bare inputs), falls back in order to
  `aria-label` -> `placeholder` -> `value` -> `title` -> `alt` on the element itself, then
  the same two attributes on up to 2 ancestor elements, then a humanized CSS class name
  (last resort, e.g. `search-icon-btn` -> `"search icon btn"`), still fully deterministic.
- `attrs`: compact dict from `id`, `aria-label`, `placeholder`, `name`, `type` (whichever
  exist) plus `href` reduced to just its host (never the full URL/query string).

## Construction

1. Stream-decrypt and stream-parse each `test.zip` shard (`unzip -P mind2web -p` piped
   through `ijson`), no full shard is ever fully materialized in memory or written to
   disk unencrypted.
2. Per step: skip if it has no `pos_candidate` (Mind2Web's own preprocessing already drops
   some) or if the gold element can't be located in `cleaned_html`. Gold = the
   `pos_candidate` with `is_original_target=True`, else the first one.
3. Negatives: take the 9 lowest-rank (hardest / most-confusable) `neg_candidates` per
   `scores_all_data.pkl`, i.e. the same ranker the Mind2Web paper uses for candidate
   generation, not a random 9. Falls back to a seeded random 9 only if a step is missing
   from the scores file (didn't happen for `cross_website`/`cross_domain`: 100% ranked).
   Steps with fewer than 9 negatives available are skipped (rare: <1%, see below).
4. Gold position: for each row, `Random(f"0:{id}:shuffle")` does a Fisher-Yates shuffle of
   the 10 candidates, so `gold_idx` is uniform over 0-9 by construction, independent of
   later sampling.
5. `eval_smoke.jsonl` (50 = 25 cross_website + 25 cross_domain): round-robins across the
   10 `gold_idx` buckets per split before sampling, so the small N still lands inside the
   6-14% sanity band (plain random draws of 25 sometimes didn't).
6. `eval_1k.jsonl` (1,000 = 500 + 500): round-robins across websites (order and
   within-website order both seeded) capping each website at 5% of the split (25 rows) , 
   *when that cap can still reach the target*. `cross_website` only has **10 distinct
   websites** in the entire official test split (see "odd about the source data" below),
   so 5% (25/website) tops out at 250/500; the script falls back to the smallest cap that
   still reaches 500, which is 50/website (10%) here. `cross_domain` easily satisfies the
   5%/25-row cap with 54 distinct websites.

Seed: **0** everywhere (negative fallback sampling, gold-position shuffle, smoke/1k
sampling), pass `Random(f"0:{...}")` per row/site, so results are fully reproducible and
order-independent of shard processing order.

## Per-split counts

| | universe (all valid steps) | `eval_smoke.jsonl` | `eval_1k.jsonl` | distinct sites (1k) | max site share (1k) |
|---|---|---|---|---|---|
| cross_website | 1,313 | 25 | 500 | 10 / 10 | 10% (structural min, see below) |
| cross_domain  | 5,582 | 25 | 500 | 54 | 2% |

Universe = `test_website` 177 tasks / 1,373 steps -> 1,313 rows after skips; `test_domain`
912 tasks / 5,911 steps -> 5,582 rows after skips. Skip reasons (cross_website /
cross_domain): no `pos_candidate` 59 / 321, gold element not found in `cleaned_html` 1 / 8.
9-negatives-found rate was 100% in both splits (no fallback-random negatives used).

## Website distribution (`eval_1k.jsonl`)

- `cross_website` (500 rows, cap 50/site, see below): nba, recreation.gov, bestbuy,
  stubhub, cars, tripadvisor, macys, trip, tiktok.music, shopping.google, **exactly 50
  each**, all 10 websites in the split.
- `cross_domain` (500 rows, cap 25/site): 54 distinct websites, top 10 each contribute 10
  rows (accuweather, glassdoor, bbb.org, dmv.virginia.gov, fedex, landwatch,
  finance.yahoo, linkedin, thumbtack, hiring.amazon), long tail below that.

## Operation distribution

| | CLICK | TYPE | SELECT |
|---|---|---|---|
| `eval_1k.jsonl` cross_website | 396 | 58 | 46 |
| `eval_1k.jsonl` cross_domain | 426 | 64 | 10 |
| `eval_smoke.jsonl` cross_website | 18 | 4 | 3 |
| `eval_smoke.jsonl` cross_domain | 21 | 3 | 1 |

Matches the universe skew (CLICK ~80%, TYPE ~14%, SELECT ~6%), this is Mind2Web's real
action mix, not a sampling artifact.

## Chance baseline

10 candidates per step, uniform `gold_idx` by construction -> **random-choice element
accuracy = 10%**. `eval/build_dataset.py`'s sanity checks assert every `gold_idx` bucket
falls in 6-14% of each output file.

## Odd about the source data

- **`test_website` only spans 10 distinct websites** for all 177 tasks / 1,313 steps (vs.
  54 for `test_domain`'s 912 tasks). This is a real structural property of Mind2Web's
  cross-website test split, not a bug in this pipeline, it's why the "no site over 5%"
  target had to relax to 10% for that split only (documented above and printed by the
  build script).
- `test_domain`'s shards are wildly uneven in size (`test_domain_9.json` is 42MB vs.
  ~300-450MB for the other 9), which shows up as `test_domain_9` contributing far fewer
  rows (65) than the other shards (~550-700 each), shard boundaries aren't uniform
  samples of the task population.
- `neg_candidates` counts per step vary enormously (seen: single digits up to ~490+ in
  this build), so "9 hardest negatives by rank" is a much more meaningful candidate set
  than a random 9 would be, a random sample would mostly pull in obviously-irrelevant
  DOM nodes (page chrome, footers) on high-negative-count pages.
