# Training data

Every builder writes rows in the eval set's schema (`data/README.md`): goal, previous actions,
10 candidate elements, gold index, operation. Outputs go to `data/` and are not committed.

| File | Builder | Rows | Websites |
|---|---|---|---|
| `data/train_m2w.jsonl` | `mind2web_train.py` | 13,118 | 66 |
| `data/val.jsonl` | `mind2web_train.py` | 1,604 | 7 |
| `data/train_webchain.jsonl` | `webchain.py --max-parts 6` | 28,803 | 97 |

No website in any of these appears in the eval set.

## Mind2Web

`mind2web_train.py` streams the plain-JSON train shards of
[osunlp/Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) (CC BY 4.0). Every step becomes
two rows: one with the 9 hardest negatives by the Mind2Web candidate ranker's rank, one with 9
random negatives from the same page, each with its own gold position. Whole websites are held out
for validation (10% of sites), so validation measures the same thing the eval does: pages the
model has never seen.

## WebChain

`webchain.py` reads [WebChain](https://huggingface.co/datasets/webagentlab/WebChain)'s action
metadata (CC BY 4.0), fetches the pre-action DOM snapshot for each step, locates the target by its
CSS selector and takes the other interactive elements on the page as candidates. There is no
ranker for WebChain, so negatives favour elements whose text overlaps the goal. It is
network-bound (about 11 rows a second) and runs to a time budget: `--minutes`, `--workers`.

## Kev format

`to_kev_format.py` turns rows into Kev's training records: the compact state text, an element
question over `[i] <tag> "text"` options and an operation question over CLICK, TYPE and SELECT.
`train/train.sh` runs it on the mixture.
