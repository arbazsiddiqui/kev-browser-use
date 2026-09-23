"""Shared fixtures for eval/tests/: real data/eval_smoke.jsonl rows, no model loading."""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.harness import load_dataset  # noqa: E402

SMOKE_PATH = REPO_ROOT / "data" / "eval_smoke.jsonl"


def _visible(row):
    """Mirrors eval/harness.py's _predict_one: adapters never see gold_idx/operation/value."""
    return {k: v for k, v in row.items() if k not in ("gold_idx", "operation", "value")}


@pytest.fixture(scope="session")
def smoke_rows():
    return [_visible(r) for r in load_dataset(SMOKE_PATH)]
