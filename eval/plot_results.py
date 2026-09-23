"""Draws assets/results.png, the README's results chart, from results/*.json.

    python3 eval/plot_results.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "results.png"

OURS = "#1f9e89"
BASE = "#b04040"
OTHER = "#9ca0a8"

# (label, result file, colour), the same rows as the README table
ROWS = [
    ("Jev (hosted, closed)", "jev_1k.json", OTHER),
    ("Kev-9B  (9B)", "kev_1k.json", OTHER),
    ("kev-0.6b-browser-use  (0.6B)", "kev_1k-final-kev06-mix.json", OURS),
    ("SemIf, frozen Qwen3.5-4B  (4B)", "semif_1k-semif-qwen35-4b.json", OTHER),
    ("Bespoke Nimble-9B  (9B)", "nimble_1k.json", OTHER),
    ("decider-2B  (2B)", "decider_1k.json", OTHER),
    ("Kev-0.6B (base)", "kev_1k-kev06.json", BASE),
    ("NanoJev  (0.6B)", "nanojev_1k.json", OTHER),
    ("Laya  (421M)", "laya_1k.json", OTHER),
    ("Random", "random_1k.json", OTHER),
]
METRICS = [
    ("step_success", "Step success"),
    ("element_acc", "Element accuracy"),
    ("operation_acc", "Operation accuracy"),
]


def load():
    rows = []
    for label, name, colour in ROWS:
        m = json.loads((ROOT / "results" / name).read_text())["metrics"]["overall"]
        rows.append((label, colour, {k: 100 * m[k] for k, _ in METRICS}))
    return rows


def main():
    rows = load()
    labels = [r[0] for r in rows][::-1]
    colours = [r[1] for r in rows][::-1]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.0), sharey=True, dpi=150)
    for ax, (key, title) in zip(axes, METRICS):
        vals = [r[2][key] for r in rows][::-1]
        bars = ax.barh(labels, vals, color=colours, height=0.66)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2, f"{v:.1f}",
                    va="center", fontsize=9.5, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.set_title(title, fontsize=12.5, fontweight="bold", pad=8)
        ax.set_xlabel("% of 1,000 steps", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=10.5)
        ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("kev-0.6b-browser-use on Mind2Web: 1,000 test steps, 10 candidate elements each",
                 fontsize=14, fontweight="bold", y=0.985)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (OURS, BASE, OTHER)]
    fig.legend(handles, ["kev-0.6b-browser-use (ours)", "Kev-0.6B (base)", "other models"],
               loc="lower left", bbox_to_anchor=(0.01, 0.045), ncol=3, frameon=False, fontsize=10.5)
    fig.text(0.017, 0.02,
             "Step success: correct element and correct action in the same step.   Element accuracy: picked "
             "the correct UI element.   Operation accuracy: correct action on it (click, type or select).",
             ha="left", fontsize=9.5, color="#555555")
    fig.tight_layout(rect=(0, 0.11, 1, 0.97))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
