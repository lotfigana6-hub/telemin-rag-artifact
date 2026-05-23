from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def box(ax, xy: tuple[float, float], width: float, height: float, text: str, color: str) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor="#222222",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=10)


def arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#333333"})


def architecture(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.set_axis_off()
    box(ax, (0.04, 0.58), 0.18, 0.18, "Alert\ntelemetry", "#d8f3dc")
    box(ax, (0.28, 0.73), 0.17, 0.16, "Importance\nMI", "#fef3c7")
    box(ax, (0.28, 0.52), 0.17, 0.16, "Semantic\nsimilarity", "#dbeafe")
    box(ax, (0.28, 0.31), 0.17, 0.16, "Rarity and\ncoherence", "#ede9fe")
    box(ax, (0.51, 0.58), 0.17, 0.18, "Poison and\nredundancy\npenalties", "#fee2e2")
    box(ax, (0.73, 0.58), 0.20, 0.18, "Minimal trusted\ncontext", "#ccfbf1")
    box(ax, (0.73, 0.26), 0.20, 0.18, "Classifier or\nLLM triage", "#e5e7eb")
    arrow(ax, (0.22, 0.67), (0.28, 0.81))
    arrow(ax, (0.22, 0.67), (0.28, 0.60))
    arrow(ax, (0.22, 0.67), (0.28, 0.39))
    arrow(ax, (0.45, 0.81), (0.51, 0.68))
    arrow(ax, (0.45, 0.60), (0.51, 0.68))
    arrow(ax, (0.45, 0.39), (0.51, 0.68))
    arrow(ax, (0.68, 0.67), (0.73, 0.67))
    arrow(ax, (0.83, 0.58), (0.83, 0.44))
    ax.text(0.5, 0.08, "TeleMin-RAG scores events before any downstream SOC assistant consumes them.", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def pipeline(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.set_axis_off()
    labels = [
        ("Public\ndatasets", "#d8f3dc"),
        ("Context\nwindows", "#fef3c7"),
        ("Noise and\npoisoning", "#fee2e2"),
        ("Selectors and\nbaselines", "#dbeafe"),
        ("Downstream\nmodels", "#ede9fe"),
        ("Metrics,\nCI, tests", "#ccfbf1"),
    ]
    x = 0.04
    for idx, (label, color) in enumerate(labels):
        box(ax, (x, 0.55), 0.13, 0.20, label, color)
        if idx < len(labels) - 1:
            arrow(ax, (x + 0.13, 0.65), (x + 0.17, 0.65))
        x += 0.16
    box(ax, (0.23, 0.22), 0.17, 0.16, "Seeds\n1..5", "#f3f4f6")
    box(ax, (0.45, 0.22), 0.20, 0.16, "Attacks:\ninstruction,\nobfuscated,\nsemantic", "#f3f4f6")
    box(ax, (0.70, 0.22), 0.20, 0.16, "Figures and\nLaTeX tables", "#f3f4f6")
    arrow(ax, (0.31, 0.38), (0.31, 0.55))
    arrow(ax, (0.55, 0.38), (0.55, 0.55))
    arrow(ax, (0.80, 0.38), (0.80, 0.55))
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    architecture(out_dir / "telemin_architecture.png")
    pipeline(out_dir / "experimental_pipeline.png")


if __name__ == "__main__":
    main()
