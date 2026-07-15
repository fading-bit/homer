"""
viz.py — shared plotting helpers (Greek font, styling, saving).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont, FontProperties

# DejaVu Sans ships with matplotlib and covers monotonic + polytonic Greek.
GREEK_FONT_PATH = findfont(FontProperties(family="DejaVu Sans"))
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

# Consistent colours for the two poems.
POEM_COLORS = {"iliad": "#b5651d", "odyssey": "#1d6fb5"}


def savefig(fig, figures_dir: str | Path, name: str):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
