"""Shared matplotlib style so every figure in reports/ and MLflow looks consistent."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: figures are saved, never shown
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = [BLUE, ORANGE, AQUA, VIOLET]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 140, "savefig.bbox": "tight",
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "font.size": 9,
        "legend.frameon": False,
    })


apply_style()
