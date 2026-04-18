from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


def load_font(optional_path: str | None, fallback_family: str):
    if optional_path and Path(optional_path).exists():
        try:
            fm.fontManager.addfont(optional_path)
            return fm.FontProperties(fname=optional_path)
        except Exception:
            pass
    return fm.FontProperties(family=fallback_family)


def setup_basic_plot_style():
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS", "SimHei"]
