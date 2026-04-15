"""Publication-style Matplotlib helpers.

The visual conventions here are adapted from the figures4papers repository:
large sans-serif typography, minimal spines, black-edged bars, direct labels,
and PNG/PDF export for presentation and paper use.
"""
from pathlib import Path
import matplotlib.pyplot as plt


PUBLICATION_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "font.size": 15,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 2.2,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "neutral_dark": "#4D4D4D",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}


METHOD_COLORS = {
    "BC MLP": PALETTE["neutral"],
    "BC LSTM": PALETTE["green_2"],
    "PPO + BC": PALETTE["blue_secondary"],
    "PPO Self-Play": PALETTE["blue_main"],
    "Random": PALETTE["red_2"],
    "PPO=P0": PALETTE["blue_main"],
    "PPO=P1": PALETTE["teal"],
}


def apply_publication_style(font_size: int = 15, axes_linewidth: float = 2.2) -> None:
    """Apply a reusable publication-style Matplotlib preset."""
    rcparams = dict(PUBLICATION_RCPARAMS)
    rcparams["font.size"] = font_size
    rcparams["axes.linewidth"] = axes_linewidth
    plt.rcParams.update(rcparams)


def save_figure(
    fig,
    output_dir: str | Path,
    stem: str,
    dpi: int = 300,
    tight_layout: bool = True,
) -> list[Path]:
    """Save a figure as both PNG and PDF, returning the written paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    png_path = output_path / f"{stem}.png"
    pdf_path = output_path / f"{stem}.pdf"
    if tight_layout:
        fig.tight_layout(pad=2)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    return [png_path, pdf_path]
