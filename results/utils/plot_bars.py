from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as plt
import numpy as np
from results.utils.plot_style import METHOD_COLORS, PALETTE, apply_publication_style, save_figure


def _auto_ylim(values: np.ndarray, errors: np.ndarray | None = None) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 1.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low = min(0.0, float(np.nanmin(finite)))
    high = float(np.nanmax(finite))
    if errors is not None and np.isfinite(errors).any():
        high = max(high, float(np.nanmax(values + errors)))
    pad = max(2.0, 0.20 * (high - low if high > low else abs(high) + 1.0))
    return low, high + pad


def plot_grouped_bars(
    groups: list[str],
    series_names: list[str],
    values: list[list[float]],
    errors: list[list[float]] | None,
    ylabel: str,
    title: str,
    output_dir: str | Path,
    stem: str,
    annotate: bool = True,
    figsize: tuple[float, float] | None = None,
    legend_inside: bool = False,
) -> list[Path]:
    """Create a figures4papers-style grouped bar chart."""
    apply_publication_style(font_size=16, axes_linewidth=2.4)
    value_arr = np.asarray(values, dtype=float)
    if value_arr.shape != (len(series_names), len(groups)):
        raise ValueError(
            f"values must have shape ({len(series_names)}, {len(groups)}), got {value_arr.shape}"
        )
    error_arr = None
    if errors is not None:
        error_arr = np.asarray(errors, dtype=float)
        if error_arr.shape != value_arr.shape:
            raise ValueError(f"errors must match values shape, got {error_arr.shape}")

    if figsize is None:
        figsize = (max(11.0, 2.4 * len(groups) + 1.35 * len(series_names)), 7.2)
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(groups))
    width = min(0.78 / max(len(series_names), 1), 0.18)
    offsets = (np.arange(len(series_names)) - (len(series_names) - 1) / 2.0) * width

    for i, series_name in enumerate(series_names):
        color = METHOD_COLORS.get(series_name, list(PALETTE.values())[i % len(PALETTE)])
        yerr = error_arr[i] if error_arr is not None else None
        bars = ax.bar(
            x + offsets[i],
            value_arr[i],
            width=width,
            yerr=yerr,
            label=series_name,
            color=color,
            edgecolor="black",
            linewidth=1.6,
            capsize=4 if yerr is not None else 0,
            error_kw={"elinewidth": 1.5, "capthick": 1.5},
        )
        if annotate:
            for bar in bars:
                height = bar.get_height()
                if not np.isfinite(height):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    rotation=0,
                )

    ax.set_title(title, pad=22, weight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.tick_params(axis="both", width=2.0, length=6)
    ax.set_ylim(*_auto_ylim(value_arr, error_arr))
    legend_cols = min(len(series_names), 2 if legend_inside else 3)
    if legend_inside:
        ax.legend(
            ncol=legend_cols,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            columnspacing=1.4,
            handletextpad=0.7,
            borderaxespad=0.0,
        )
        fig.subplots_adjust(top=0.88)
    else:
        ax.legend(
            ncol=legend_cols,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.13),
            columnspacing=1.8,
            handletextpad=0.8,
            borderaxespad=0.0,
        )
        fig.subplots_adjust(top=0.68)
    return save_figure(fig, output_dir, stem)


def plot_small_multipanel_bars(
    panels: Iterable[dict],
    output_dir: str | Path,
    stem: str,
    title: str,
) -> list[Path]:
    """Create compact side-by-side bar panels for BC validation metrics."""
    panel_list = list(panels)
    apply_publication_style(font_size=14, axes_linewidth=2.0)
    fig, axes = plt.subplots(1, len(panel_list), figsize=(5.3 * len(panel_list), 5.2), sharey=False)
    if len(panel_list) == 1:
        axes = [axes]

    for ax, panel in zip(axes, panel_list):
        groups = panel["groups"]
        series = panel["series"]
        values = np.asarray(panel["values"], dtype=float)
        x = np.arange(len(groups))
        width = min(0.7 / len(series), 0.28)
        offsets = (np.arange(len(series)) - (len(series) - 1) / 2.0) * width
        for i, name in enumerate(series):
            ax.bar(
                x + offsets[i],
                values[i],
                width=width,
                label=name,
                color=METHOD_COLORS.get(name, PALETTE["neutral"]),
                edgecolor="black",
                linewidth=1.4,
            )
        ax.set_title(panel["title"], weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=20, ha="center", rotation_mode="anchor")
        ax.set_ylabel(panel.get("ylabel", "Metric"))
        ax.set_ylim(*_auto_ylim(values))
        ax.tick_params(axis="x", width=1.8, length=5, pad=12)
        ax.tick_params(axis="y", width=1.8, length=5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(labels), loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(title, weight="bold", y=1.12)
    fig.subplots_adjust(bottom=0.22)
    return save_figure(fig, output_dir, stem)
