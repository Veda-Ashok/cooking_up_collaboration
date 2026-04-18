from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from results.utils.plot_style import PALETTE, apply_publication_style, save_figure


HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "overcooked_blue_green",
    ["#FFFFFF", PALETTE["green_1"], PALETTE["green_3"], PALETTE["blue_secondary"], PALETTE["blue_main"]],
)


def plot_heatmap_grid(
    matrices: dict[str, list[list[float]]],
    row_labels: list[str],
    col_labels: list[str],
    output_dir: str | Path,
    stem: str,
    title: str,
    value_label: str = "Mean reward",
) -> list[Path]:
    """Plot one annotated cross-partner heatmap per layout."""
    apply_publication_style(font_size=13, axes_linewidth=2.0)
    layout_names = list(matrices.keys())
    if not layout_names:
        raise ValueError("No matrices provided for heatmap plot")

    fig = plt.figure(figsize=(5.4 * len(layout_names) + 0.75, 5.8))
    grid = fig.add_gridspec(
        1,
        len(layout_names) + 1,
        width_ratios=[1.0] * len(layout_names) + [0.055],
        wspace=0.50,
    )
    axes_flat = [fig.add_subplot(grid[0, idx]) for idx in range(len(layout_names))]
    cbar_ax = fig.add_subplot(grid[0, len(layout_names)])
    all_values = np.asarray([v for matrix in matrices.values() for row in matrix for v in row], dtype=float)
    finite = all_values[np.isfinite(all_values)]
    vmin = 0.0
    vmax = float(np.max(finite)) if finite.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    image = None
    for ax, layout_name in zip(axes_flat, layout_names):
        data = np.asarray(matrices[layout_name], dtype=float)
        image = ax.imshow(data, cmap=HEATMAP_CMAP, vmin=vmin, vmax=vmax)
        ax.set_title(layout_name, weight="bold")
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels, rotation=35, ha="center", rotation_mode="anchor")
        ax.set_yticklabels(row_labels)
        ax.set_xlabel("Player 1 partner", labelpad=10)
        ax.set_ylabel("Player 0 agent", labelpad=8)
        ax.tick_params(axis="x", length=0, pad=16)
        ax.tick_params(axis="y", length=0, pad=8)

        threshold = vmin + 0.58 * (vmax - vmin)
        for row_idx in range(data.shape[0]):
            for col_idx in range(data.shape[1]):
                value = data[row_idx, col_idx]
                color = "white" if value >= threshold else "#272727"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    weight="bold",
                    color=color,
                )

    if image is not None:
        cbar = fig.colorbar(image, cax=cbar_ax)
        cbar.set_label(value_label, labelpad=12)
        cbar.outline.set_linewidth(1.5)

    fig.suptitle(title, weight="bold", y=0.98)
    fig.subplots_adjust(left=0.055, right=0.965, bottom=0.22, top=0.82)
    return save_figure(fig, output_dir, stem, tight_layout=False)
