from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def build_rwgame_cmap():
    border_color_3 = np.array([204, 153, 255]) / 255.0
    border_color_2 = np.array([178, 102, 255]) / 255.0
    border_color_1 = np.array([141, 29, 255]) / 255.0
    white_color = np.array([255, 255, 255]) / 255.0
    center_color_1 = np.array([255, 202, 26]) / 255.0
    center_color_2 = np.array([255, 219, 102]) / 255.0
    center_color_3 = np.array([255, 231, 153]) / 255.0
    return mcolors.LinearSegmentedColormap.from_list(
        "rwgame",
        [
            border_color_1,
            border_color_2,
            border_color_3,
            white_color,
            center_color_3,
            center_color_2,
            center_color_1,
        ],
    )


def _plot_strategy_matrix(
    strategy_matrix,
    output_path,
    title,
    one_label,
    zero_label,
    colorbar_top_label,
    colorbar_bottom_label,
    dpi=300,
    value_font_size=5,
    title_font_size=8,
):
    strategy_matrix = np.asarray(strategy_matrix, dtype=float)
    if strategy_matrix.ndim != 2:
        raise ValueError("strategy_matrix must be a 2D array.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rwgame_cmap = build_rwgame_cmap()

    value_font = {"family": "sans-serif", "size": value_font_size}
    mpl.rc("font", **value_font)

    fig, ax = plt.subplots(1, 1, dpi=dpi)
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    image = ax.imshow(strategy_matrix, cmap=rwgame_cmap)

    for (j, i), label in np.ndenumerate(strategy_matrix):
        print_str_1 = f"\n{one_label}" if np.isclose(label, 1.0) else ""
        print_str_2 = f"\n{zero_label}" if np.isclose(label, 0.0) else ""
        text = f"{label:.2f} " + print_str_1 + print_str_2
        plt.text(i, j, text, ha="center", va="center")
        plt.text(i, j, text, ha="center", va="center")

    title_font = {"family": "sans-serif", "size": title_font_size}
    mpl.rc("font", **title_font)

    cbar = fig.colorbar(image)
    cbar.ax.text(4, 1, colorbar_top_label, ha="center", va="center")
    cbar.ax.text(2, 0.03, "0", ha="center", va="center")
    cbar.ax.text(4, 0.03, colorbar_bottom_label, ha="center", va="center")

    ax.set_title(title)
    fig.savefig(output_path)
    plt.close(fig)


def plot_border_strategy_matrix(
    strategy_matrix,
    output_path,
    title="Вероятность игрока \"за центр\" выбрать 1 стратегию",
    dpi=300,
    value_font_size=5,
    title_font_size=8,
):
    _plot_strategy_matrix(
        strategy_matrix=strategy_matrix,
        output_path=output_path,
        title=title,
        one_label="\u2191\u2193",
        zero_label="\u2192\u2190",
        colorbar_top_label="\u2191\n\u2193",
        colorbar_bottom_label="\u2192\n\u2190",
        dpi=dpi,
        value_font_size=value_font_size,
        title_font_size=title_font_size,
    )


def plot_real_border_strategy_from_npy(
    strategy_path="../data/strategy_pve_border.npy",
    output_path="../output/real_border_strategy_pve.png",
    title="Вероятность игрока \"за центр\" выбрать 1 стратегию (PVE).",
):
    strategy_matrix = np.load(strategy_path)
    plot_border_strategy_matrix(
        strategy_matrix=strategy_matrix,
        output_path=output_path,
        title=title,
    )
    return strategy_matrix


def plot_center_strategy_matrix(
    strategy_matrix,
    output_path,
    title="Вероятность игрока \"за центр\" выбрать 1 стратегию (PVE).",
    dpi=300,
    value_font_size=5,
    title_font_size=8,
):
    _plot_strategy_matrix(
        strategy_matrix=strategy_matrix,
        output_path=output_path,
        title=title,
        one_label="\u2191\u2192",
        zero_label="\u2193\u2190",
        colorbar_top_label="\u2191\n\u2192",
        colorbar_bottom_label="\u2193\n\u2190",
        dpi=dpi,
        value_font_size=value_font_size,
        title_font_size=title_font_size,
    )


def plot_real_center_strategy_from_npy(
    strategy_path="../data/strategy_pve_center.npy",
    output_path="../output/real_center_strategy_pve.png",
    title="Вероятность игрока \"за центр\" выбрать 1 стратегию (PVE).",
):
    strategy_matrix = np.load(strategy_path).copy()
    strategy_matrix[1:-1, 1:-1] = 1.0 - strategy_matrix[1:-1, 1:-1]
    plot_center_strategy_matrix(
        strategy_matrix=strategy_matrix,
        output_path=output_path,
        title=title,
    )
    return strategy_matrix
