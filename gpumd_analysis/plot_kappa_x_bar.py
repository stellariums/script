"""
功能说明
--------
本脚本用于从 Excel 文件中读取热导数据，按指定方向绘制带误差棒的柱状图。
当前实现会从 `kappa.xlsx` 中提取 `x` 或 `y` 方向的数据，计算每个材料的平均值
和标准误（SEM），然后输出柱状图。

脚本会自动：
- 读取 Excel 中对应方向的数据分区
- 将第 2 到第 4 列视为重复计算结果
- 计算每个材料的平均值和标准误
- 若表中存在 `mean` 列，会校验计算结果与其是否一致

使用方式
--------
默认读取当前目录下的 `kappa.xlsx`，绘制 x 方向柱状图：

    python plot_kappa_x_bar.py

指定输入文件、方向和输出文件：

    python plot_kappa_x_bar.py -i kappa.xlsx -d y -o kappa_y_bar.png

保存图片后同时显示窗口：

    python plot_kappa_x_bar.py --show

参数说明
--------
- `-i, --input`
  输入 Excel 文件路径，默认 `kappa.xlsx`
- `-d, --direction`
  绘图方向，可选 `x` 或 `y`，默认 `x`
- `-o, --output`
  输出图片路径，默认 `kappa_<direction>_bar.png`
- `--show`
  保存后显示图像窗口

主要输出
--------
- 柱状图图片文件，例如 `kappa_x_bar.png`
- 终端打印各材料的标签、平均值、SEM 和图片保存路径

注意事项
--------
- Excel 数据需按脚本约定格式排布，方向分区需能被脚本识别
- 第 2 到第 4 列应为重复值；若有效重复数小于 2，则 SEM 会记为 `NaN`
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABEL_MAP = {
    "1f": "COF1F",
    "2f": "COF2F",
    "3f": "COF3F",
    "4f": "COF4F",
    "tppa": "Tppa",
    "tppacof": "Tppa",
}

def normalize_label(raw_label):
    text = str(raw_label).strip()
    if not text:
        return text
    return LABEL_MAP.get(text.lower(), text)


def load_direction_section(excel_path, direction):
    df = pd.read_excel(excel_path)
    direction = direction.lower()
    first_column_name = str(df.columns[0]).strip().lower()

    section_rows = []
    if direction == first_column_name:
        for _, row in df.iterrows():
            name = row.iloc[0]
            if pd.isna(name):
                break
            section_rows.append(row)
    else:
        in_section = False
        for _, row in df.iterrows():
            name = row.iloc[0]
            if pd.isna(name):
                if in_section:
                    break
                continue

            name_text = str(name).strip().lower()
            if name_text == direction:
                in_section = True
                continue

            if in_section:
                if name_text in {"x", "y", "z"}:
                    break
                section_rows.append(row)

    if not section_rows:
        raise ValueError(f"No {direction}-direction data found in the Excel file.")

    section_df = pd.DataFrame(section_rows).copy()
    replicate_cols = [1, 2, 3]

    values = section_df[replicate_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    labels = [normalize_label(label) for label in section_df.iloc[:, 0]]
    valid_counts = np.sum(~np.isnan(values), axis=1)

    means = np.full(len(labels), np.nan)
    mean_mask = valid_counts > 0
    means[mean_mask] = np.nanmean(values[mean_mask], axis=1)

    sems = np.full(len(labels), np.nan)
    valid_mask = valid_counts >= 2
    sems[valid_mask] = (
        np.nanstd(values[valid_mask], axis=1, ddof=1) / np.sqrt(valid_counts[valid_mask])
    )

    if "mean" in section_df.columns:
        excel_means = pd.to_numeric(section_df["mean"], errors="coerce").to_numpy()
        mask = ~np.isnan(excel_means)
        if mask.any() and not np.allclose(means[mask], excel_means[mask], atol=5e-4):
            raise ValueError("Computed means do not match the 'mean' column in kappa.xlsx.")

    return labels, means, sems


def plot_kappa_bar(labels, means, sems, output_path, direction, show=False):
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
    x = np.arange(len(labels))
    valid_idx = np.where(~np.isnan(means))[0]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(
        x[valid_idx],
        means[valid_idx],
        yerr=sems[valid_idx],
        width=0.68,
        color=[colors[i] for i in valid_idx],
        edgecolor="black",
        linewidth=1.2,
        ecolor="black",
        capsize=6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(rf"$\kappa_{direction}$ (W m$^{{-1}}$ K$^{{-1}}$)")
    ax.set_title(f"{direction.upper()}-direction thermal conductivity")
    ymax = np.nanmax(means + np.nan_to_num(sems, nan=0.0))
    ax.set_ylim(0, ymax * 1.22)
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for idx, bar in zip(valid_idx, bars):
        mean = means[idx]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot a bar chart with error bars for thermal conductivity."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="kappa.xlsx",
        help="Input Excel file path. Default: kappa.xlsx",
    )
    parser.add_argument(
        "-d",
        "--direction",
        default="x",
        choices=["x", "y"],
        help="Thermal conductivity direction to plot. Default: x",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output figure path. Default: kappa_<direction>_bar.png",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving it.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else Path(f"kappa_{args.direction}_bar.png")

    labels, means, sems = load_direction_section(input_path, args.direction)
    plot_kappa_bar(labels, means, sems, output_path, args.direction, show=args.show)

    print("Labels:", ", ".join(labels))
    print("Means :", ", ".join(f"{value:.4f}" if not np.isnan(value) else "NaN" for value in means))
    print("SEM   :", ", ".join(f"{value:.4f}" if not np.isnan(value) else "NaN" for value in sems))
    print(f"Figure saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
