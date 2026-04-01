"""
功能说明
--------
本脚本基于 `plot_5cof_mean_spectra.py` 中的 `compute_cof_mean()` 结果，
进一步计算 5 个 COF 平均 classical 谱热导曲线的累积积分，并绘制归一化后的
累计热导曲线对比图。

脚本处理流程为：
- 先调用 `compute_cof_mean()` 得到每个 COF 的平均谱热导
- 对 `kappa(omega)` 做累计梯形积分
- 用最终积分值对累计曲线归一化，使终点为 1
- 导出每个体系的累计数据表，并绘制 5 条归一化累计曲线

使用方式
--------
在脚本所在目录运行：

    python plot_5cof_cumulative_shc.py

如果只想保存图片、不弹出绘图窗口：

    set SHC_SHOW=0
    python plot_5cof_cumulative_shc.py

前置条件
--------
本脚本依赖 `plot_5cof_mean_spectra.py` 中定义的：
- `COF_DIRS`
- `compute_cof_mean()`

因此目录结构和输入文件要求与 `plot_5cof_mean_spectra.py` 保持一致，也就是说
脚本所在目录下默认应存在 `COF1F`、`COF2F`、`COF3F`、`COF4F`、`TppaCOF`
等体系目录，以及各自的 `fold1`、`fold2`、`fold3` 数据。

主要输出
--------
- `five_cof_cumulative_shc_normalized.png`
- `five_cof_cumulative_shc_normalized.pdf`
  5 个 COF 的归一化累计热导对比图
- `COF1F_kv_cumulative_mean3.out`、`COF2F_kv_cumulative_mean3.out` 等
  各体系两列数据：`nu` 和 `normalized_cumulative_kappa`
- 终端打印每个体系的累计积分终值与归一化终值

注意事项
--------
- 若某个体系的最终累计值为 0 或非法值，归一化会报错
- 本脚本绘制的是归一化累计热导，不是未归一化的原始累计积分曲线
"""

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from plot_5cof_mean_spectra import COF_DIRS, compute_cof_mean


def cumulative_trapezoid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    cumulative = np.zeros_like(y)
    if len(x) < 2:
        return cumulative
    increments = 0.5 * (y[:-1] + y[1:]) * (x[1:] - x[:-1])
    cumulative[1:] = np.cumsum(increments)
    return cumulative


def normalize_cumulative(cumulative: np.ndarray) -> np.ndarray:
    final_value = float(cumulative[-1])
    if not np.isfinite(final_value) or abs(final_value) < 1e-15:
        raise ValueError("Cannot normalize cumulative curve with zero or invalid final value.")
    return cumulative / final_value


def save_cumulative_tables(base_dir: str, cof_results: Dict[str, Dict[str, np.ndarray]]) -> None:
    for cof_name, result in cof_results.items():
        output_path = os.path.join(base_dir, f"{cof_name}_kv_cumulative_mean3.out")
        np.savetxt(
            output_path,
            np.c_[result["nu"], result["kappa_cumulative_norm"]],
            header="Frequency (THz), normalized_cumulative_kappa",
            comments="",
            fmt="%.6e",
        )


def plot_results(base_dir: str, cof_results: Dict[str, Dict[str, np.ndarray]]) -> None:
    x_max_data = max(float(np.nanmax(result["nu"])) for result in cof_results.values())
    x_max = min(50.0, x_max_data) if np.isfinite(x_max_data) else 50.0
    x_max = float(np.ceil(x_max / 5.0) * 5.0) if x_max > 0 else 50.0

    y_candidates = []
    for result in cof_results.values():
        in_view = (result["nu"] >= 0) & (result["nu"] <= x_max)
        y_candidates.append(result["kappa_cumulative_norm"][in_view])
    y_max_data = float(np.nanmax(np.concatenate(y_candidates)))
    y_max = y_max_data * 1.02 if np.isfinite(y_max_data) and y_max_data > 0 else 1.0

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.8), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors

    for index, cof_name in enumerate(COF_DIRS):
        result = cof_results[cof_name]
        color = colors[index % len(colors)]
        ax.plot(result["nu"], result["kappa_cumulative_norm"], linewidth=2.0, color=color, label=cof_name)

    ax.set_title("Normalized cumulative SHC of 5 COFs")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, max(1.0, y_max))
    ax.set_xlabel(r"$\nu$ (THz)")
    ax.set_ylabel(r"Normalized cumulative $\kappa$")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(frameon=False, fontsize=9)

    png_path = os.path.join(base_dir, "five_cof_cumulative_shc_normalized.png")
    pdf_path = os.path.join(base_dir, "five_cof_cumulative_shc_normalized.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)

    if os.environ.get("SHC_SHOW", "1") == "1":
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cof_results: Dict[str, Dict[str, np.ndarray]] = {}

    for cof_name in COF_DIRS:
        mean_result = compute_cof_mean(base_dir, cof_name)
        kappa_cumulative = cumulative_trapezoid(mean_result["nu"], mean_result["kw_classical"])
        kappa_cumulative_norm = normalize_cumulative(kappa_cumulative)
        cof_results[cof_name] = {
            "nu": mean_result["nu"],
            "kappa_cumulative": kappa_cumulative,
            "kappa_cumulative_norm": kappa_cumulative_norm,
        }
        print(f"{cof_name}: cumulative_end={kappa_cumulative[-1]:.6f}, normalized_end={kappa_cumulative_norm[-1]:.6f}")

    save_cumulative_tables(base_dir, cof_results)
    plot_results(base_dir, cof_results)


if __name__ == "__main__":
    main()
