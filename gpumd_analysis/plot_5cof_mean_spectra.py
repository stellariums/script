"""
功能说明
--------
本脚本用于汇总 5 个 COF 体系的 3 组 HNEMD 重复计算结果，计算每个体系的
平均 classical 谱热导曲线，并绘制 5 条平均谱线的对比图。

脚本默认处理的体系名称为：
- `COF1F`
- `COF2F`
- `COF3F`
- `COF4F`
- `TppaCOF`

每个体系下默认读取 `fold1`、`fold2`、`fold3` 三个子目录中的：
- `shc.out`
- `run.in`
- `dump.xyz`

脚本会自动：
- 从 `shc.out` 读取频谱热流数据
- 从 `run.in` 读取 `compute_hnemd` 的驱动力
- 从 `dump.xyz` 读取晶胞并计算体积
- 将每个体系的 3 条谱线对齐到公共频率网格后取平均
- 导出每个体系的平均谱数据文件和总对比图

目录要求
--------
脚本所在目录下默认应包含如下结构：

    .
    ├── plot_5cof_mean_spectra.py
    ├── COF1F
    │   ├── fold1
    │   ├── fold2
    │   └── fold3
    ├── COF2F
    ├── COF3F
    ├── COF4F
    └── TppaCOF

使用方式
--------
在脚本所在目录运行：

    python plot_5cof_mean_spectra.py

如果只想保存图片、不弹出绘图窗口：

    set SHC_SHOW=0
    python plot_5cof_mean_spectra.py

主要输出
--------
- `five_cof_mean_spectra.png` / `five_cof_mean_spectra.pdf`
  5 个 COF 的平均 classical 谱热导对比图
- `COF1F_kv_mean3.out`、`COF2F_kv_mean3.out` 等
  各体系的两列数据：`nu` 和 `kappa_mean`
- 终端打印每个体系平均谱线积分值

可调整参数
----------
- `TEMPERATURE`
  热导换算使用的温度，默认 `300 K`
- `WHICH_POSITION`
  指定 `compute_hnemd` 中使用哪个方向分量；默认 `None`，
  即自动读取唯一的非零分量
- `COF_DIRS`
  要处理的体系名称列表

注意事项
--------
- 若某个体系的 3 个 fold 频率网格不一致，脚本会插值到公共网格后再平均
- `run.in` 中若存在多个非零 `compute_hnemd` 分量，需要显式设置 `WHICH_POSITION`
"""

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


NUM_CORR_POINTS = 250
LABELS_CORR = ["t", "Ki", "Ko"]
LABELS_OMEGA = ["omega", "jwi", "jwo"]
TEMPERATURE = 300.0
WHICH_POSITION = None
FOLD_NAMES = ["fold1", "fold2", "fold3"]
COF_DIRS = ["COF1F", "COF2F", "COF3F", "COF4F", "TppaCOF"]


def read_shc_out(shc_out_path: str) -> Dict[str, np.ndarray]:
    num_corr_points_in_run = NUM_CORR_POINTS * 2 - 1
    corr_array = np.loadtxt(shc_out_path, max_rows=num_corr_points_in_run)
    omega_array = np.loadtxt(shc_out_path, skiprows=num_corr_points_in_run)

    shc: Dict[str, np.ndarray] = {}
    for index, key in enumerate(LABELS_CORR):
        shc[key] = corr_array[:, index]
    for index, key in enumerate(LABELS_OMEGA):
        shc[key] = omega_array[:, index]

    shc["nu"] = shc["omega"] / (2 * np.pi)
    return shc


def calc_spectral_kappa(shc: Dict[str, np.ndarray], driving_force: float, temperature: float, volume: float) -> None:
    convert = 1602.17662
    shc["kwi"] = shc["jwi"] * convert / (driving_force * temperature * volume)
    shc["kwo"] = shc["jwo"] * convert / (driving_force * temperature * volume)
    shc["kw"] = shc["kwi"] + shc["kwo"]


def read_fe_from_runin(runin_path: str, direction_index: int | None = None) -> float:
    with open(runin_path, "r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "compute_hnemd" not in line:
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            try:
                values = [float(x) for x in parts[-3:]]
            except ValueError:
                continue

            if direction_index is not None:
                return values[direction_index]

            nonzero_values = [value for value in values if abs(value) > 0.0]
            if len(nonzero_values) == 1:
                return nonzero_values[0]
            if len(nonzero_values) > 1:
                raise ValueError(
                    f"Multiple non-zero compute_hnemd components found in {runin_path}; "
                    "set WHICH_POSITION explicitly."
                )
            raise ValueError(f"All compute_hnemd components are zero in {runin_path}")

    raise ValueError(f"Failed to read compute_hnemd driving force from {runin_path}")


def calculate_volume(dump_path: str) -> float:
    with open(dump_path, "r", encoding="utf-8", errors="ignore") as file:
        file.readline()
        header = file.readline().strip()

    key = 'Lattice="'
    start = header.find(key)
    if start == -1:
        raise ValueError(f'Lattice information not found in {dump_path}')
    start += len(key)

    end = header.find('"', start)
    if end == -1:
        raise ValueError(f'Unterminated Lattice field in {dump_path}')

    lattice_numbers = header[start:end].split()
    if len(lattice_numbers) != 9:
        raise ValueError(f"Expected 9 Lattice values in {dump_path}, got {len(lattice_numbers)}")

    lattice = np.array([float(x) for x in lattice_numbers], dtype=float).reshape(3, 3)
    return float(abs(np.linalg.det(lattice)))


def ensure_sorted(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    return x[order], y[order]


def average_curves_on_common_grid(x_list: List[np.ndarray], y_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    same_grid = True
    ref_x = x_list[0]
    for x in x_list[1:]:
        if x.shape != ref_x.shape or not np.allclose(x, ref_x, rtol=0.0, atol=1e-12):
            same_grid = False
            break

    if same_grid:
        return ref_x, np.mean(np.vstack(y_list), axis=0)

    x_min = max(float(np.min(x)) for x in x_list)
    x_max = min(float(np.max(x)) for x in x_list)
    mask = (ref_x >= x_min) & (ref_x <= x_max)
    x_common = ref_x[mask]
    y_interp = [np.interp(x_common, x, y) for x, y in zip(x_list, y_list)]
    return x_common, np.mean(np.vstack(y_interp), axis=0)


def compute_cof_mean(base_dir: str, cof_name: str) -> Dict[str, np.ndarray]:
    nu_list: List[np.ndarray] = []
    kw_classical_list: List[np.ndarray] = []

    for fold_name in FOLD_NAMES:
        fold_dir = os.path.join(base_dir, cof_name, fold_name)
        shc_path = os.path.join(fold_dir, "shc.out")
        runin_path = os.path.join(fold_dir, "run.in")
        dump_path = os.path.join(fold_dir, "dump.xyz")

        shc = read_shc_out(shc_path)
        driving_force = read_fe_from_runin(runin_path, direction_index=WHICH_POSITION)
        volume = calculate_volume(dump_path)
        calc_spectral_kappa(shc, driving_force=driving_force, temperature=TEMPERATURE, volume=volume)

        nu = np.asarray(shc["nu"], dtype=float)
        kw_classical = np.asarray(shc["kw"], dtype=float)
        nu, kw_classical = ensure_sorted(nu, kw_classical)

        nu_list.append(nu)
        kw_classical_list.append(kw_classical)

    nu_mean, kw_classical_mean = average_curves_on_common_grid(nu_list, kw_classical_list)

    return {
        "nu": nu_mean,
        "kw_classical": kw_classical_mean,
    }


def save_cof_tables(base_dir: str, cof_results: Dict[str, Dict[str, np.ndarray]]) -> None:
    for cof_name, result in cof_results.items():
        output_path = os.path.join(base_dir, f"{cof_name}_kv_mean3.out")
        np.savetxt(
            output_path,
            np.c_[result["nu"], result["kw_classical"]],
            header="Frequency (THz), kappa_mean (W/mK/THz)",
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
        y_candidates.append(result["kw_classical"][in_view])
    y_max_data = float(np.nanmax(np.concatenate(y_candidates)))
    y_max = y_max_data * 1.08 if np.isfinite(y_max_data) and y_max_data > 0 else 1.0

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.8), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors

    for index, cof_name in enumerate(COF_DIRS):
        result = cof_results[cof_name]
        color = colors[index % len(colors)]
        ax.plot(result["nu"], result["kw_classical"], linewidth=2.0, color=color, label=cof_name)

    ax.set_title("Mean spectra of 5 COFs")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel(r"$\nu$ (THz)")
    ax.set_ylabel(r"$\kappa(\omega)$ (W/m/K/THz)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(frameon=False, fontsize=9)

    png_path = os.path.join(base_dir, "five_cof_mean_spectra.png")
    pdf_path = os.path.join(base_dir, "five_cof_mean_spectra.pdf")
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
        cof_results[cof_name] = compute_cof_mean(base_dir, cof_name)
        classical_integral = float(np.trapezoid(cof_results[cof_name]["kw_classical"], cof_results[cof_name]["nu"]))
        print(f"{cof_name}: integral={classical_integral:.6f}")

    save_cof_tables(base_dir, cof_results)
    plot_results(base_dir, cof_results)


if __name__ == "__main__":
    main()
