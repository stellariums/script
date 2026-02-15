"""
用途
----
从 fold1 / fold2 / fold3 三个目录中读取 HNEMD 的 shc.out、run.in、dump.xyz，
分别计算每个 fold 的谱热导率 κ(ν)（Classical）及其量子修正曲线（Quantum-corrected），
然后对三条曲线按频率逐点求平均，绘制并保存平均线图与数据文件。

输入文件（每个 fold 目录下都需要）
-----------------------------
- shc.out：谱相关输出（默认：前 2*250-1 行为相关函数，后续为频谱数据）
- run.in：用于读取 driving force（compute_hnemd 行末 3 个方向分量）
- dump.xyz：用于读取晶胞 Lattice 并计算体积 V（取 det(Lattice) 的绝对值）

使用方法
--------
在本脚本所在目录（COF/compute/COF）运行：

    python shc_3.py

如需在无图形界面环境运行且不弹窗显示图像，可设置环境变量：

    set SHC_SHOW=0
    python shc_3.py

输出文件（生成在“运行时的当前工作目录”）
------------------------------------
- shc_spectral_kappa_mean3.png / shc_spectral_kappa_mean3.pdf：左右两幅平均曲线（Classical / Quantum）
- shc_spectral_kappa_mean3_classical.png：仅 Classical 平均曲线
- shc_spectral_kappa_mean3_quantum.png：仅 Quantum 平均曲线
- kv_mean3.out：三列数据（nu, kappa_classical_mean, kappa_quantum_mean）

注意事项
--------
- 默认温度 temperature=300 K，默认方向 which_position=0（x 方向）；如需修改可在 main() 内调整。
- 如果三个 fold 的频率网格不完全一致，会将曲线插值到 fold1 的频率网格（取三者重叠区间）后再平均。
- 若 shc.out、run.in、dump.xyz 缺失或格式不符合预期，将抛出异常并停止。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


NUM_CORR_POINTS, NUM_OMEGA = 250, 1000
LABELS_CORR = ["t", "Ki", "Ko"]
LABELS_OMEGA = ["omega", "jwi", "jwo"]


def read_shc_out(shc_out_path: str) -> dict:
    num_corr_points_in_run = NUM_CORR_POINTS * 2 - 1
    coor_array = np.loadtxt(shc_out_path, max_rows=num_corr_points_in_run)
    omega_array = np.loadtxt(shc_out_path, skiprows=num_corr_points_in_run)

    shc = {}
    for label_num, key in enumerate(LABELS_CORR):
        shc[key] = coor_array[:, label_num]
    for label_num, key in enumerate(LABELS_OMEGA):
        shc[key] = omega_array[:, label_num]

    shc["nu"] = shc["omega"] / (2 * np.pi)
    return shc


def calc_spectral_kappa(shc: dict, driving_force: float, temperature: float, volume: float) -> None:
    convert = 1602.17662
    shc["kwi"] = shc["jwi"] * convert / (driving_force * temperature * volume)
    shc["kwo"] = shc["jwo"] * convert / (driving_force * temperature * volume)
    shc["kw"] = shc["kwi"] + shc["kwo"]


def read_Fe_from_runin(direction_index: int = 0, runin_path: str = "run.in") -> float:
    with open(runin_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "compute_hnemd" in s:
            parts = s.split()
            if len(parts) < 4:
                continue
            last_three = parts[-3:]
            try:
                values = [float(x) for x in last_three]
            except ValueError:
                continue
            if not (0 <= direction_index < 3):
                raise ValueError(f"direction_index must be 0/1/2, got {direction_index}")
            return float(values[direction_index])

    raise ValueError("run.in 中未找到包含 compute_hnemd 的行，无法读取 Fe")


def calculate_volume(path: str = "dump.xyz") -> float:
    if path.lower().endswith(".xyz"):
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            _ = file.readline()
            header = file.readline().strip()

        key = 'Lattice="'
        start = header.find(key)
        if start == -1:
            raise ValueError(f"{path} 第二行未找到 Lattice 字段")
        start += len(key)
        end = header.find('"', start)
        if end == -1:
            raise ValueError(f"{path} 第二行 Lattice 字段缺少结束引号")

        lattice_numbers = header[start:end].split()
        if len(lattice_numbers) != 9:
            raise ValueError(f"{path} 第二行 Lattice 字段不是 9 个数: {header[start:end]}")

        mat = np.array([float(x) for x in lattice_numbers], dtype=float).reshape(3, 3)
        return float(abs(np.linalg.det(mat)))

    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        lines = file.readlines()
    if not lines:
        raise ValueError(f"{path} 为空，无法读取体积")

    last_line = lines[-1].strip()
    params = last_line.split()
    try:
        Lx = float(params[-1])
        Ly = float(params[-5])
        Lz = float(params[-9])
    except (IndexError, ValueError) as e:
        raise ValueError(f"{path} 最后一行盒子尺寸字段不足或格式错误: {e}")
    return Lx * Ly * Lz


def quantum_correct_kw(nu: np.ndarray, kw: np.ndarray, temperature: float) -> np.ndarray:
    h = 1.054e-34
    kb = 1.38e-23
    omega = nu * (2 * np.pi)
    x = h * omega / (kb * temperature) * 1e12
    expx = np.exp(x)
    factor = x**2 * expx / ((expx - 1) ** 2)
    return kw * factor


def ensure_sorted(nu: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(nu)
    return nu[order], y[order]


def main() -> None:
    which_position = 0
    temperature = 300.0

    base_dir = os.path.dirname(os.path.abspath(__file__))
    fold_names = ["fold1", "fold2", "fold3"]
    fold_dirs = [os.path.join(base_dir, name) for name in fold_names]

    nu_list = []
    kwc_list = []
    kwq_list = []
    fold_info = []

    for fold_name, fold_dir in zip(fold_names, fold_dirs, strict=True):
        shc_out_path = os.path.join(fold_dir, "shc.out")
        runin_path = os.path.join(fold_dir, "run.in")
        dump_path = os.path.join(fold_dir, "dump.xyz")

        shc = read_shc_out(shc_out_path)
        Fe = read_Fe_from_runin(which_position, runin_path=runin_path)
        V = calculate_volume(dump_path)

        calc_spectral_kappa(shc, driving_force=Fe, temperature=temperature, volume=V)
        nu = np.asarray(shc["nu"], dtype=float)
        kwc = np.asarray(shc["kw"], dtype=float)
        nu, kwc = ensure_sorted(nu, kwc)
        kwq = quantum_correct_kw(nu, kwc, temperature=temperature)

        nu_list.append(nu)
        kwc_list.append(kwc)
        kwq_list.append(kwq)
        fold_info.append((fold_name, Fe, V))

    same_grid = True
    for nu in nu_list[1:]:
        if nu.shape != nu_list[0].shape or not np.allclose(nu, nu_list[0], rtol=0, atol=1e-12):
            same_grid = False
            break

    if same_grid:
        nu_common = nu_list[0]
        kwc_stack = np.vstack(kwc_list)
        kwq_stack = np.vstack(kwq_list)
    else:
        nu_min = max(float(np.min(nu)) for nu in nu_list)
        nu_max = min(float(np.max(nu)) for nu in nu_list)
        nu_common_full = nu_list[0]
        mask = (nu_common_full >= nu_min) & (nu_common_full <= nu_max)
        nu_common = nu_common_full[mask]
        kwc_stack = np.vstack([np.interp(nu_common, nu, kwc) for nu, kwc in zip(nu_list, kwc_list, strict=True)])
        kwq_stack = np.vstack([np.interp(nu_common, nu, kwq) for nu, kwq in zip(nu_list, kwq_list, strict=True)])

    kwc_mean = np.mean(kwc_stack, axis=0)
    kwq_mean = np.mean(kwq_stack, axis=0)

    x_max_data = float(np.nanmax(nu_common))
    x_max = min(50.0, x_max_data) if np.isfinite(x_max_data) else 50.0
    x_max = float(np.ceil(x_max / 5.0) * 5.0) if x_max > 0 else 50.0

    in_view = (nu_common >= 0) & (nu_common <= x_max)
    y_max_data = float(np.nanmax(np.r_[kwc_mean[in_view], kwq_mean[in_view]]))
    y_max = y_max_data * 1.08 if np.isfinite(y_max_data) and y_max_data > 0 else 1.0

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.2), sharex=True, sharey=True, constrained_layout=True)
    axes[0].plot(nu_common, kwc_mean, color="tab:blue", linewidth=2.5)
    axes[0].set_title("Classical (Mean of 3 folds)")

    axes[1].plot(nu_common, kwq_mean, color="tab:red", linewidth=2.5)
    axes[1].set_title("Quantum-corrected (Mean of 3 folds)")

    for ax in axes:
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xlabel(r"$\nu$ (THz)")
        ax.set_ylabel(r"$\kappa(\omega)$ (W/m/K/THz)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    fig_c, ax_c = plt.subplots(1, 1, figsize=(5.2, 4.2), constrained_layout=True)
    ax_c.plot(nu_common, kwc_mean, color="tab:blue", linewidth=2.5)
    ax_c.set_title("Classical (Mean of 3 folds)")
    ax_c.set_xlim(0, x_max)
    ax_c.set_ylim(0, y_max)
    ax_c.set_xlabel(r"$\nu$ (THz)")
    ax_c.set_ylabel(r"$\kappa(\omega)$ (W/m/K/THz)")
    ax_c.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax_c.yaxis.set_major_locator(MaxNLocator(nbins=6))

    fig_q, ax_q = plt.subplots(1, 1, figsize=(5.2, 4.2), constrained_layout=True)
    ax_q.plot(nu_common, kwq_mean, color="tab:red", linewidth=2.5)
    ax_q.set_title("Quantum-corrected (Mean of 3 folds)")
    ax_q.set_xlim(0, x_max)
    ax_q.set_ylim(0, y_max)
    ax_q.set_xlabel(r"$\nu$ (THz)")
    ax_q.set_ylabel(r"$\kappa(\omega)$ (W/m/K/THz)")
    ax_q.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax_q.yaxis.set_major_locator(MaxNLocator(nbins=6))

    k_spec_mean = float(np.trapz(kwc_mean, nu_common))
    k_spec_q_mean = float(np.trapz(kwq_mean, nu_common))

    for fold_name, Fe, V in fold_info:
        print(f"{fold_name}: Fe={Fe}  T={temperature}  V={V}")
    print(f"Mean spectral thermal conductivity (classical) ∫k(ν)dν = {k_spec_mean}")
    print(f"Mean spectral thermal conductivity (quantum)   ∫k_q(ν)dν = {k_spec_q_mean}")

    out_png = os.path.join(os.getcwd(), "shc_spectral_kappa_mean3.png")
    out_pdf = os.path.join(os.getcwd(), "shc_spectral_kappa_mean3.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)

    fig_c.savefig(os.path.join(os.getcwd(), "shc_spectral_kappa_mean3_classical.png"), dpi=300)
    fig_q.savefig(os.path.join(os.getcwd(), "shc_spectral_kappa_mean3_quantum.png"), dpi=300)

    np.savetxt(
        "kv_mean3.out",
        np.c_[nu_common, kwc_mean, kwq_mean],
        header="Frequency (THz), kappa_classical (W/mK/THz), kappa_quantum (W/mK/THz)",
        comments="",
        fmt="%.6e",
    )

    if os.environ.get("SHC_SHOW", "1") == "1":
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig_c)
        plt.close(fig_q)


if __name__ == "__main__":
    main()

