"""
用途
----
将 5 套体系的量子修正谱热导率曲线 κ_q(ν) 叠加到同一张图上进行对比，并用不同颜色区分：
- TpPa-COF
- COF-2F
- COF-4F

数据来源
--------
优先读取每个体系目录下由 shc_3.py 生成的 kv_mean3.out（包含 nu、classical、quantum 三列）。
该方式是“按数据重绘”，比直接拼接 PNG 更准确。

使用方法
--------
在本脚本所在目录运行：

    python shc_quantum_compare_5.py

输出文件（生成在“运行时的当前工作目录”）
------------------------------------
- shc_spectral_kappa_quantum_compare_5.png
- shc_spectral_kappa_quantum_compare_5.pdf

注意事项
--------
- 若某个体系的 kv_mean3.out 不存在或格式不匹配，将抛出异常并停止。
- 默认绘图横轴范围为 0–50 THz；若数据最大频率不足 50 THz，曲线会在其最大频率处结束。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def read_kv_mean3(kv_path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(kv_path, skiprows=1)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"{kv_path} 不是至少三列的数值表格")
    nu = np.asarray(data[:, 0], dtype=float)
    kw_quantum = np.asarray(data[:, 2], dtype=float)
    order = np.argsort(nu)
    return nu[order], kw_quantum[order]


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    series = [
        ("TpPa-COF", os.path.join(here, "kv_mean3.out")),
        ("COF-2F", os.path.join(here, "..", "..", "..", "COF2F", "kv_mean3.out")),
        ("COF-4F", os.path.join(here, "..", "COF4F", "kv_mean3.out")),
    ]
    series = [(name, os.path.abspath(path)) for name, path in series]

    colors = ["red", "gold", "blue"]

    curves = []
    for (name, kv_path), color in zip(series, colors, strict=True):
        if not os.path.exists(kv_path):
            raise FileNotFoundError(f"未找到 {name} 的 kv_mean3.out：{kv_path}")
        nu, kwq = read_kv_mean3(kv_path)
        curves.append((name, nu, kwq, color))

    x_max = 50.0
    y_max_data = float(
        np.nanmax(
            np.concatenate([kwq[(nu >= 0) & (nu <= x_max)] for _, nu, kwq, _ in curves if kwq.size > 0])
        )
    )
    y_max = y_max_data * 1.08 if np.isfinite(y_max_data) and y_max_data > 0 else 1.0

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.4), constrained_layout=True)
    for name, nu, kwq, color in curves:
        ax.plot(nu, kwq, linewidth=2.2, color=color, label=name)

    ax.set_title("Quantum-corrected Spectral Thermal Conductivity")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel(r"$\nu$ (THz)")
    ax.set_ylabel(r"$\kappa_q(\omega)$ (W/m/K/THz)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(frameon=False)

    for name, nu, kwq, _ in curves:
        kq = float(np.trapz(kwq, nu))
        print(f"{name}: ∫k_q(ν)dν = {kq}")

    out_png = os.path.join(os.getcwd(), "shc_spectral_kappa_quantum_compare_5.png")
    out_pdf = os.path.join(os.getcwd(), "shc_spectral_kappa_quantum_compare_5.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)

    if os.environ.get("SHC_SHOW", "1") == "1":
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()

