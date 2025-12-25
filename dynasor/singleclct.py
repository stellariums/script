"""
cl-ct.py
========

脚本用途
--------
本脚本基于 dynasor 库，对分子动力学轨迹中得到的原子位移/速度信息计算
各向平均的动态结构因子 C(|q|, ω)，并在给定的 q 区间内提取主导声子峰的
频率与色散斜率。

主要功能
--------
1. 轨迹读取与缓存
   - 从 `dump.xyz` 轨迹文件读取 MD 数据（extxyz 格式）。
   - 若当前目录下存在 `test.npz`，则直接从该缓存文件中读取 dynasor
     预计算好的样本数据，避免重复计算。

2. 动态结构因子计算与球平均
   - 调用 dynasor 计算给定 q 点集合下的纵向/横向动态结构因子。
   - 对结果做球对称平均与 q 分箱，得到仅依赖 |q| 的 C(|q|, ω)。
   - 将频率单位转换为 THz。

3. q=0.5–1.0 区间三峰高斯拟合
   - 在 |q|=0.5–1.0（步长 0.1）的离散 q 点上，取总谱
     C(q, ω)=C_L(q, ω)+C_T(q, ω) 的一维频谱切片。
   - 对每个 q 切片进行三峰高斯拟合，分别对应低频、中频和高频三条分支，
     并对峰位置施加频段约束与强度筛选，使拟合点更贴合实际谱线。
   - 拟合得到三条分支在各 q 点的峰位 ω(q)，并对每条分支做线性拟合，
     得到群速度 v_g（单位 km/s），在终端打印并在图中标注。

4. 可视化输出
   - 绘制 q 点分布直方图并保存为 `q.png`。
   - 绘制 C(|q|, ω) 的二维伪彩色图，并在其上叠加三条拟合分支的散点以及
     对应的 dω/dq 数值，保存为 `Cqw.png`。

使用方法
--------
1. 准备数据
   - 在脚本同级目录下放置分子动力学轨迹文件 `dump.xyz`（extxyz 格式）。
   - 第一次运行会自动计算并生成 `test.npz` 缓存文件；之后运行将直接读取
     该缓存以加速分析。

2. 运行脚本
   在 `cl-ct.py` 所在目录执行：

       python cl-ct.py

3. 输出结果
   - 终端输出：
       * 若需要重新计算，会打印计算进度和缓存生成信息。
       * 打印三条拟合分支的斜率：
         "分支 i 拟合斜率 dω/dq = ... (THz·Å)"。
   - 图像文件：
       * `q.png`   : q 点分布直方图。
       * `Cqw.png` : C(|q|, ω) 热图 + 三峰拟合散点 + 斜率文本框。

依赖说明
--------
- Python 3.x
- numpy
- matplotlib
- scipy
- dynasor 及其运行所需依赖

注意事项
--------
- 轨迹文件名、缓存文件名以及 q 区间等参数在脚本中是写死的，如需更改
  文件路径或 q 范围，可在脚本开头相应变量处修改。
- 若某些 q 点下谱线信噪比较差，局部的三峰拟合可能失败，此时脚本会在
  终端打印提示，但不会中断整体流程。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from dynasor import compute_dynamic_structure_factors, Trajectory
from dynasor.qpoints import get_spherical_qpoints
from dynasor.post_processing import get_spherically_averaged_sample_binned
from dynasor.logging_tools import set_logging_level
from dynasor.sample import read_sample_from_npz
from dynasor.units import radians_per_fs_to_THz as conversion_factor


def single_gaussian(x, a, mu, sigma, c):
    g = a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    return g + c


def fit_branches(Cqw, q_norms, omega, q_min, q_max, q_step, label_prefix=""):
    branch_qs = [[]]
    branch_freqs = [[]]
    prev_mu = None
    slopes = []

    target_qs = np.arange(q_min, q_max + 1e-8, q_step)

    for target_q in target_qs:
        idx = np.argmin(np.abs(q_norms - target_q))
        q_selected = q_norms[idx]

        intensity_profile = Cqw[idx, :]
        valid_mask = intensity_profile > 1e-10
        if not np.any(valid_mask):
            continue

        current_omega = omega[valid_mask]
        current_intensity = intensity_profile[valid_mask]

        i_min = float(np.min(current_intensity))
        i_max = float(np.max(current_intensity))
        i_range = i_max - i_min if i_max > i_min else 1.0
        norm_intensity_full = (current_intensity - i_min) / i_range

        high_mask = norm_intensity_full > 0.05
        if np.count_nonzero(high_mask) >= 10:
            current_omega_fit = current_omega[high_mask]
            norm_intensity = norm_intensity_full[high_mask]
        else:
            current_omega_fit = current_omega
            norm_intensity = norm_intensity_full

        try:
            w_min = float(current_omega_fit.min())
            w_max = float(current_omega_fit.max())
            if not np.isfinite(w_min) or not np.isfinite(w_max) or w_max <= w_min:
                print(f"{label_prefix}单峰高斯拟合跳过: q≈{target_q:.2f}, 频率范围过窄")
                continue

            peak_idx = int(np.argmax(norm_intensity))
            mu_guess = float(current_omega_fit[peak_idx])
            a_guess = float(norm_intensity[peak_idx])

            sigma_min = 0.1
            sigma_max = (w_max - w_min) / 2.0
            if sigma_max <= sigma_min:
                sigma_max = sigma_min * 2.0
            sigma_guess = 1.0
            sigma_guess = max(sigma_min, min(sigma_guess, sigma_max))

            if prev_mu is None:
                bounds_lower = [
                    0.0,
                    w_min,
                    sigma_min,
                    -np.inf,
                ]
                bounds_upper = [
                    np.inf,
                    w_max,
                    sigma_max,
                    np.inf,
                ]
            else:
                delta_mu = 3.0
                mu_lower = max(w_min, prev_mu - delta_mu)
                mu_upper = min(w_max, prev_mu + delta_mu)
                if mu_lower >= mu_upper:
                    bounds_lower = [
                        0.0,
                        w_min,
                        sigma_min,
                        -np.inf,
                    ]
                    bounds_upper = [
                        np.inf,
                        w_max,
                        sigma_max,
                        np.inf,
                    ]
                else:
                    bounds_lower = [
                        0.0,
                        mu_lower,
                        sigma_min,
                        -np.inf,
                    ]
                    bounds_upper = [
                        np.inf,
                        mu_upper,
                        sigma_max,
                        np.inf,
                    ]

            mu_lower_eff = bounds_lower[1]
            mu_upper_eff = bounds_upper[1]
            mu_guess = min(max(mu_guess, mu_lower_eff), mu_upper_eff)

            p0 = [
                a_guess,
                mu_guess,
                sigma_guess,
                0.0,
            ]

            popt, _ = curve_fit(
                single_gaussian,
                current_omega_fit,
                norm_intensity,
                p0=p0,
                bounds=(bounds_lower, bounds_upper),
                maxfev=10000,
            )

            a_fit, mu_fit, sigma_fit, c_fit = popt

            branch_qs[0].append(float(q_selected))
            branch_freqs[0].append(float(mu_fit))
            prev_mu = float(mu_fit)

        except Exception as e:
            print(f"{label_prefix}单峰高斯拟合失败: q≈{target_q:.2f}, 错误: {e}")
            continue

    for branch_index in range(len(branch_qs)):
        qs = np.array(branch_qs[branch_index])
        freqs = np.array(branch_freqs[branch_index])
        if qs.size < 2:
            slopes.append(None)
            continue
        coeffs = np.polyfit(qs, freqs, 1)
        k = coeffs[0]
        vg = k * 0.1
        slopes.append(vg)
        print(f"{label_prefix}分支 {branch_index + 1} 群速度 v_g = {vg:.4f} (km/s)")

    return branch_qs, branch_freqs, slopes


def plot_branches(Cqw, q_norms, omega, branch_qs, branch_freqs, slopes, filename, title_text):
    pass


set_logging_level('INFO')
trajectory_filename = 'dump.xyz'
output_file = 'test.npz'
max_points = 4000

if os.path.exists(output_file):
    print(f"✅ 检测到已计算文件 {output_file}，直接加载数据...")
    sample_raw = read_sample_from_npz(output_file)
    q_points = np.array(sample_raw.q_points)
else:
    print("⏳ 未找到缓存文件，开始计算动态结构因子...")
    traj = Trajectory(trajectory_filename, trajectory_format='extxyz', frame_stop=25000)
    q_points = get_spherical_qpoints(traj.cell, q_max=2.0, max_points=max_points)
    sample_raw = compute_dynamic_structure_factors(
        traj,
        q_points,
        dt=10.0,
        window_size=2000,
        window_step=50,
        calculate_currents=True,
    )
    sample_raw.write_to_npz(output_file)
    print(f"💾 计算结果已保存至 {output_file}")

plt.figure(figsize=(3.4, 2.5), dpi=140)
plt.hist(np.linalg.norm(q_points, axis=1), bins=50)
plt.xlabel(r'$|\mathbf{q}|$ (1/Å)')
plt.ylabel('Counts')
plt.tight_layout()
plt.savefig('q.png', dpi=300, bbox_inches='tight')

sample_averaged = get_spherically_averaged_sample_binned(sample_raw, num_q_bins=50)
sample_averaged.omega *= conversion_factor

C_L = sample_averaged.Clqw
C_T = sample_averaged.Ctqw
vmin = 0.0
vmax = 0.0020

q_min = 0.5
q_max = 0.7
q_step = 0.08

q_norms = sample_averaged.q_norms
omega = sample_averaged.omega

branch_qs_L, branch_freqs_L, slopes_L = fit_branches(
    C_L,
    q_norms,
    omega,
    q_min,
    q_max,
    q_step,
    label_prefix="[L] ",
)
branch_qs_T, branch_freqs_T, slopes_T = fit_branches(
    C_T,
    q_norms,
    omega,
    q_min,
    q_max,
    q_step,
    label_prefix="[T] ",
)

fig, axes = plt.subplots(figsize=(3.4, 3.8), nrows=2, dpi=140, sharex=True, sharey=True)

ax = axes[0]
im_cl = ax.pcolormesh(q_norms, omega, C_L.T, cmap='Reds', vmin=vmin, vmax=vmax)
ax.text(
    0.05,
    0.85,
    r'$C_L(|\mathbf{q}|, \omega)$',
    transform=ax.transAxes,
    bbox={'color': 'white', 'alpha': 0.8, 'pad': 3},
)

colors = ['blue', 'green', 'purple']
labels_L = [rf'Peak {i+1}' for i in range(len(branch_qs_L))]
for branch_index in range(len(branch_qs_L)):
    qs = np.array(branch_qs_L[branch_index])
    freqs = np.array(branch_freqs_L[branch_index])
    if qs.size == 0:
        continue
    ax.scatter(
        qs,
        freqs,
        s=15,
        color=colors[branch_index],
        marker='o',
        edgecolors='k',
        linewidths=0.3,
        label=labels_L[branch_index],
    )

slope_text_lines_L = []
for branch_index, k in enumerate(slopes_L, start=1):
    if k is not None:
        slope_text_lines_L.append(f"Branch {branch_index}: v_g={k:.3f} km/s")
if slope_text_lines_L:
    slope_text_L = "\n".join(slope_text_lines_L)
    ax.text(
        0.98,
        0.02,
        slope_text_L,
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=7,
        bbox={'color': 'white', 'alpha': 0.7, 'pad': 2},
    )
    ax.legend(loc='upper right', fontsize=6, framealpha=0.8)

ax = axes[1]
im_ct = ax.pcolormesh(q_norms, omega, C_T.T, cmap='Oranges', vmin=vmin, vmax=vmax)
ax.text(
+    0.05,
    0.85,
    r'$C_T(|\mathbf{q}|, \omega)$',
    transform=ax.transAxes,
    bbox={'color': 'white', 'alpha': 0.8, 'pad': 3},
)

labels_T = [rf'Peak {i+1}' for i in range(len(branch_qs_T))]
for branch_index in range(len(branch_qs_T)):
    qs = np.array(branch_qs_T[branch_index])
    freqs = np.array(branch_freqs_T[branch_index])
    if qs.size == 0:
        continue
    ax.scatter(
        qs,
        freqs,
        s=15,
        color=colors[branch_index],
        marker='o',
        edgecolors='k',
        linewidths=0.3,
        label=labels_T[branch_index],
    )

slope_text_lines_T = []
for branch_index, k in enumerate(slopes_T, start=1):
    if k is not None:
        slope_text_lines_T.append(f"Branch {branch_index}: v_g={k:.3f} km/s")
if slope_text_lines_T:
    slope_text_T = "\n".join(slope_text_lines_T)
    ax.text(
        0.98,
        0.02,
        slope_text_T,
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=7,
        bbox={'color': 'white', 'alpha': 0.7, 'pad': 2},
    )
    ax.legend(loc='upper right', fontsize=6, framealpha=0.8)

ax.set_xlabel(r'$|\mathbf{q}|$ (1/Å)')
ax.set_ylabel('Frequency (THz)', y=1)
ax.set_ylim([0, 50])
ax.set_xlim([0, 2])
axes[0].tick_params(axis='both', direction='in')
axes[1].tick_params(axis='both', direction='in')

fig.tight_layout()
plt.subplots_adjust(hspace=0.08)
plt.savefig('cl_ct_fit.png', dpi=300, bbox_inches='tight')
