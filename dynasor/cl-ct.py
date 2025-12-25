"""
cl-ct.py
========

脚本用途
--------
基于 dynasor 计算各向平均的动态结构因子 C(|q|, ω)，并在给定 |q| 区间内
用多峰高斯拟合提取主导声子分支的色散关系。

主要功能
--------
1. 轨迹读取与缓存
   - 从 `dump.xyz` 轨迹文件读取 MD 数据（extxyz 格式）。
   - 若当前目录存在 `test.npz`，则直接读取 dynasor 预计算样本以避免重复计算。

2. 动态结构因子计算与球平均
   - 调用 dynasor 计算给定 q 点集合下的纵向 / 横向动态结构因子。
   - 对结果做球对称平均与 q 分箱，得到仅依赖 |q| 的 C(|q|, ω)。
   - 将频率单位转换为 THz。

3. q=0.5–0.7 区间多峰高斯拟合
   - 在 |q|=0.5–0.7（步长 0.1）的离散 q 点上，取 C_L 与 C_T 的谱线切片。
   - 对每个 q 切片进行多峰高斯拟合，分别追踪不同色散分支的峰位 ω(q)。
   - 对提取出的分支做线性拟合，得到群速度 v_g（单位 km/s），并打印在终端。

4. 可视化输出
   - 绘制 q 点分布直方图，保存为 `q.png`。
   - 绘制 C(|q|, ω) 的二维伪彩色图，并叠加拟合分支散点与斜率文本，保存为
     `Cqw.png`。

使用方法
--------
1. 准备数据
   - 在脚本同级目录下放置分子动力学轨迹文件 `dump.xyz`（extxyz 格式）。
   - 首次运行会自动计算并生成 `test.npz` 缓存文件；之后运行将优先读取缓存。

2. 运行脚本
   在 `cl-ct.py` 所在目录执行：

       python cl-ct.py

3. 输出结果
   - 终端输出：
       * 若需要重新计算，会打印计算进度和缓存生成信息。
       * 打印各分支的拟合斜率：
         "分支 i 拟合斜率 dω/dq = ... (THz·Å)"。
   - 图像文件：
       * `q.png`   : q 点分布直方图。
       * `Cqw.png` : C(|q|, ω) 热图 + 拟合散点 + 斜率文本框。
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
from dynasor.units import radians_per_fs_to_THz as conversion_factor # conversion from 1/fs to meV


def double_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2, c):
    g1 = a1 * np.exp(-(x - mu1) ** 2 / (2 * sigma1 ** 2))
    g2 = a2 * np.exp(-(x - mu2) ** 2 / (2 * sigma2 ** 2))
    return g1 + g2 + c


def fit_branches(Cqw, q_norms, omega, q_min, q_max, q_step, label_prefix=""):
    branch_qs = [[], []]
    branch_freqs = [[], []]
    prev_mus = [None, None]
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
            edges = np.linspace(w_min, w_max, 3)

            mu_guesses = []
            a_guesses = []
            for seg_idx in range(2):
                seg_mask = (current_omega_fit >= edges[seg_idx]) & (current_omega_fit < edges[seg_idx + 1])
                if np.any(seg_mask):
                    local_idx = np.argmax(norm_intensity[seg_mask])
                    global_indices = np.where(seg_mask)[0]
                    peak_idx = global_indices[local_idx]
                else:
                    peak_idx = int(np.argmax(norm_intensity))
                mu_guesses.append(float(current_omega_fit[peak_idx]))
                a_guesses.append(float(norm_intensity[peak_idx]))

            sigma_min = 0.1
            sigma_max = (w_max - w_min) / 2.0 if w_max > w_min else 10.0
            sigma_guess = 1.0

            p0 = [
                a_guesses[0], mu_guesses[0], sigma_guess,
                a_guesses[1], mu_guesses[1], sigma_guess,
                0.0,
            ]

            if prev_mus[0] is None or prev_mus[1] is None:
                bounds_lower = [
                    0.0, edges[0], sigma_min,
                    0.0, edges[1], sigma_min,
                    -np.inf,
                ]
                bounds_upper = [
                    np.inf, edges[1], sigma_max,
                    np.inf, edges[2], sigma_max,
                    np.inf,
                ]
            else:
                delta_mu = 3.0
                mu1_lower = max(w_min, prev_mus[0] - delta_mu)
                mu1_upper = min(w_max, prev_mus[0] + delta_mu)
                mu2_lower = max(w_min, prev_mus[1] - delta_mu)
                mu2_upper = min(w_max, prev_mus[1] + delta_mu)
                if mu1_lower >= mu1_upper or mu2_lower >= mu2_upper:
                    bounds_lower = [
                        0.0, edges[0], sigma_min,
                        0.0, edges[1], sigma_min,
                        -np.inf,
                    ]
                    bounds_upper = [
                        np.inf, edges[1], sigma_max,
                        np.inf, edges[2], sigma_max,
                        np.inf,
                    ]
                else:
                    bounds_lower = [
                        0.0, mu1_lower, sigma_min,
                        0.0, mu2_lower, sigma_min,
                        -np.inf,
                    ]
                    bounds_upper = [
                        np.inf, mu1_upper, sigma_max,
                        np.inf, mu2_upper, sigma_max,
                        np.inf,
                    ]

            popt, _ = curve_fit(
                double_gaussian,
                current_omega_fit,
                norm_intensity,
                p0=p0,
                bounds=(bounds_lower, bounds_upper),
                maxfev=10000,
            )

            a1, mu1, sigma1, a2, mu2, sigma2, c_fit = popt

            peak_candidates = [(mu1, a1, sigma1), (mu2, a2, sigma2)]

            if prev_mus[0] is None or prev_mus[1] is None:
                peaks = sorted(peak_candidates, key=lambda x: x[0])
            else:
                assigned = [None, None]
                used = set()
                for branch_index in range(2):
                    prev_mu = prev_mus[branch_index]
                    distances = []
                    for idx, (mu_peak, _, _) in enumerate(peak_candidates):
                        if idx in used:
                            continue
                        distances.append((abs(mu_peak - prev_mu), idx))
                    if not distances:
                        continue
                    distances.sort(key=lambda x: x[0])
                    _, best_idx = distances[0]
                    assigned[branch_index] = peak_candidates[best_idx]
                    used.add(best_idx)
                for idx, peak in enumerate(assigned):
                    if peak is None:
                        remaining = [p for i, p in enumerate(peak_candidates) if i not in used]
                        if remaining:
                            assigned[idx] = remaining[0]
                            used.add(peak_candidates.index(remaining[0]))
                peaks = assigned

            for branch_index, (mu_peak, _, _) in enumerate(peaks):
                if mu_peak is None:
                    continue
                branch_qs[branch_index].append(float(q_selected))
                branch_freqs[branch_index].append(float(mu_peak))
                prev_mus[branch_index] = float(mu_peak)

        except Exception as e:
            print(f"{label_prefix}三峰高斯拟合失败: q≈{target_q:.2f}, 错误: {e}")
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
output_file = 'test.npz'  # 数据保存路径
max_points=4000
# ===================== 修复后的数据加载逻辑 =====================
if os.path.exists(output_file):
    print(f"✅ 检测到已计算文件 {output_file}，直接加载数据...")
    # 使用 dynasor 专用加载方法
    sample_raw = read_sample_from_npz(output_file)  # 返回 Sample 对象
    q_points = np.array(sample_raw.q_points)
else:
    print("⏳ 未找到缓存文件，开始计算动态结构因子...")
    traj = Trajectory(trajectory_filename, trajectory_format='extxyz', frame_stop=25000)
    q_points = get_spherical_qpoints(traj.cell, q_max=2.0, max_points=max_points)
    sample_raw = compute_dynamic_structure_factors(
        traj, q_points, 
        dt=10.0, 
        window_size=2000,
        window_step=50,
        calculate_currents=True
    )
    sample_raw.write_to_npz(output_file)
    print(f"💾 计算结果已保存至 {output_file}")

# ===================== 数据预处理 =====================
# 检查q点分布
plt.figure(figsize=(3.4, 2.5), dpi=140)
plt.hist(np.linalg.norm(q_points, axis=1), bins=50)
plt.xlabel(r'$|\mathbf{q}|$ (1/Å)')
plt.ylabel('Counts')
plt.tight_layout()
plt.savefig('q.png', dpi=300, bbox_inches='tight')

# 球对称平均与分箱处理 (提高小q分辨率)###################################
sample_averaged = get_spherically_averaged_sample_binned(sample_raw, num_q_bins=50)
sample_averaged.omega *= conversion_factor
# ===== 不分横纵波：C(q,w) =====
C_L = sample_averaged.Clqw
C_T = sample_averaged.Ctqw
vmin = 0.0
vmax = 0.0020

# ===================== q=0.5–1.0 区间三峰高斯拟合 =====================
# 仅在指定的 q 区间内 (0.5–1.0, 步长 0.1) 做三峰高斯拟合
q_min = 0.5
q_max = 0.7
q_step = 0.08

q_norms = sample_averaged.q_norms
omega = sample_averaged.omega

branch_qs_L, branch_freqs_L, slopes_L = fit_branches(C_L, q_norms, omega, q_min, q_max, q_step, label_prefix="[L] ")
branch_qs_T, branch_freqs_T, slopes_T = fit_branches(C_T, q_norms, omega, q_min, q_max, q_step, label_prefix="[T] ")

fig, axes = plt.subplots(figsize=(3.4, 3.8), nrows=2, dpi=140, sharex=True, sharey=True)

ax = axes[0]
im_cl = ax.pcolormesh(q_norms, omega, C_L.T, cmap='Reds', vmin=vmin, vmax=vmax)
ax.text(0.05, 0.85, r'$C_L(|\mathbf{q}|, \omega)$', transform=ax.transAxes,
        bbox={'color': 'white', 'alpha': 0.8, 'pad': 3})

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
ax.text(0.05, 0.85, r'$C_T(|\mathbf{q}|, \omega)$', transform=ax.transAxes,
        bbox={'color': 'white', 'alpha': 0.8, 'pad': 3})

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
# plt.show()


