"""
scatter.py
-----------

脚本用途：
    - 基于 dynasor 计算并绘制纵向 / 横向动态结构因子 C_L、C_T 的热图
    - 在给定 |q| 区间内对主色散分支做单峰高斯拟合，并叠加拟合点和误差棒

使用方法：
    - 在包含 `dump.xyz` 或 `test.npz` 的目录运行：
          python scatter.py
    - 若当前目录存在 `test.npz`，脚本直接读取缓存样本；
      若不存在，则从 `dump.xyz` 重新计算动态结构因子并写入缓存

输出文件：
    - q.png         : q 点模长分布直方图
    - cl_ct_fit.png : C_L / C_T 热图叠加拟合峰位置与误差棒
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
    branch_errs = [[]]
    prev_mus = [None]

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

        freq_min = 20.0
        freq_mask = current_omega >= freq_min
        if not np.any(freq_mask):
            continue
        current_omega = current_omega[freq_mask]
        current_intensity = current_intensity[freq_mask]

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

            peak_idx = int(np.argmax(norm_intensity))
            mu_guess = float(current_omega_fit[peak_idx])
            a_guess = float(norm_intensity[peak_idx])

            sigma_min = 0.1
            sigma_max = (w_max - w_min) / 2.0 if w_max > w_min else 10.0
            sigma_guess = 1.0

            p0 = [
                a_guess, mu_guess, sigma_guess,
                0.0,
            ]

            if prev_mus[0] is None:
                bounds_lower = [
                    0.0, w_min, sigma_min,
                    -np.inf,
                ]
                bounds_upper = [
                    np.inf, w_max, sigma_max,
                    np.inf,
                ]
            else:
                delta_mu = 3.0
                mu_lower = max(w_min, prev_mus[0] - delta_mu)
                mu_upper = min(w_max, prev_mus[0] + delta_mu)
                if mu_lower >= mu_upper:
                    bounds_lower = [
                        0.0, w_min, sigma_min,
                        -np.inf,
                    ]
                    bounds_upper = [
                        np.inf, w_max, sigma_max,
                        np.inf,
                    ]
                else:
                    bounds_lower = [
                        0.0, mu_lower, sigma_min,
                        -np.inf,
                    ]
                    bounds_upper = [
                        np.inf, mu_upper, sigma_max,
                        np.inf,
                    ]

            popt, pcov = curve_fit(
                single_gaussian,
                current_omega_fit,
                norm_intensity,
                p0=p0,
                bounds=(bounds_lower, bounds_upper),
                maxfev=10000,
            )

            a_fit, mu_fit, sigma_fit, c_fit = popt

            if mu_fit < freq_min:
                continue

            mu_err = float(abs(sigma_fit)) if np.isfinite(sigma_fit) else np.nan
            scale_factor = 2.0
            if not np.isnan(mu_err):
                mu_err *= scale_factor

            branch_qs[0].append(float(q_selected))
            branch_freqs[0].append(float(mu_fit))
            branch_errs[0].append(float(mu_err))
            prev_mus[0] = float(mu_fit)

        except Exception as e:
            print(f"{label_prefix}单峰高斯拟合失败: q≈{target_q:.2f}, 错误: {e}")
            continue

    return branch_qs, branch_freqs, branch_errs


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
        traj, q_points, 
        dt=10.0, 
        window_size=2000,
        window_step=50,
        calculate_currents=True
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

q_min = 1.5
q_max = 2.0
q_step = 0.05

q_norms = sample_averaged.q_norms
omega = sample_averaged.omega

branch_qs_L, branch_freqs_L, branch_errs_L = fit_branches(C_L, q_norms, omega, q_min, q_max, q_step, label_prefix="[L] ")
branch_qs_T, branch_freqs_T, branch_errs_T = fit_branches(C_T, q_norms, omega, q_min, q_max, q_step, label_prefix="[T] ")

fig, axes = plt.subplots(figsize=(3.4, 3.8), nrows=2, dpi=140, sharex=True, sharey=True)

ax = axes[0]
im_cl = ax.pcolormesh(q_norms, omega, C_L.T, cmap='Reds', vmin=vmin, vmax=vmax)
ax.text(0.05, 0.85, r'$C_L(|\mathbf{q}|, \omega)$', transform=ax.transAxes,
        bbox={'color': 'white', 'alpha': 0.8, 'pad': 3})

colors = ['black', 'black', 'black']
for branch_index in range(len(branch_qs_L)):
    qs = np.array(branch_qs_L[branch_index])
    freqs = np.array(branch_freqs_L[branch_index])
    if qs.size == 0:
        continue
    errs = np.array(branch_errs_L[branch_index])
    mask = freqs >= 20.0
    qs = qs[mask]
    freqs = freqs[mask]
    errs = errs[mask]
    if qs.size == 0:
        continue
    ax.errorbar(
        qs,
        freqs,
        yerr=errs,
        fmt='o',
        ms=3.5,
        color=colors[branch_index],
        ecolor=colors[branch_index],
        elinewidth=0.6,
        capsize=2,
        markeredgecolor='k',
        markeredgewidth=0.3,
        linestyle='none',
    )

ax = axes[1]
im_ct = ax.pcolormesh(q_norms, omega, C_T.T, cmap='Oranges', vmin=vmin, vmax=vmax)
ax.text(0.05, 0.85, r'$C_T(|\mathbf{q}|, \omega)$', transform=ax.transAxes,
        bbox={'color': 'white', 'alpha': 0.8, 'pad': 3})

for branch_index in range(len(branch_qs_T)):
    qs = np.array(branch_qs_T[branch_index])
    freqs = np.array(branch_freqs_T[branch_index])
    if qs.size == 0:
        continue
    errs = np.array(branch_errs_T[branch_index])
    mask = freqs >= 20.0
    qs = qs[mask]
    freqs = freqs[mask]
    errs = errs[mask]
    if qs.size == 0:
        continue
    ax.errorbar(
        qs,
        freqs,
        yerr=errs,
        fmt='o',
        ms=3.5,
        color=colors[branch_index],
        ecolor=colors[branch_index],
        elinewidth=0.6,
        capsize=2,
        markeredgecolor='k',
        markeredgewidth=0.3,
        linestyle='none',
    )

ax.set_xlabel(r'$|\mathbf{q}|$ (1/Å)')
ax.set_ylabel('Frequency (THz)', y=1)
ax.set_ylim([0, 50])
ax.set_xlim([0, 2])
axes[0].tick_params(axis='both', direction='in')
axes[1].tick_params(axis='both', direction='in')

fig.tight_layout()
plt.subplots_adjust(hspace=0.08)
plt.savefig('cl_ct_fit.png', dpi=300, bbox_inches='tight')


