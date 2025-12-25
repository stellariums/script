"""
gv-line.py
-----------

脚本用途：
    - 基于 dynasor 读取 / 计算各向平均的动态结构因子 C_L、C_T
    - 在小 |q| 区间内提取主峰位置，做线性拟合得到纵向 / 横向群速度
    - 生成包含拟合直线与纯热图的双列光谱图

使用方法：
    - 在包含 `dump.xyz` 或 `test.npz` 的上级目录中运行：
          python gv-line.py
    - 若上级目录存在 `test.npz`，脚本直接读取该缓存；
      否则会从 `dump.xyz` 重新计算并写入缓存

输出文件：
    - dual_spectra_with_fits.png : 左列为带线性拟合曲线的 C_L / C_T 热图，
                                  右列为对应的纯热图
    - 终端会打印纵向与横向声速 v_g（单位 km/s）
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
import matplotlib as mpl

# 设置绘图使用的全局字体与字号
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 9

# 配置日志等级与输入 / 输出路径
set_logging_level('INFO')
trajectory_filename = '../dump.xyz'
output_file = '../test.npz'
max_points = 8000

# 数据加载：优先从缓存 npz 读取，缺失时从轨迹重新计算
if os.path.exists(output_file):
    print(f"✅ 检测到已计算文件 {output_file}，直接加载数据...")
    # 使用 dynasor 专用加载方法
    sample_raw = read_sample_from_npz(output_file)  # 返回 Sample 对象
    q_points = np.array(sample_raw.q_points)
else:
    print("⏳ 未找到缓存文件，开始计算动态结构因子...")
    traj = Trajectory(trajectory_filename, trajectory_format='extxyz', frame_stop=None)
    q_points = get_spherical_qpoints(traj.cell, q_max=0.4, max_points=max_points)
    sample_raw = compute_dynamic_structure_factors(
        traj, q_points, 
        dt=25.0, 
        window_size=2000,
        window_step=50,
        calculate_currents=True
    )
    sample_raw.write_to_npz(output_file)
    print(f"💾 计算结果已保存至 {output_file}")

# 数据预处理：球对称平均 + q 分箱
sample_averaged = get_spherically_averaged_sample_binned(sample_raw, num_q_bins=80)
sample_averaged.omega *= conversion_factor  # 转换为THz

# 设置统一色标上限，便于不同子图对比
vmax = 0.0015
q_norms = sample_averaged.q_norms
omega = sample_averaged.omega
Clqw = sample_averaged.Clqw
Ctqw = sample_averaged.Ctqw

# 计算群速度：在低 q 区间用线性模型拟合主峰位置
def linear_model(q, v_g):
    return v_g * q

# 计算纵向(CL)群速度
q_low_L, omega_low_L = [], []
for i, q in enumerate(q_norms):
    if q < 0.6:  # 仅使用低q区域数据
        idx = np.argmax(Clqw[i, :])
        if 0 < omega[idx] < 30.0:
            q_low_L.append(q)
            omega_low_L.append(omega[idx])
            
if q_low_L:
    params_L, _ = curve_fit(linear_model, q_low_L, omega_low_L)
    v_g_L = params_L[0] * 0.1  # 转换为km/s
else:
    v_g_L = 0.0

# 计算横向(CT)群速度
q_low_T, omega_low_T = [], []
for i, q in enumerate(q_norms):
    if q < 0.6:  # 仅使用低q区域数据
        idx = np.argmax(Ctqw[i, :])
        if 0 < omega[idx] < 30.0:
            q_low_T.append(q)
            omega_low_T.append(omega[idx])
            
if q_low_T:
    params_T, _ = curve_fit(linear_model, q_low_T, omega_low_T)
    v_g_T = params_T[0] * 0.1  # 转换为km/s
else:
    v_g_T = 0.0

# ===================== 创建双列热图 =====================
fig = plt.figure(figsize=(7, 6.5), dpi=300)
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                      wspace=0.15, hspace=0.15, left=0.1, right=0.95,
                      top=0.95, bottom=0.1)

# 左侧热图 (含拟合曲线)
ax1 = fig.add_subplot(gs[0, 0])  # 左上 - CL
ax2 = fig.add_subplot(gs[1, 0])  # 左下 - CT

# 右侧热图 (纯热图)
ax3 = fig.add_subplot(gs[0, 1])  # 右上 - CL
ax4 = fig.add_subplot(gs[1, 1])  # 右下 - CT

# ===== 左侧面板 (含拟合曲线) =====
# CL纵向 (左上)
im1 = ax1.pcolormesh(q_norms, omega, Clqw.T, 
                     cmap='Reds', shading='nearest',
                     vmin=0, vmax=vmax)
ax1.set_title(r'$C_L(|\mathbf{q}|, \omega)$', pad=5)
ax1.set_ylabel('Frequency (THz)', labelpad=5)
ax1.set_ylim(0, 50)  # 与右侧保持一致
ax1.set_xlim(0, 2.0)  # 完整范围

# 添加纵向拟合曲线
if q_low_L:
    q_fit = np.linspace(0, max(q_norms), 50)
    omega_fit_L = linear_model(q_fit, v_g_L/0.1)  # 转换回THz单位
    ax1.plot(q_fit, omega_fit_L, 'b--', lw=1.5, alpha=0.8)
    ax1.text(0.05, 0.9, f'$v_g^L$ = {v_g_L:.2f} km/s', 
             transform=ax1.transAxes, color='b', fontsize=9,
             bbox=dict(facecolor='white', alpha=0.7, pad=2))

# CT横向 (左下)
im2 = ax2.pcolormesh(q_norms, omega, Ctqw.T, 
                     cmap='Oranges', shading='nearest',
                     vmin=0, vmax=vmax)
ax2.set_title(r'$C_T(|\mathbf{q}|, \omega)$', pad=5)
ax2.set_xlabel(r'$|\mathbf{q}|$ (1/Å)', labelpad=5)
ax2.set_ylabel('Frequency (THz)', labelpad=5)
ax2.set_ylim(0, 50)  # 与右侧保持一致
ax2.set_xlim(0, 2.0)  # 完整范围

# 添加横向拟合曲线
if q_low_T:
    q_fit = np.linspace(0, max(q_norms), 50)
    omega_fit_T = linear_model(q_fit, v_g_T/0.1)  # 转换回THz单位
    ax2.plot(q_fit, omega_fit_T, 'b--', lw=1.5, alpha=0.8)
    ax2.text(0.05, 0.9, f'$v_g^T$ = {v_g_T:.2f} km/s', 
             transform=ax2.transAxes, color='b', fontsize=9,
             bbox=dict(facecolor='white', alpha=0.7, pad=2))

# ===== 右侧面板 (纯热图) =====
# CL纵向 (右上)
im3 = ax3.pcolormesh(q_norms, omega, Clqw.T, 
                     cmap='Reds', shading='nearest',
                     vmin=0, vmax=vmax)
ax3.set_title(r'$C_L(|\mathbf{q}|, \omega)$', pad=5)
ax3.set_ylim(0, 50)
ax3.set_xlim(0, 2.0)

# CT横向 (右下)
im4 = ax4.pcolormesh(q_norms, omega, Ctqw.T, 
                     cmap='Oranges', shading='nearest',
                     vmin=0, vmax=vmax)
ax4.set_title(r'$C_T(|\mathbf{q}|, \omega)$', pad=5)
ax4.set_xlabel(r'$|\mathbf{q}|$ (1/Å)', labelpad=5)
ax4.set_ylim(0, 50)
ax4.set_xlim(0, 2.0)

# 添加色条
cax = fig.add_axes([0.96, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
cbar.set_label('Intensity (a.u.)', labelpad=8)

# 保存结果
plt.savefig('dual_spectra_with_fits.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ 结果图已保存，纵向声速: {v_g_L:.2f} km/s, 横向声速: {v_g_T:.2f} km/s")
