"""
群速度计算与可视化工具 (gv-line.py)
=====================================

简介 (Introduction)
-------------------
本脚本用于计算并可视化晶格动力学中的动态结构因子 (Dynamic Structure Factor, DSF) 以及声子群速度。
它基于 `dynasor` 库读取分子动力学模拟轨迹，计算纵向 (Longitudinal) 和横向 (Transverse) 的声子谱，
并通过有限差分法估算群速度。

主要功能 (Features)
-------------------
1. 计算/加载 DSF:
   - 从 .xyz 轨迹文件计算动态结构因子。
   - 或直接加载已缓存的 .npz 结果（加速后续分析）。
2. 数据处理:
   - 进行球对称平均和 q-bin 分箱处理。
3. 群速度分析:
   - 计算加权平均频率及其标准差（误差）。
   - 使用有限差分法 (v_g = dω/dq) 计算群速度。
   - 统计群速度的平均值和标准差。
4. 高级可视化:
   - 生成包含热图 (Heatmap) 和散点图 (Scatter Plot) 的组合图表。
   - 热图显示声子谱密度 (CL 和 CT)，叠加加权平均频率点及误差棒。
   - 散点图显示群速度随频率的变化，并标注统计信息。
   - 支持响应式布局、黄金分割比例画布以及防遮挡标题设计。

依赖库 (Dependencies)
---------------------
- numpy
- matplotlib
- scipy
- dynasor

使用方法 (Usage)
----------------
1. 基本运行 (默认 q_interval=0.1):
   $ python gv-line.py

2. 自定义 q 间隔参数 (例如 0.05):
   $ python gv-line.py --q 0.05
   (更小的 q 间隔可以提供更精细的 q 点分辨率)

核心算法 (Algorithms)
---------------------
1. 加权平均频率 (Weighted Avg Frequency):
   对每个 q bin 计算强度加权的平均频率和标准差：
   ω_avg = Σ(I * ω) / ΣI
   σ = sqrt( Σ(I * (ω - ω_avg)^2) / ΣI )

2. 群速度 (Group Velocity):
   采用相邻点有限差分法：
   v_g = (ω_{i+1} - ω_i) / (q_{i+1} - q_i)
   (单位已转换为 km/s)

输出 (Output)
-------------
- 缓存文件: ../test.npz (存储计算好的 DSF 数据)
- 结果图片: dual_spectra_with_vg_scatter.png (包含四张子图的综合分析图)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import curve_fit
from dynasor import compute_dynamic_structure_factors, Trajectory
from dynasor.qpoints import get_spherical_qpoints
from dynasor.post_processing import get_spherically_averaged_sample_binned
from dynasor.logging_tools import set_logging_level
from dynasor.sample import read_sample_from_npz
from dynasor.units import radians_per_fs_to_THz as conversion_factor
import matplotlib as mpl

# 设置全局字体
# 优先使用 Arial，如果不可用则使用 sans-serif (Linux系统通常会自动回退到 DejaVu Sans)
try:
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
except Exception:
    pass
mpl.rcParams['font.size'] = 9

# 配置日志和文件路径
# 将日志级别设置为 ERROR 以屏蔽非致命的 WARNING 信息
set_logging_level('ERROR')

# 解析命令行参数
parser = argparse.ArgumentParser(description='Calculate group velocity from DSF.')
parser.add_argument('--q', type=float, default=0.1, help='q-interval for binning (default: 0.1)')
args = parser.parse_args()

q_interval = args.q
print(f"🔧 使用 q 间隔: {q_interval}")

trajectory_filename = '../dump.xyz'
output_file = '../test.npz'
max_points = 2000

# ===================== 数据加载 =====================
if os.path.exists(output_file):
    print(f"✅ 检测到已计算文件 {output_file}，直接加载数据...")
    sample_raw = read_sample_from_npz(output_file)
    # 如果直接加载了数据，后续处理直接使用 sample_raw 即可
    # 不需要再次读取 traj 或计算 q_points，除非你需要它们做额外的验证
    # 原代码这里读取traj只是为了获得cell计算q_points，但如果下面直接用sample_averaged，其实不需要重算q_points
    print("✅ 数据加载完成，跳过轨迹读取。")
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

# ===================== 数据预处理 =====================
# 球对称平均与分箱处理
sample_averaged = get_spherically_averaged_sample_binned(sample_raw, num_q_bins=100)
sample_averaged.omega *= conversion_factor  # 转换为THz

# 设置一致的色标范围 (0-0.025 a.u.)
vmax = 0.0015
q_norms = sample_averaged.q_norms
omega = sample_averaged.omega
Clqw = sample_averaged.Clqw
Ctqw = sample_averaged.Ctqw

# ===================== 计算群速度 (差分法) =====================
def calculate_weighted_avg_freq(q_norms, omega, intensity_map, q_interval=0.05, q_max=2.0):
    """
    按指定间隔对q进行分箱，计算每个q点的强度加权平均频率及其标准差
    """
    # 生成目标q点
    target_qs = np.arange(0, q_max + q_interval/2, q_interval)
    weighted_omegas = []
    std_omegas = []  # 用于存储标准差（误差）
    valid_qs = []
    
    # 为了找到每个目标q对应的实际q索引
    # 假设q_norms是排序的，我们找最接近的点
    
    for target_q in target_qs:
        # 找到最接近target_q的实际q值的索引
        # 使用一定的容差范围，或者简单地找最近邻
        # 这里我们选择找落在 [target_q - interval/2, target_q + interval/2] 范围内的数据
        
        mask = (q_norms >= target_q - q_interval/2) & (q_norms < target_q + q_interval/2)
        
        if not np.any(mask):
            continue
            
        # 获取该q范围内的所有强度数据
        # intensity_map shape is (n_q_points, n_omega_points)
        # 我们需要对mask选中的所有q行的强度进行平均，或者分别处理
        # 需求描述："对于每个q点...收集该q值下所有数据点"
        # 这里的"每个q点"指的是 target_qs 中的点
        
        # 提取对应的强度切片 (n_selected_q, n_omega)
        intensities = intensity_map[mask, :]
        
        # 将这些q点的强度合并（例如取平均或求和），得到该target_q下的综合强度分布
        # shape: (n_omega,)
        avg_intensity_profile = np.mean(intensities, axis=0)
        
        # 过滤掉强度为0的点 (虽然加权平均时0强度自然不贡献，但为了严谨)
        valid_points = avg_intensity_profile > 1e-10  # 使用一个小阈值避免浮点误差
        
        if not np.any(valid_points):
            continue
            
        current_omegas = omega[valid_points]
        current_intensities = avg_intensity_profile[valid_points]
        
        # 计算加权平均频率: sum(freq * intensity) / sum(intensity)
        total_intensity = np.sum(current_intensities)
        weighted_omega = np.sum(current_omegas * current_intensities) / total_intensity
        
        # 计算加权标准差: sqrt( sum(intensity * (freq - avg)^2) / sum(intensity) )
        variance = np.sum(current_intensities * (current_omegas - weighted_omega)**2) / total_intensity
        std_omega = np.sqrt(variance)
        
        valid_qs.append(target_q)
        weighted_omegas.append(weighted_omega)
        std_omegas.append(std_omega)
        
    return np.array(valid_qs), np.array(weighted_omegas), np.array(std_omegas)

def calculate_group_velocity_diff(q_points, omega_points):
    """
    通过相邻点差分计算群速度
    v_g = d(omega) / d(q)
    返回: (频率中点, 群速度)
    """
    if len(q_points) < 2:
        return np.array([]), np.array([])
    
    # 对数组进行排序以防万一
    sorted_indices = np.argsort(q_points)
    q_sorted = q_points[sorted_indices]
    omega_sorted = omega_points[sorted_indices]
    
    v_g_list = []
    freq_mid_list = []
    
    for i in range(len(q_sorted) - 1):
        dq = q_sorted[i+1] - q_sorted[i]
        domega = omega_sorted[i+1] - omega_sorted[i]
        
        if dq == 0:
            continue
            
        # 计算斜率 (THz / (1/Angstrom)) -> Angstrom/ps * 10 (?)
        # 单位转换: 1 THz * 1 Angstrom = 10^12 s^-1 * 10^-10 m = 100 m/s = 0.1 km/s
        # 所以 v_g (km/s) = slope * 0.1
        slope = domega / dq
        v_g = slope * 0.1
        
        # 对应的频率取中点
        freq_mid = (omega_sorted[i] + omega_sorted[i+1]) / 2
        
        v_g_list.append(v_g)
        freq_mid_list.append(freq_mid)
        
    return np.array(freq_mid_list), np.array(v_g_list)

# 计算纵向(CL)群速度分布
q_weighted_L, omega_weighted_L, std_L = calculate_weighted_avg_freq(q_norms, omega, Clqw, q_interval=q_interval)
freq_L_diff, vg_L_diff = calculate_group_velocity_diff(q_weighted_L, omega_weighted_L)

# 计算横向(CT)群速度分布
q_weighted_T, omega_weighted_T, std_T = calculate_weighted_avg_freq(q_norms, omega, Ctqw, q_interval=q_interval)
freq_T_diff, vg_T_diff = calculate_group_velocity_diff(q_weighted_T, omega_weighted_T)


# ===================== 创建双列图 =====================
# 修改布局：使用 constrained_layout 实现响应式调整
# 调整为黄金比例 (16.18 : 10) 以满足视觉美学要求
fig = plt.figure(figsize=(16.18, 10), layout='constrained', dpi=300)

# 使用 3 列布局：左侧热图 | 中间色条 | 右侧散点图
# width_ratios 控制列宽比例
gs = fig.add_gridspec(2, 3, width_ratios=[1, 0.05, 1.2], height_ratios=[1, 1])

# 左侧热图 (含加权平均点)
ax1 = fig.add_subplot(gs[0, 0])  # 左上 - CL Heatmap
ax2 = fig.add_subplot(gs[1, 0])  # 左下 - CT Heatmap

# 中间色条轴 (跨两行)
cax1 = fig.add_subplot(gs[:, 1])

# 右侧群速度图
ax3 = fig.add_subplot(gs[0, 2])  # 右上 - CL Vg vs Freq
ax4 = fig.add_subplot(gs[1, 2])  # 右下 - CT Vg vs Freq

# ===== 左侧面板 (热图 + 散点) =====
# CL纵向 (左上)
im1 = ax1.pcolormesh(q_norms, omega, Clqw.T, 
                     cmap='Reds', shading='nearest',
                     vmin=0, vmax=vmax)
# 调整标题：字体增大至16pt，位置调整至图表内部左上角 (y < 1.0)
# y=0.95 将标题置于图内顶部，pad=-10 配合下移
ax1.set_title(r'$C_L(|\mathbf{q}|, \omega)$', loc='left', fontsize=16, y=0.95, pad=-10)
ax1.set_ylabel('Frequency (THz)', labelpad=5)
ax1.set_ylim(0, 50)
ax1.set_xlim(0, 2.0)

# 绘制加权平均点 (带误差棒)
if len(q_weighted_L) > 0:
    ax1.errorbar(q_weighted_L, omega_weighted_L, yerr=std_L, 
                 fmt='o', color='blue', markersize=4, alpha=0.8,
                 elinewidth=1.5, capsize=3, label='Weighted Avg ± Std')
    # ax1.legend(loc='upper left', fontsize='small')

# CT横向 (左下)
im2 = ax2.pcolormesh(q_norms, omega, Ctqw.T, 
                     cmap='Oranges', shading='nearest',
                     vmin=0, vmax=vmax)
ax2.set_title(r'$C_T(|\mathbf{q}|, \omega)$', loc='left', fontsize=16, y=0.95, pad=-10)
ax2.set_xlabel(r'$|\mathbf{q}|$ (1/Å)', labelpad=5)
ax2.set_ylabel('Frequency (THz)', labelpad=5)
ax2.set_ylim(0, 50)
ax2.set_xlim(0, 2.0)

# 绘制加权平均点 (带误差棒)
if len(q_weighted_T) > 0:
    ax2.errorbar(q_weighted_T, omega_weighted_T, yerr=std_T, 
                 fmt='o', color='blue', markersize=4, alpha=0.8,
                 elinewidth=1.5, capsize=3, label='Weighted Avg ± Std')
    # ax2.legend(loc='upper left', fontsize='small')

# 添加热图色条 (共用)
cbar1 = fig.colorbar(im1, cax=cax1, orientation='vertical')
cbar1.set_label('Intensity (a.u.)', labelpad=5)


# ===== 右侧面板 (群速度 vs 频率) =====
# CL纵向 (右上)
if len(freq_L_diff) > 0:
    ax3.scatter(freq_L_diff, vg_L_diff, color='crimson', s=40, alpha=0.8, edgecolors='k', label=r'$v_g^L$')
    # ax3.plot(freq_L_diff, vg_L_diff, color='crimson', alpha=0.4, lw=1.5) # 连线可选
    
    # 计算统计信息
    mean_vg_L = np.mean(vg_L_diff)
    std_vg_L = np.std(vg_L_diff)
    stats_text_L = f"Avg $v_g$: {mean_vg_L:.3f} ± {std_vg_L:.3f} km/s"
    
    # 添加文本框 (左下角)
    ax3.text(0.05, 0.05, stats_text_L, transform=ax3.transAxes, fontsize=12,
             verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

ax3.set_title(r'Longitudinal Group Velocity', pad=5)
ax3.set_ylabel(r'$v_g$ (km/s)', labelpad=5)
ax3.set_xlabel('Frequency (THz)', labelpad=5)
ax3.grid(True, linestyle='--', alpha=0.4)
# ax3.set_ylim(bottom=0) # 可选：限制y轴从0开始

# CT横向 (右下)
if len(freq_T_diff) > 0:
    ax4.scatter(freq_T_diff, vg_T_diff, color='darkorange', s=40, alpha=0.8, edgecolors='k', label=r'$v_g^T$')
    # ax4.plot(freq_T_diff, vg_T_diff, color='darkorange', alpha=0.4, lw=1.5) # 连线可选
    
    # 计算统计信息
    mean_vg_T = np.mean(vg_T_diff)
    std_vg_T = np.std(vg_T_diff)
    stats_text_T = f"Avg $v_g$: {mean_vg_T:.3f} ± {std_vg_T:.3f} km/s"
    
    # 添加文本框 (左下角)
    ax4.text(0.05, 0.05, stats_text_T, transform=ax4.transAxes, fontsize=12,
             verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

ax4.set_title(r'Transverse Group Velocity', pad=5)
ax4.set_ylabel(r'$v_g$ (km/s)', labelpad=5)
ax4.set_xlabel('Frequency (THz)', labelpad=5)
ax4.grid(True, linestyle='--', alpha=0.4)
# ax4.set_ylim(bottom=0)

# 保存结果
plt.savefig('dual_spectra_with_vg_scatter.png', dpi=300, bbox_inches='tight')
# plt.show() # 服务器端通常无法直接show

print(f"✅ 结果图已保存至 dual_spectra_with_vg_scatter.png")

