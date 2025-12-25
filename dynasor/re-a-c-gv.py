import os
import numpy as np
import matplotlib.pyplot as plt
from dynasor import compute_dynamic_structure_factors, Trajectory
from dynasor.qpoints import get_spherical_qpoints
from dynasor.post_processing import get_spherically_averaged_sample_binned
from dynasor.logging_tools import set_logging_level
# 添加核心修复
from dynasor.sample import read_sample_from_npz
from dynasor.units import radians_per_fs_to_THz as conversion_factor # conversion from 1/fs to meV


# 配置日志和文件路径
set_logging_level('INFO')
trajectory_filename = 'dump.xyz'
output_file = 'test.npz'  # 数据保存路径
max_points=8000
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
# ===================== 可视化电流相关函数 =====================
# 动态设置vmin/vmax（基于数据百分位数）+
vmax_L=0.0015
#vmax_L = np.percentile(sample_averaged.Clqw, 100)  
#vmax_T = np.percentile(sample_averaged.Ctqw, 100)
# 创建双面板图 (纵向CL + 横向CT)
fig, axes = plt.subplots(figsize=(3.4, 3.8), nrows=2, dpi=140,
                         sharex=True, sharey=True)

# 纵向电流相关 CL
ax = axes[0]
im_cl = ax.pcolormesh(sample_averaged.q_norms, sample_averaged.omega,
                      sample_averaged.Clqw.T, 
                      cmap='Reds', vmin=0, vmax=vmax_L)
ax.text(0.05, 0.85, r'$C_L(|\mathbf{q}|, \omega)$', transform=ax.transAxes,
        bbox={'color': 'white', 'alpha': 0.8, 'pad': 3})
#ax.plot([0, 1.5], [0, 54], alpha=0.5, ls='--', c='0.3', lw=2)

# 横向电流相关 CT
ax = axes[1]
im_ct = ax.pcolormesh(sample_averaged.q_norms, sample_averaged.omega,
                      sample_averaged.Ctqw.T, 
                      cmap='Oranges', vmin=0, vmax=vmax_L)
ax.text(0.05, 0.85, r'$C_T(|\mathbf{q}|, \omega)$', transform=ax.transAxes,
        bbox={'color': 'white', 'alpha': 0.8, 'pad': 3})
#ax.plot([0, 1.5], [0, 25], alpha=0.5, ls='--', c='0.3', lw=2)
#####################cai tiao###############
#fig.colorbar(im_cl, ax=axes[0], label='Intensity (a.u.)')
#fig.colorbar(im_ct, ax=axes[1], label='Intensity (a.u.)')
###########################################
# 坐标轴标签设置
ax.set_xlabel(r'$|\mathbf{q}|$ (1/Å)')
ax.set_ylabel('Frequency (THz)', y=1)
ax.set_ylim([0, 50])
ax.set_xlim([0, 2])
axes[0].tick_params(axis='both', direction='in')
axes[1].tick_params(axis='both', direction='in')
# 布局优化
fig.tight_layout()
plt.subplots_adjust(hspace=0.08)
plt.savefig('c.png', dpi=300, bbox_inches='tight')
plt.show()