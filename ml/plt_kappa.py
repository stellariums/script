# 导入必要的库
"""
plt_kappa.py
=============

此脚本用于分析和绘制 GPUMD 计算得到的热导率数据 (kappa.out)。
它计算各个方向（kx, ky, kz）的热导率运行平均值，并根据用户指定的方向绘制曲线。

功能：
1. 读取 `kappa.out` 文件（优先查找当前目录，其次查找父目录）。
2. 计算热导率的运行平均值 (Running Average)。
3. 绘制热导率随时间变化的曲线。
4. 在终端输出最后时刻的热导率数值。

用法：
    python plt_kappa.py [direction]

参数：
    direction : 可选参数，指定要绘制和输出的方向。
                可选值: 'kx', 'ky', 'kz', 'all'
                默认值: 'all' (显示所有方向)

示例：
    python plt_kappa.py        # 绘制并输出所有方向
    python plt_kappa.py ky     # 仅绘制并输出 ky 方向
    python plt_kappa.py kz     # 仅绘制并输出 kz 方向
"""
import os
import argparse
from pylab import *
from ase.build import graphene_nanoribbon
from ase.io import write
from scipy.integrate import cumulative_trapezoid

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Plot thermal conductivity and print final values.')
parser.add_argument('direction', nargs='?', default='all', choices=['kx', 'ky', 'kz', 'all'],
                    help='Direction to plot and print: kx, ky, kz, or all (default: all)')
args = parser.parse_args()

# 设置图形属性参数
aw = 2      # 坐标轴线宽
fs = 16     # 字体大小
font = {'size'   : fs}
matplotlib.rc('font', **font)
matplotlib.rc('axes' , linewidth=aw)

# 设置图形属性的函数
def set_fig_properties(ax_list):
    tl = 8    # 主刻度线长度
    tw = 2
    tlm = 4

    # 为每个坐标轴设置刻度属性
    for ax in ax_list:
        ax.tick_params(which='major', length=tl, width=tw)
        ax.tick_params(which='minor', length=tlm, width=tw)
        ax.tick_params(which='both', axis='both', direction='in', right=True, top=True)
        
# 定义热导率标签和加载数据
labels_kappa = ['kxi', 'kxo', 'kyi', 'kyo', 'kz']  # 不同方向的热导率标签

# 尝试从当前目录或父目录加载 kappa.out
if os.path.exists("kappa.out"):
    kappa_array = np.loadtxt("kappa.out")
elif os.path.exists("../kappa.out"):
    kappa_array = np.loadtxt("../kappa.out")
else:
    raise FileNotFoundError("kappa.out not found in current or parent directory")

# 将数据整理成字典格式
kappa = dict()
for label_num, key in enumerate(labels_kappa):
    kappa[key] = kappa_array[:, label_num]
# 计算运行平均值的函数
def running_ave(y, x):
    return cumulative_trapezoid(y, x, initial=0) / x

# 创建时间数组（纳秒单位）并计算各分量的运行平均值
t = np.arange(1,kappa['kxi'].shape[0]+1)*0.001  # ns
kappa['kyi_ra'] = running_ave(kappa['kyi'],t)
kappa['kyo_ra'] = running_ave(kappa['kyo'],t)
kappa['kxi_ra'] = running_ave(kappa['kxi'],t)
kappa['kxo_ra'] = running_ave(kappa['kxo'],t)
kappa['kz_ra'] = running_ave(kappa['kz'],t)

# 计算总的 x 和 y 方向热导率
kappa['kx_ra'] = kappa['kxi_ra'] + kappa['kxo_ra']
kappa['ky_ra'] = kappa['kyi_ra'] + kappa['kyo_ra']

# 绘图：展示 kx, ky, kz 三个方向的热导率
figure(figsize=(8, 6))
set_fig_properties([gca()])

# 根据参数选择绘制的曲线
if args.direction == 'all' or args.direction == 'kx':
    plot(t, kappa['kx_ra'], linewidth=2, label=r'$\kappa_{x}$')
if args.direction == 'all' or args.direction == 'ky':
    plot(t, kappa['ky_ra'], linewidth=2, label=r'$\kappa_{y}$')
if args.direction == 'all' or args.direction == 'kz':
    plot(t, kappa['kz_ra'], linewidth=2, label=r'$\kappa_{z}$')

# 设置坐标轴和标签
xlim([0, 10])
gca().set_xticks(range(0, 11, 2))
# 根据数据自动调整 ylim，或者您可以指定范围
# ylim([-100, 2000]) 

xlabel('Time (ns)')
ylabel(r'$\kappa$ (W/m/K)')
legend()
title(f'Thermal Conductivity ({args.direction} direction)')

# 获取并打印最后一个数据点的坐标
last_line = plt.gca().lines[-1]
last_x, last_y = last_line.get_xdata(), last_line.get_ydata()
last_point = (last_x[-1], last_y[-1])
# print("最后一个点的坐标是:", last_point)

print("-" * 30)
print(f"Final Thermal Conductivity (W/m/K) at t = {t[-1]:.3f} ns:")
if args.direction == 'all' or args.direction == 'kx':
    print(f"kx: {kappa['kx_ra'][-1]:.4f}")
if args.direction == 'all' or args.direction == 'ky':
    print(f"ky: {kappa['ky_ra'][-1]:.4f}")
if args.direction == 'all' or args.direction == 'kz':
    print(f"kz: {kappa['kz_ra'][-1]:.4f}")
print("-" * 30)

# 调整图形布局并显示
tight_layout()
show()
