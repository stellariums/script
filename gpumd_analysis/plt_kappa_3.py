"""
plt_kappa_3.py
===============

此脚本用于分析和绘制三个独立运行 (fold1, fold2, fold3) 的 GPUMD 热导率数据。
它计算各个方向（kx, ky, kz）的运行平均值，并支持对比不同 fold 的结果或绘制平均结果。

功能：
1. 读取 `fold1/kappa.out`, `fold2/kappa.out`, `fold3/kappa.out`。
2. 计算每个 fold 的热导率运行平均值 (kx, ky, kz)。
3. 计算三个 fold 的平均值。
4. 根据用户参数绘制曲线：
   - 指定方向 (kx/ky/kz)：绘制三个 fold 的曲线和平均曲线。
   - all：绘制 kx, ky, kz 三个方向的平均曲线。
5. 在终端输出最后时刻的热导率数值。

用法：
    python plt_kappa_3.py [direction]

参数：
    direction : 可选参数，指定要绘制和输出的方向。
                可选值: 'kx', 'ky', 'kz', 'all'
                默认值: 'all'

示例：
    python plt_kappa_3.py        # 绘制 kx, ky, kz 的平均值曲线
    python plt_kappa_3.py kz     # 绘制 kz 方向的 fold1, fold2, fold3 和平均值
"""
import os
import argparse
from pylab import *
from scipy.integrate import cumulative_trapezoid

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Plot thermal conductivity from 3 folds.')
parser.add_argument('direction', nargs='?', default='all', choices=['kx', 'ky', 'kz', 'all'],
                    help='Direction to plot: kx, ky, kz, or all (default: all)')
args = parser.parse_args()

aw = 2
fs = 16
font = {'size': fs}
matplotlib.rc('font', **font)
matplotlib.rc('axes', linewidth=aw)

def set_fig_properties(ax_list):
    tl = 8
    tw = 2
    tlm = 4
    for ax in ax_list:
        ax.tick_params(which='major', length=tl, width=tw)
        ax.tick_params(which='minor', length=tlm, width=tw)
        ax.tick_params(which='both', axis='both', direction='in', right=True, top=True)

labels_kappa = ['kxi', 'kxo', 'kyi', 'kyo', 'kz']

def load_kappa(path):
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return None
    arr = np.loadtxt(path)
    d = {}
    for i, key in enumerate(labels_kappa):
        d[key] = arr[:, i]
    # 计算总的 kx 和 ky
    d['kx'] = d['kxi'] + d['kxo']
    d['ky'] = d['kyi'] + d['kyo']
    return d

def running_ave(y, x):
    return cumulative_trapezoid(y, x, initial=0) / x

k1 = load_kappa('fold1/kappa.out')
k2 = load_kappa('fold2/kappa.out')
k3 = load_kappa('fold3/kappa.out')

# 检查是否成功加载
if k1 is None or k2 is None or k3 is None:
    print("Error: One or more folders (fold1, fold2, fold3) or kappa.out files are missing.")
    exit(1)

n = min(k1['kz'].shape[0], k2['kz'].shape[0], k3['kz'].shape[0])
t = np.arange(1, n + 1) * 0.001

# 计算各分量的运行平均值
data = {'fold1': {}, 'fold2': {}, 'fold3': {}, 'mean': {}}
directions = ['kx', 'ky', 'kz']

for d in directions:
    data['fold1'][d] = running_ave(k1[d][:n], t)
    data['fold2'][d] = running_ave(k2[d][:n], t)
    data['fold3'][d] = running_ave(k3[d][:n], t)
    data['mean'][d] = (data['fold1'][d] + data['fold2'][d] + data['fold3'][d]) / 3.0

# 绘图
figure(figsize=(8, 6))
set_fig_properties([gca()])

if args.direction == 'all':
    # 绘制三个方向的平均值
    plot(t, data['mean']['kx'], linewidth=2, label=r'$\kappa_{x,mean}$')
    plot(t, data['mean']['ky'], linewidth=2, label=r'$\kappa_{y,mean}$')
    plot(t, data['mean']['kz'], linewidth=2, label=r'$\kappa_{z,mean}$')
    title('Mean Thermal Conductivity (x, y, z)')
else:
    # 绘制特定方向的各个 fold 和平均值
    d = args.direction
    plot(t, data['fold1'][d], linewidth=2, color='C0', alpha=0.5, label='fold1')
    plot(t, data['fold2'][d], linewidth=2, color='C1', alpha=0.5, label='fold2')
    plot(t, data['fold3'][d], linewidth=2, color='C2', alpha=0.5, label='fold3')
    plot(t, data['mean'][d], linewidth=2, color='k', label='mean')
    title(f'Thermal Conductivity ({d})')

# ylim([-100, 2000]) # 可根据需要取消注释
xlabel('Time (ns)')
ylabel(r'$\kappa$ (W/m/K)')
legend()

# 终端输出
print("-" * 30)
print(f"Final Mean Thermal Conductivity (W/m/K) at t = {t[-1]:.3f} ns:")
if args.direction == 'all':
    print(f"kx (mean): {data['mean']['kx'][-1]:.4f}")
    print(f"ky (mean): {data['mean']['ky'][-1]:.4f}")
    print(f"kz (mean): {data['mean']['kz'][-1]:.4f}")
else:
    d = args.direction
    print(f"{d} (fold1): {data['fold1'][d][-1]:.4f}")
    print(f"{d} (fold2): {data['fold2'][d][-1]:.4f}")
    print(f"{d} (fold3): {data['fold3'][d][-1]:.4f}")
    print(f"{d} (mean) : {data['mean'][d][-1]:.4f}")
print("-" * 30)

os.makedirs('figs', exist_ok=True)
tight_layout()
savefig('figs/kappa_mean.png', dpi=300)
show()