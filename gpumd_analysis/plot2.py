# 导入必要的库
# pylab: 用于绘图和数据处理
# ase: 原子模拟环境库，用于构建纳米结构
# scipy: 科学计算库，用于积分运算
from pylab import *
from ase.build import graphene_nanoribbon
from ase.io import write
from scipy.integrate import cumulative_trapezoid

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
kappa_array = np.loadtxt("kappa.out")              # 从文件加载热导率数据

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

# 以下是被注释掉的图形绘制代码（原本用于创建多个子图）
#figure(figsize=(12,10))
#subplot(2,2,1)
#set_fig_properties([gca()])  # 设置图形属性
#plot(t, kappa['kyi'],color='C7',alpha=0.5)
#plot(t, kappa['kyi_ra'], linewidth=2)
#xlim([0, 10])
#gca().set_xticks(range(0,11,2))
#ylim([-2000, 4000])
#gca().set_yticks(range(-2000,4001,1000))
#xlabel('time (ns)')
#ylabel(r'$\kappa_{in}$ W/m/K')
#title('(a)')

#subplot(2,2,2)
#set_fig_properties([gca()])
#plot(t, kappa['kyo'],color='C7',alpha=0.5)
#plot(t, kappa['kyo_ra'], linewidth=2, color='C3')
#xlim([0, 10])
#gca().set_xticks(range(0,11,2))
#ylim([0, 4000])
#gca().set_yticks(range(0,4001,1000))
#xlabel('time (ns)')
#ylabel(r'$\kappa_{out}$ (W/m/K)')
#title('(b)')

#subplot(2,2,3)
#set_fig_properties([gca()])
#plot(t, kappa['kyi_ra'], linewidth=2)
#plot(t, kappa['kyo_ra'], linewidth=2, color='C3')
#plot(t, kappa['kyi_ra']+kappa['kyo_ra'], linewidth=2, color='k')
#xlim([0, 10])
#gca().set_xticks(range(0,11,2))
#ylim([0, 4000])
#gca().set_yticks(range(0,4001,1000))
#xlabel('time (ns)')
#ylabel(r'$\kappa$ (W/m/K)')
#legend(['in', 'out', 'total'])
#title('(c)')


# 创建绘图（大部分子图被注释掉了）
#subplot(2,2,4)
set_fig_properties([gca()])
#plot(t, kappa['kxi_ra']+kappa['kxo_ra'],color='k', linewidth=2)
#plot(t, kappa['kxi_ra']+kappa['kxo_ra'], color='C0', linewidth=2)
plot(t, kappa['kz_ra'], color='C3', linewidth=2)  # 只绘制z方向热导率
#xlim([0, 10])
#gca().set_xticks(range(0,11,2))
ylim([-100, 900])
#gca().set_yticks(range(-2000,4001,1000))
xlabel('time (ns)')
ylabel(r'$\kappa$ (W/m/K)')
legend(['xx', 'xy', 'zx'])
title('(d)')

# 获取并打印最后一个数据点的坐标
last_line = plt.gca().lines[-1]
last_x, last_y = last_line.get_xdata(), last_line.get_ydata()
last_point = (last_x[-1], last_y[-1])
print("最后一个点的坐标是:", last_point)

# 调整图形布局并显示
tight_layout()
show()
