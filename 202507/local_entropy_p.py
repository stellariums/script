#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ovito.io import import_file
from ovito.data import CutoffNeighborFinder, DataCollection
from ovito.modifiers import ComputePropertyModifier
from scipy.signal import find_peaks

def calculate_local_entropy(data: DataCollection, 
                           cutoff: float = 6.0, 
                           sigma: float = 0.2, 
                           use_local_density: bool = False,
                           compute_average: bool = False,
                           average_cutoff: float = 5.0) -> np.ndarray:
    """计算体系中每个粒子的局部熵"""
    # 验证输入参数
    assert cutoff > 0.0, "Cutoff must be positive"
    assert 0 < sigma < cutoff, "Sigma must be between 0 and cutoff"
    assert average_cutoff > 0, "Average cutoff must be positive"

    # 全局粒子密度
    global_rho = data.particles.count / data.cell.volume

    # 初始化邻居查找器
    finder = CutoffNeighborFinder(cutoff, data)

    # 创建存储局部熵的数组
    local_entropy = np.empty(data.particles.count)

    # 数值积分参数设置
    nbins = int(cutoff / sigma) + 1
    r = np.linspace(0.0, cutoff, num=nbins)
    rsq = r**2
    prefactor = rsq * (4 * np.pi * global_rho * np.sqrt(2 * np.pi * sigma**2))
    prefactor[0] = prefactor[1]  # 避免除以零

    # 计算每个粒子的局部熵
    for particle_index in range(data.particles.count):
        # 获取邻居距离
        r_ij = finder.neighbor_distances(particle_index)
        
        # 计算 g_m(r)
        r_diff = np.expand_dims(r, 0) - np.expand_dims(r_ij, 1)
        g_m = np.sum(np.exp(-r_diff**2 / (2.0 * sigma**2)), axis=0) / prefactor
        
        # 使用局部密度（如果需要）
        if use_local_density:
            local_volume = 4/3 * np.pi * cutoff**3
            rho = len(r_ij) / local_volume
            g_m *= global_rho / rho
        else:
            rho = global_rho
        
        # 计算积分函数
        integrand = np.where(g_m >= 1e-10, (g_m * np.log(g_m) - g_m + 1.0) * rsq, rsq)
        
        # 数值积分
        local_entropy[particle_index] = -2.0 * np.pi * rho * np.trapezoid(integrand, r)
    
    # 添加熵属性到粒子
    data.particles_.create_property('Entropy', data=local_entropy)
    
    # 计算邻域平均值（如果需要）
    if compute_average:
        data.apply(ComputePropertyModifier(
            output_property = 'Entropy',
            operate_on = 'particles',
            cutoff_radius = average_cutoff,
            expressions = ['Entropy / (NumNeighbors + 1)'],
            neighbor_expressions = ['Entropy / (NumNeighbors + 1)']))
        return data.particles['Entropy'][:]
    
    return local_entropy

def calculate_rdf(data: DataCollection, atom_index: int, 
                 r_cutoff: float = 10.0, bin_width: float = 0.1) -> tuple:
    """
    计算单个原子的径向分布函数(RDF)
    
    参数:
    data: 包含粒子数据的DataCollection对象
    atom_index: 中心原子的索引
    r_cutoff: RDF计算的截断半径(Å)
    bin_width: 直方图的分组宽度(Å)
    
    返回:
    (r, g_r): 距离数组和对应的RDF值数组
    """
    # 获取系统参数
    num_bins = int(r_cutoff / bin_width)
    global_rho = data.particles.count / data.cell.volume
    
    # 初始化邻居查找器
    finder = CutoffNeighborFinder(r_cutoff, data)
    
    # 获取所有邻居的距离
    distances = finder.neighbor_distances(atom_index)
    
    # 排除自身(距离为0)
    distances = distances[distances > 0]
    
    # 创建直方图
    hist, bin_edges = np.histogram(distances, bins=num_bins, range=(0, r_cutoff))
    r = bin_edges[:-1] + bin_width/2  # 使用bin中心作为距离
    
    # 计算每个bin的体积 (4/3πr^3的外壳体积)
    shell_volumes = (4/3) * np.pi * (np.power(bin_edges[1:], 3) - np.power(bin_edges[:-1], 3))
    
    # 避免除以零
    shell_volumes[shell_volumes == 0] = np.finfo(float).eps
    
    # 计算RDF: g(r) = [dN(r)/dr] / [4πr²ρ]
    g_r = hist / (shell_volumes * global_rho)
    
    return r, g_r

def plot_rdf_matplotlib(data: DataCollection, output_prefix: str, 
                      r_cutoff: float = 10.0, bin_width: float = 0.1):
    """
    使用Matplotlib绘制随机三个原子的径向分布函数图
    
    参数:
    data: 包含粒子数据的DataCollection对象
    output_prefix: 输出文件前缀
    r_cutoff: RDF计算的截断半径(Å)
    bin_width: 直方图的分组宽度(Å)
    """
    # 随机选择三个原子
    np.random.seed(42)  # 固定随机种子以确保可重复性
    atom_indices = np.random.choice(data.particles.count, size=3, replace=False)
    
    # 创建绘图
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    
    # 计算并绘制每个原子的RDF
    for i, atom_index in enumerate(atom_indices):
        r, g_r = calculate_rdf(data, atom_index, r_cutoff, bin_width)
        
        # 绘制RDF曲线
        line, = plt.plot(r, g_r, linewidth=2.5, alpha=0.85, 
                         label=f'Atom {atom_index}')
        color = line.get_color()
        
        # 识别关键特征点
        peaks, _ = find_peaks(g_r, height=0.1, distance=10)
        minima, _ = find_peaks(-g_r, distance=10)
        
        # 标记前三个峰值
        for j, peak_idx in enumerate(peaks[:3]):
            peak_r = r[peak_idx]
            peak_val = g_r[peak_idx]
            plt.annotate(f'Peak {j+1}\n({peak_r:.2f}Å, {peak_val:.2f})',
                         xy=(peak_r, peak_val),
                         xytext=(peak_r+0.5, peak_val+0.5),
                         arrowprops=dict(facecolor='black', shrink=0.05, 
                                         width=1.5, headwidth=7),
                         fontsize=10, bbox=dict(boxstyle='round,pad=0.3', 
                                               alpha=0.2, color=color))
            plt.scatter(peak_r, peak_val, s=120, color=color, 
                       edgecolor='black', zorder=5)
        
        # 标记前两个谷值
        for j, min_idx in enumerate(minima[:2]):
            min_r = r[min_idx]
            min_val = g_r[min_idx]
            plt.annotate(f'Valley {j+1}\n({min_r:.2f}Å, {min_val:.2f})',
                         xy=(min_r, min_val),
                         xytext=(min_r+0.5, min_val-0.5),
                         arrowprops=dict(facecolor='black', shrink=0.05, 
                                         width=1.5, headwidth=7),
                         fontsize=10, bbox=dict(boxstyle='round,pad=0.3', 
                                               alpha=0.2, color=color))
            plt.scatter(min_r, min_val, s=120, color=color, 
                       marker='s', edgecolor='black', zorder=5)
    
    # 添加图例和标签
    plt.title('Radial Distribution Functions (RDF) for Random Atoms', fontsize=16, pad=15)
    plt.xlabel('Distance r (Å)', fontsize=14)
    plt.ylabel('g(r)', fontsize=14)
    plt.legend(fontsize=12, framealpha=0.9)
    
    # 设置网格和样式
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=1.0, color='gray', linestyle='-', alpha=0.5)
    
    # 设置坐标轴范围
    plt.xlim(0, r_cutoff)
    plt.ylim(bottom=0)
    
    # 添加文本说明
    plt.figtext(0.5, 0.01, 
               f"System: {data.particles.count} atoms | Cell Volume: {data.cell.volume:.1f} Å³",
               ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    
    # 保存图像
    output_path = f"{output_prefix}_rdf_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"RDF plot saved to {output_path}")
    
    # 显示图像
    plt.tight_layout()
    plt.show()

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='计算分子动力学模拟的局部熵并绘制RDF')
    parser.add_argument('input', help='输入文件路径 (支持xyz, lammps-data, POSCAR等格式)')
    parser.add_argument('-o', '--output', default='entropy.dat', help='熵值输出文件路径')
    parser.add_argument('--cutoff', type=float, default=6.0, help='邻居搜索截断半径(Å)')
    parser.add_argument('--sigma', type=float, default=0.2, help='高斯平滑参数(Å)')
    parser.add_argument('--local-density', action='store_true', help='使用局部密度校正')
    parser.add_argument('--average', action='store_true', help='计算邻域平均熵')
    parser.add_argument('--avg-cutoff', type=float, default=6.0, help='邻域平均截断半径(Å)')
    # RDF绘图参数
    parser.add_argument('--plot-rdf', action='store_true', help='绘制随机三个原子的径向分布函数图')
    parser.add_argument('--rdf-cutoff', type=float, default=10.0, help='RDF计算的截断半径(Å)')
    parser.add_argument('--bin-width', type=float, default=0.1, help='RDF计算的直方图分组宽度(Å)')
    
    args = parser.parse_args()

    # 导入文件
    pipeline = import_file(args.input)
    data = pipeline.compute()

    # 计算局部熵
    entropy = calculate_local_entropy(
        data,
        cutoff=args.cutoff,
        sigma=args.sigma,
        use_local_density=args.local_density,
        compute_average=args.average,
        average_cutoff=args.avg_cutoff
    )

    # 创建包含索引和熵值的数组
    output_data = np.column_stack((np.arange(len(entropy)), entropy))
    
    # 保存结果
    np.savetxt(args.output, output_data, header='粒子索引\t熵值', fmt='%d %.6f')
    
    # 打印统计信息
    print(f"分析完成! 结果保存至: {args.output}")
    print(f"体系统计:")
    print(f"  粒子总数: {data.particles.count}")
    print(f"  晶胞体积: {data.cell.volume:.2f} Å³")
    print(f"  密度: {data.particles.count/data.cell.volume:.4f} atoms/Å³")
    print(f"熵值统计:")
    print(f"  最小值: {np.min(entropy):.6f}")
    print(f"  最大值: {np.max(entropy):.6f}")
    print(f"  平均值: {np.mean(entropy):.6f}")
    print(f"  标准差: {np.std(entropy):.6f}")
    print(f"归一化标准差：{(np.std(entropy)/np.mean(entropy)):.6f}")

    # 绘制RDF图（如果指定）
    if args.plot_rdf:
        output_prefix = args.output.split('.')[0]  # 移除扩展名
        plot_rdf_matplotlib(data, output_prefix, args.rdf_cutoff, args.bin_width)

if __name__ == "__main__":
    main()