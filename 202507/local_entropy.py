#!/usr/bin/env python3
import argparse
import numpy as np
from ovito.io import import_file
from ovito.data import CutoffNeighborFinder, DataCollection
from ovito.modifiers import ComputePropertyModifier

def calculate_local_entropy(data: DataCollection, 
                           cutoff: float = 6.0, 
                           sigma: float = 0.2, 
                           use_local_density: bool = False,
                           compute_average: bool = False,
                           average_cutoff: float = 5.0) -> np.ndarray:
    """
    计算体系中每个粒子的局部熵
    
    参数:
    data: 包含粒子数据的DataCollection对象
    cutoff: 邻居搜索截断半径 (Å)
    sigma: 高斯平滑参数 (Å)
    use_local_density: 是否使用局部密度校正
    compute_average: 是否计算邻域平均熵
    average_cutoff: 邻域平均截断半径 (Å)
    
    返回:
    包含每个粒子熵值的NumPy数组
    """
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
        
        # 数值积分 - 使用 trapezoid 替代 trapz
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

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='计算分子动力学模拟的局部熵')
    parser.add_argument('input', help='输入文件路径 (支持xyz, lammps-data, POSCAR等格式)')
    parser.add_argument('-o', '--output', default='entropy.dat', help='输出文件路径')
    parser.add_argument('--cutoff', type=float, default=6.0, help='邻居搜索截断半径(Å)')
    parser.add_argument('--sigma', type=float, default=0.2, help='高斯平滑参数(Å)')
    parser.add_argument('--local-density', action='store_true', help='使用局部密度校正')
    parser.add_argument('--average', action='store_true', help='计算邻域平均熵')
    parser.add_argument('--avg-cutoff', type=float, default=6.0, help='邻域平均截断半径(Å)')
    
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
    
    # 保存结果 - 修正格式字符串问题
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

if __name__ == "__main__":
    main()