###导入单个dump.xyz文件，输出某一帧结构的密度，sp2.sp3占比，sp2/sp3比例，结晶度以及局部熵信息
###使用方法：python script.py dump.xyz -f 0 --entropy \
#           --cutoff 6.0 --sigma 0.2 \
#           --local-density --average \
#           --entropy-output entropy_data.dat

import numpy as np
import sys
import argparse
from ovito.io import import_file
from ovito.modifiers import CreateBondsModifier, CoordinationAnalysisModifier, IdentifyDiamondModifier
from ovito.data import CutoffNeighborFinder, DataCollection
from ovito.modifiers import ComputePropertyModifier

def calculate_local_entropy(data: DataCollection, 
                           cutoff: float = 6.0, 
                           sigma: float = 0.2, 
                           use_local_density: bool = False,
                           compute_average: bool = False,
                           average_cutoff: float = 6.0) -> np.ndarray:
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

def analyze_frame(file_path, frame_index, entropy_options=None):
    """
    分析GPUMD生成的dump.xyz文件的指定帧，可选的熵计算
    """
    # 导入文件并获取总帧数
    pipeline = import_file(file_path, multiple_frames=True)
    num_frames = pipeline.source.num_frames
    
    # 处理负数索引（Python风格：-1表示最后一帧）
    if frame_index < 0:
        actual_index = num_frames + frame_index
    else:
        actual_index = frame_index
    
    # 验证帧索引有效性
    if actual_index < 0 or actual_index >= num_frames:
        raise ValueError(f"无效帧索引：{frame_index}（有效范围：0到{num_frames-1}，或负数索引）")
    
    print(f"检测到 {num_frames} 帧数据，将分析第 {actual_index} 帧（输入索引：{frame_index})")
    
    # 计算指定帧的基础属性
    data = pipeline.compute(actual_index)
    volume = data.cell.volume  # 系统体积（Å³）
    num_atoms = data.particles.count  # 原子总数
    
    # 计算质量（假设为碳体系）
    atomic_mass = 12.01  # 碳原子质量（g/mol）
    total_relative_mass = num_atoms * atomic_mass  # 系统总质量（相对质量）
    actual_mass = total_relative_mass / 6.022e23  # 实际质量（克）
    
    # 计算密度（g/cm³）
    density = (actual_mass * 1e24) / volume
    
    # 添加创建键的修饰器
    pipeline.modifiers.append(CreateBondsModifier(cutoff=1.85))
    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=1.85))
    
    # 计算指定帧的配位数
    data = pipeline.compute(actual_index)
    coord_numbers = data.particles['Coordination']
    
    # 计算sp²和sp³比例
    sp2_count = np.sum(coord_numbers == 3)
    sp3_count = np.sum(coord_numbers == 4)
    sp2_percent = (sp2_count / num_atoms) * 100
    sp3_percent = (sp3_count / num_atoms) * 100
    
    # 计算sp3/sp2比值
    sp3_sp2_ratio = sp3_count / sp2_count if sp2_count > 0 else np.inf
    
    # 添加金刚石结构识别修饰器
    pipeline.modifiers.append(IdentifyDiamondModifier())
    
    # 计算指定帧的结晶度
    data = pipeline.compute(actual_index)
    structure_types = data.particles['Structure Type']
    
    # 计算结晶度（识别为金刚石结构的原子百分比）
    crystal_atoms = np.sum((structure_types >= 1) & (structure_types <= 6))
    crystallinity = (crystal_atoms / num_atoms) * 100
    
    # 准备结果字典
    results = {
        "Density (g/cm³)": density,
        "sp² Atoms (%)": sp2_percent,
        "sp³ Atoms (%)": sp3_percent,
        "sp³/sp² Ratio": sp3_sp2_ratio,
        "Crystallinity (%)": crystallinity
    }
    
    # 计算熵（如果指定了选项）
    entropy_stats = {}
    if entropy_options:
        entropy = calculate_local_entropy(
            data,
            cutoff=entropy_options['cutoff'],
            sigma=entropy_options['sigma'],
            use_local_density=entropy_options['use_local_density'],
            compute_average=entropy_options['compute_average'],
            average_cutoff=entropy_options['average_cutoff']
        )
        
        # 保存熵数据到文件
        output_data = np.column_stack((np.arange(len(entropy)), entropy))
        np.savetxt(entropy_options['output'], output_data, header='粒子索引\t熵值', fmt='%d %.6f')
        
        # 计算熵统计信息
        entropy_min = np.min(entropy)
        entropy_max = np.max(entropy)
        entropy_mean = np.mean(entropy)
        entropy_std = np.std(entropy)
        
        # 添加到熵统计字典
        entropy_stats = {
            "Entropy Min": entropy_min,
            "Entropy Max": entropy_max,
            "Entropy Mean": entropy_mean,
            "Entropy Std": entropy_std,
            "Normalized Std": entropy_std / entropy_mean if entropy_mean != 0 else np.nan
        }
        
        # 添加到主结果字典
        results.update(entropy_stats)
        
        # 打印熵统计信息
        print("\n熵值统计:")
        print(f"  最小值: {entropy_min:.6f}")
        print(f"  最大值: {entropy_max:.6f}")
        print(f"  平均值: {entropy_mean:.6f}")
        print(f"  标准差: {entropy_std:.6f}")
        print(f"  归一化标准差: {entropy_std/entropy_mean:.6f}")
    
    return results, entropy_stats

if __name__ == "__main__":
    # 配置命令行参数解析
    parser = argparse.ArgumentParser(description="分析GPUMD生成的dump.xyz文件")
    parser.add_argument("dump_file", help="输入文件路径（dump.xyz）")
    parser.add_argument("-f", "--frame", type=int, default=-1,
                        help="分析指定帧（0-based索引，负数表示倒数，默认：-1=最后一帧）")
    
    # 熵计算选项
    entropy_group = parser.add_argument_group('熵计算选项')
    entropy_group.add_argument('--entropy', action='store_true', 
                              help='启用局部熵计算')
    entropy_group.add_argument('--entropy-output', default='entropy.dat', 
                              help='熵输出文件路径')
    entropy_group.add_argument('--cutoff', type=float, default=6.0, 
                              help='邻居搜索截断半径(Å)')
    entropy_group.add_argument('--sigma', type=float, default=0.2, 
                              help='高斯平滑参数(Å)')
    entropy_group.add_argument('--local-density', action='store_true', 
                              help='使用局部密度校正')
    entropy_group.add_argument('--average', action='store_true', 
                              help='计算邻域平均熵')
    entropy_group.add_argument('--avg-cutoff', type=float, default=6.0, 
                              help='邻域平均截断半径(Å)')
    
    args = parser.parse_args()
    
    # 准备熵选项字典（如果需要）
    entropy_options = None
    if args.entropy:
        entropy_options = {
            'output': args.entropy_output,
            'cutoff': args.cutoff,
            'sigma': args.sigma,
            'use_local_density': args.local_density,
            'compute_average': args.average,
            'average_cutoff': args.avg_cutoff
        }
    
    try:
        # 分析指定帧
        results, entropy_stats = analyze_frame(args.dump_file, args.frame, entropy_options)
        
        # 打印结果
        print(f"\n系统属性分析（第 {args.frame} 帧）:")
        print("-" * 60)
        for prop, value in results.items():
            if "%" in prop:
                print(f"{prop}: {value:.4f}")
            elif "Ratio" in prop:
                print(f"{prop}: {'Infinity (no sp² atoms)' if value == np.inf else f'{value:.4f}'}")
            elif "Normalized Std" in prop:
                print(f"{prop}: {value:.6f}")
            elif "Entropy" in prop:
                print(f"{prop}: {value:.6f}")
            else:
                print(f"{prop}: {value:.6f}")
        
        # 如果计算了熵，打印保存位置
        if args.entropy:
            print(f"\n熵数据已保存至: {args.entropy_output}")
    
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)