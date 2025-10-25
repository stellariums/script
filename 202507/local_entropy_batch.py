###对导入的一系列model.xyz文件进行计算，以.csv文件的形式导出其局部熵

import argparse
import os
import numpy as np
import multiprocessing as mp
import pandas as pd
from ovito.io import import_file
from ovito.data import CutoffNeighborFinder, DataCollection
from ovito.modifiers import ComputePropertyModifier
from functools import partial

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
    if cutoff <= 0.0 or not (0 < sigma < cutoff) or average_cutoff <= 0:
        raise ValueError("Invalid parameters: cutoff and average_cutoff must be positive, sigma between 0 and cutoff")

    # 全局粒子密度
    global_rho = data.particles.count / data.cell.volume

    # 初始化邻居查找器
    finder = CutoffNeighborFinder(cutoff, data)

    # 创建存储局部熵的数组
    local_entropy = np.empty(data.particles.count)

    # 数值积分参数设置
    nbins = max(50, int(cutoff / sigma))  # 确保足够的采样点
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
        
        # 使用局部密度校正
        if use_local_density:
            local_volume = 4/3 * np.pi * cutoff**3
            rho = len(r_ij) / local_volume if local_volume > 0 else global_rho
            g_m *= global_rho / rho
        else:
            rho = global_rho
        
        # 计算积分函数（添加数值稳定性）
        g_m_clipped = np.clip(g_m, 1e-10, None)
        integrand = (g_m_clipped * np.log(g_m_clipped) - g_m_clipped + 1.0) * rsq
        
        # 数值积分
        local_entropy[particle_index] = -2.0 * np.pi * rho * np.trapz(integrand, r)
    
    # 添加熵属性到粒子
    data.particles_.create_property('Entropy', data=local_entropy)
    
    # 计算邻域平均值
    if compute_average:
        data.apply(ComputePropertyModifier(
            output_property = 'Entropy',
            operate_on = 'particles',
            cutoff_radius = average_cutoff,
            expressions = ['Entropy / (NumNeighbors + 1)'],
            neighbor_expressions = ['Entropy / (NumNeighbors + 1)']))
        return data.particles['Entropy'][:]
    
    return local_entropy

def process_single_file(file_path, args):
    """
    处理单个XYZ文件并计算熵统计量（独立函数用于并行化）
    
    返回:
    包含统计信息的元组 (压缩目录, 预热目录, 文件路径, 平均值, 标准差, 方差, 归一化标准差, 归一化方差)
    """
    try:
        pipeline = import_file(file_path)
        data = pipeline.compute()
        
        entropy = calculate_local_entropy(
            data,
            cutoff=args.cutoff,
            sigma=args.sigma,
            use_local_density=args.local_density,
            compute_average=args.average,
            average_cutoff=args.avg_cutoff
        )
        
        # 计算统计量
        mean = np.mean(entropy)
        std = np.std(entropy)
        variance = np.var(entropy)
        
        # 归一化统计量（防除零处理）
        normalized_std = std / mean if abs(mean) > 1e-10 else np.nan
        normalized_variance = variance / (mean ** 2) if abs(mean) > 1e-10 else np.nan
        
        # 提取目录结构信息
        dir_path = os.path.dirname(file_path)
        warmup_dir = os.path.basename(dir_path)
        compress_dir = os.path.basename(os.path.dirname(dir_path))
        
        return (compress_dir, warmup_dir, file_path, 
                mean, std, variance, 
                normalized_std, normalized_variance)
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        # 返回错误信息用于记录
        return (os.path.basename(os.path.dirname(os.path.dirname(file_path))),
                os.path.basename(os.path.dirname(file_path)),
                file_path, 
                f"ERROR: {str(e)}", np.nan, np.nan, np.nan, np.nan)

def main():
    parser = argparse.ArgumentParser(description='批量计算分子动力学模拟的局部熵统计量')
    parser.add_argument('root_dir', help='根目录路径')
    parser.add_argument('-o', '--output', default='entropy_stats.csv', help='输出CSV文件路径')
    parser.add_argument('--cutoff', type=float, default=6.0, help='邻居搜索截断半径(Å)')
    parser.add_argument('--sigma', type=float, default=0.2, help='高斯平滑参数(Å)')
    parser.add_argument('--local-density', action='store_true', help='使用局部密度校正')
    parser.add_argument('--average', action='store_true', help='计算邻域平均熵')
    parser.add_argument('--avg-cutoff', type=float, default=6.0, help='邻域平均截断半径(Å)')
    parser.add_argument('--skip-existing', action='store_true', help='跳过已存在的输出文件')
    parser.add_argument('-j', '--jobs', type=int, default=0, 
                        help='并行进程数 (0=所有核心, -1=比核心少一个, 正数=指定数量)')
    
    args = parser.parse_args()

    # 检查输出文件是否已存在
    if args.skip_existing and os.path.exists(args.output):
        print(f"输出文件 {args.output} 已存在，跳过处理")
        return

    # 查找所有model.xyz文件
    xyz_files = []
    for root, dirs, files in os.walk(args.root_dir):
        for file in files:
            if file == 'model.xyz':
                xyz_files.append(os.path.join(root, file))
    
    total_files = len(xyz_files)
    if total_files == 0:
        print("未找到任何model.xyz文件")
        return
    
    # 配置并行处理
    if args.jobs == 0:
        num_workers = mp.cpu_count()
    elif args.jobs == -1:
        num_workers = max(1, mp.cpu_count() - 1)
    else:
        num_workers = max(1, min(args.jobs, mp.cpu_count()))
    
    print(f"找到 {total_files} 个需要处理的文件, 使用 {num_workers} 个进程...")
    
    # 创建部分函数用于传递args
    process_func = partial(process_single_file, args=args)
    
    # 使用进度条可视化并行处理（需要安装tqdm）
    use_tqdm = False
    try:
        from tqdm import tqdm
        use_tqdm = True
        print("使用tqdm显示进度...")
    except ImportError:
        print("未安装tqdm，使用简单计数器...")
    
    results = []
    
    # 并行处理文件
    with mp.Pool(processes=num_workers) as pool:
        if use_tqdm:
            for result in tqdm(pool.imap(process_func, xyz_files), total=total_files):
                results.append(result)
        else:
            # 不使用tqdm时打印简单计数
            for i, result in enumerate(pool.imap(process_func, xyz_files)):
                results.append(result)
                print(f"已完成 {i+1}/{total_files} 个文件...", end='\r')
            print("")  # 换行
    
    # 输出结果到CSV
    df = pd.DataFrame(results, columns=[
        '压缩目录', '预热目录', '文件路径', 
        '熵平均值', '熵标准差', '熵方差', 
        '归一化标准差', '归一化方差'
    ])
    df.to_csv(args.output, index=False)
    
    # 分析处理结果
    errors = df[df['熵平均值'].astype(str).str.startswith('ERROR')]
    success = df[~df['熵平均值'].astype(str).str.startswith('ERROR')]
    
    # 输出总结信息
    success_rate = len(success)/total_files*100
    print(f"\n处理完成! 成功率: {success_rate:.1f}%")
    print(f"成功处理: {len(success)}/{total_files} 文件")
    print(f"失败文件: {len(errors)} 个 - 查看CSV获取详情")
    print(f"结果保存至: {args.output}")
    
    # 保存失败文件列表
    if len(errors) > 0:
        error_file = os.path.splitext(args.output)[0] + "_errors.csv"
        errors.to_csv(error_file, index=False)
        print(f"失败文件列表保存至: {error_file}")

if __name__ == "__main__":
    # 在Windows上多进程需要这个保护
    mp.freeze_support()
    main()

#python local_entropy_batch.py C:\Users\USTC\Desktop\杂项\x\x -o C:\Users\USTC\Desktop\杂项\x\entropy_cor_925.csv    --cutoff 9.25 --avg-cutoff 9.25 --sigma 0.15 --local-density --jobs -1    