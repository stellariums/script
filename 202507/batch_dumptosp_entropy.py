###导入一系列dump.xyz文件，输出某一帧体系下的密度，sp2、sp3占比,sp2/sp3,结晶度，以及局部熵信息
###使用方式：python script.py --root C:/Users/USTC/Desktop/data/y/y -o output.csv
#!/usr/bin/env python3
import numpy as np
import sys
import argparse
import os
import glob
import pandas as pd
from ovito.io import import_file
from ovito.modifiers import CreateBondsModifier, CoordinationAnalysisModifier, IdentifyDiamondModifier
from ovito.data import CutoffNeighborFinder, DataCollection
from ovito.modifiers import ComputePropertyModifier
import multiprocessing as mp
from functools import partial
import warnings
from tqdm import tqdm
import csv
import time
import threading
from multiprocessing import Lock

# 忽略Ovito的警告
warnings.filterwarnings("ignore", module="ovito")

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
    use_local density: 是否使用局部密度校正
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
        
        # 数值积分 - 使用 trapezoid
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

def analyze_frame(file_path, frame_index, entropy_options=None, quiet=False):
    """
    分析GPUMD生成的dump.xyz文件的指定帧，可选的熵计算
    
    参数:
    file_path: 输入文件路径
    frame_index: 分析的帧索引
    entropy_options: 熵计算选项字典
    quiet: 是否静默模式（不打印进度）
    """
    try:
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
        
        if not quiet:
            print(f"分析文件: {os.path.basename(file_path)} - 帧 {actual_index}/{num_frames-1}")
        
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
            "File": file_path,
            "Frame": actual_index,
            "Density": density,
            "sp2_Percent": sp2_percent,
            "sp3_Percent": sp3_percent,
            "sp3_sp2_Ratio": sp3_sp2_ratio,
            "Crystallinity": crystallinity,
            "Error": None  # 添加默认错误字段
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
            
            # 计算熵统计信息
            entropy_min = np.min(entropy)
            entropy_max = np.max(entropy)
            entropy_mean = np.mean(entropy)
            entropy_std = np.std(entropy)
            
            # 添加到熵统计字典
            entropy_stats.update({
                "Entropy_Min": entropy_min,
                "Entropy_Max": entropy_max,
                "Entropy_Mean": entropy_mean,
                "Entropy_Std": entropy_std,
                "Entropy_Norm_Std": entropy_std / entropy_mean if entropy_mean != 0 else np.nan
            })
            
            # 添加到主结果字典
            results.update(entropy_stats)
            
            if not quiet:
                print(f"  熵统计: 均值={entropy_mean:.4f}, 标准差={entropy_std:.4f}")
        
        return results
    
    except Exception as e:
        print(f"分析 {file_path} 帧 {frame_index} 时出错: {str(e)}")
        return {
            "File": file_path,
            "Frame": frame_index,
            "Error": str(e)
        }

def find_dump_files(root_dir, quiet=False):
    """
    递归查找所有compress/warmup目录中的dump.xyz文件
    
    参数:
    root_dir: 根目录路径
    quiet: 是否静默模式（不显示找到的文件）
    
    返回:
    排序后的dump.xyz文件列表
    """
    dump_files = []
    
    # 查找所有compress_目录
    compress_dirs = sorted(glob.glob(os.path.join(root_dir, "compress_*")))
    
    for compress_dir in compress_dirs:
        # 在每个compress目录下查找warmup_目录
        warmup_dirs = sorted(glob.glob(os.path.join(compress_dir, "warmup_*")))
        
        for warmup_dir in warmup_dirs:
            dump_path = os.path.join(warmup_dir, "dump.xyz")
            if os.path.exists(dump_path):
                dump_files.append(dump_path)
                if not quiet:
                    print(f"找到文件: {dump_path}")
    
    # 直接在根目录下查找compress_目录
    root_compress_dirs = sorted(glob.glob(os.path.join(root_dir, "compress_*", "warmup_*", "dump.xyz")))
    dump_files.extend(root_compress_dirs)
    
    # 确保不重复并排序
    dump_files = sorted(set(dump_files))
    
    return dump_files

def write_csv_header(output_file, entropy_enabled):
    """
    写入CSV文件表头
    
    参数:
    output_file: 输出文件路径
    entropy_enabled: 是否启用了熵计算
    """
    fieldnames = [
        "File", "Frame", "Density(g/cm³)", "sp2(%)", "sp3(%)", 
        "sp3/sp2 Ratio", "Crystallinity(%)"
    ]
    
    if entropy_enabled:
        fieldnames.extend([
            "Entropy Min", "Entropy Max", "Entropy Mean", 
            "Entropy Std", "Normalized Std"
        ])
    
    fieldnames.append("Error")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

def write_result_to_csv(output_file, result, entropy_enabled, lock=None):
    """
    将单个结果写入CSV文件，支持线程锁
    
    参数:
    output_file: 输出文件路径
    result: 单文件分析结果字典
    entropy_enabled: 是否启用了熵计算
    lock: 线程锁对象，用于多进程安全写入
    """
    # 准备写入的行数据
    row = {
        "File": result["File"],
        "Frame": result["Frame"],
        "Density(g/cm³)": result["Density"],
        "sp2(%)": result["sp2_Percent"],
        "sp3(%)": result["sp3_Percent"],
        "sp3/sp2 Ratio": result["sp3_sp2_Ratio"],
        "Crystallinity(%)": result["Crystallinity"],
        "Error": result["Error"]
    }
    
    # 添加熵统计信息（如果启用）
    if entropy_enabled:
        row.update({
            "Entropy Min": result.get("Entropy_Min", ""),
            "Entropy Max": result.get("Entropy_Max", ""),
            "Entropy Mean": result.get("Entropy_Mean", ""),
            "Entropy Std": result.get("Entropy_Std", ""),
            "Normalized Std": result.get("Entropy_Norm_Std", "")
        })
    
    # 获取锁（如果提供）
    if lock:
        lock.acquire()
    
    try:
        # 写入CSV文件
        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
    finally:
        # 释放锁（如果获取了）
        if lock:
            lock.release()

def process_wrapper(args, file_path, output_file, entropy_enabled, entropy_options, lock=None):
    """
    封装处理单个文件的函数，用于并行处理
    
    参数:
    args: 命令行参数对象
    file_path: 要处理的文件路径
    output_file: 输出CSV文件路径
    entropy_enabled: 是否启用熵计算
    entropy_options: 熵计算选项
    lock: 用于文件写入的锁对象
    """
    try:
        # 分析文件
        result = analyze_frame(
            file_path, 
            args.frame, 
            entropy_options, 
            args.quiet
        )
        
        # 将结果写入CSV
        write_result_to_csv(output_file, result, entropy_enabled, lock)
        
        return result
    except Exception as e:
        error_result = {
            "File": file_path,
            "Frame": args.frame,
            "Error": str(e)
        }
        write_result_to_csv(output_file, error_result, entropy_enabled, lock)
        return error_result

def batch_process(args, output_file):
    """
    批处理多个文件，保持文件顺序，边计算边写入CSV
    
    参数:
    args: 命令行参数
    output_file: 输出CSV文件路径
    
    返回:
    处理结果的DataFrame
    """
    # 递归查找所有dump.xyz文件
    dump_files = find_dump_files(args.root, args.quiet)
    
    if not dump_files:
        print("错误: 未找到任何dump.xyz文件")
        return None
    
    num_files = len(dump_files)
    if not args.quiet:
        print(f"找到 {num_files} 个dump.xyz文件")
    
    # 准备熵选项
    entropy_options = None
    entropy_enabled = args.entropy
    if entropy_enabled:
        entropy_options = {
            'cutoff': args.cutoff,
            'sigma': args.sigma,
            'use_local_density': args.local_density,
            'compute_average': args.average,
            'average_cutoff': args.avg_cutoff
        }
    
    # 创建输出目录（如果需要）
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 初始化CSV文件并写入表头
    write_csv_header(output_file, entropy_enabled)
    
    # 配置并行处理
    num_workers = 1
    if args.jobs > 1:
        num_workers = min(args.jobs, mp.cpu_count(), num_files)
    
    if num_workers > 1 and not args.quiet:
        print(f"使用 {num_workers} 个进程并行处理 {num_files} 个文件")
    
    # 创建锁对象用于多进程安全写入
    manager = mp.Manager()
    lock = manager.Lock()
    
    # 创建部分函数用于并行处理
    process_func = partial(
        process_wrapper,
        args,
        output_file=output_file,
        entropy_enabled=entropy_enabled,
        entropy_options=entropy_options,
        lock=lock
    )
    
    # 处理所有文件
    results = []
    
    if num_workers > 1:
        # 并行处理
        with mp.Pool(processes=num_workers) as pool:
            # 使用imap_unordered提高并行效率
            if not args.quiet:
                # 使用tqdm显示进度
                pbar = tqdm(total=num_files, desc="处理文件", unit="文件")
                
                # 使用imap_unordered并按顺序收集结果
                processed_results = []
                for result in pool.imap_unordered(process_func, dump_files):
                    processed_results.append(result)
                    pbar.update(1)
                
                pbar.close()
            else:
                # 静默模式
                processed_results = pool.map(process_func, dump_files)
            
            # 按原始顺序整理结果
            file_index_map = {file_path: idx for idx, file_path in enumerate(dump_files)}
            results = [None] * len(dump_files)
            for result in processed_results:
                file_path = result["File"]
                idx = file_index_map[file_path]
                results[idx] = result
    else:
        # 单进程处理
        if not args.quiet:
            for file_path in tqdm(dump_files, desc="处理文件", unit="文件"):
                result = process_func(file_path)
                results.append(result)
        else:
            for file_path in dump_files:
                result = process_func(file_path)
                results.append(result)
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 重命名列以提升可读性
    column_mapping = {
        "Density": "Density(g/cm³)",
        "sp2_Percent": "sp2(%)",
        "sp3_Percent": "sp3(%)",
        "sp3_sp2_Ratio": "sp3/sp2 Ratio",
        "Crystallinity": "Crystallinity(%)",
        "Entropy_Min": "Entropy Min",
        "Entropy_Max": "Entropy Max",
        "Entropy_Mean": "Entropy Mean",
        "Entropy_Std": "Entropy Std",
        "Entropy_Norm_Std": "Normalized Std"
    }
    df = df.rename(columns=column_mapping)
    
    # 重新排序列
    preferred_order = [
        "File", "Frame", "Density(g/cm³)", "sp2(%)", "sp3(%)", 
        "sp3/sp2 Ratio", "Crystallinity(%)", "Entropy Min", 
        "Entropy Max", "Entropy Mean", "Entropy Std", "Normalized Std",
        "Error"
    ]
    
    # 只保留实际存在的列
    ordered_columns = [col for col in preferred_order if col in df.columns]
    df = df[ordered_columns]
    
    return df

def main():
    # 配置命令行参数解析
    parser = argparse.ArgumentParser(description="分析GPUMD生成的dump.xyz文件")
    parser.add_argument("--root", required=True, help="包含compress和warmup目录的根目录")
    parser.add_argument("-f", "--frame", type=int, default=-1,
                        help="分析指定帧（0-based索引，负数表示倒数，默认：-1=最后一帧）")
    parser.add_argument("-o", "--output", default="analysis_results.csv",
                        help="CSV输出文件路径")
    
    # 批处理选项
    parser.add_argument("-j", "--jobs", type=int, default=1,
                        help="并行进程数（默认=1，单进程）")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="静默模式（不显示进度）")
    
    # 熵计算选项
    entropy_group = parser.add_argument_group('熵计算选项')
    entropy_group.add_argument('--entropy', action='store_true', 
                              help='启用局部熵计算')
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
    
    # 处理单个文件或批处理
    start_time = time.time()
    df = batch_process(args, args.output)
    
    if df is None:
        return
    
    # 保存最终的DataFrame（包含所有结果）
    df.to_csv(args.output, index=False)
    
    # 统计信息
    success = df[df['Error'].isna()]
    errors = df[~df['Error'].isna()]
    
    if not args.quiet:
        elapsed_time = time.time() - start_time
        print(f"\n分析完成! 用时: {elapsed_time:.2f} 秒")
        print(f"结果保存至: {args.output}")
        print(f"成功处理: {len(success)} 个文件")
        if not errors.empty:
            print(f"失败文件: {len(errors)} 个")
        
        if not success.empty and 'Entropy Mean' in success.columns:
            entropy_means = success['Entropy Mean']
            if entropy_means.notna().any():
                avg_entropy = entropy_means.mean()
                print(f"平均熵值: {avg_entropy:.6f} ± {entropy_means.std():.6f}")

if __name__ == "__main__":
    mp.freeze_support()
    main()