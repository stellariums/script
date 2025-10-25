import numpy as np
import argparse
import os
import csv
from ovito.io import import_file
from collections import deque
import itertools  # 新增：用于参数组合迭代

# 新增：扫描范围配置
BINS_RANGE = range(10,11)  # 2×2×2 到 15×15×15
N_RANGE = range(20,21)     # n值从2到10

def compute_atom_distribution(input_file: str, num_bins: tuple = (10, 10, 10), 
                              atomic_mass: float = 12.01, n: int = 2):
    """
    计算碳原子体系的空间分布和均匀性指标
    :param input_file: 输入文件路径
    :param num_bins: 空间分割数 (默认10x10x10)
    :param atomic_mass: 碳原子质量=12.01 g/mol
    :param n: 均匀性敏感度参数 (默认2)
    :return: 原子分布矩阵, 晶格尺寸, 总原子数, 密度网格, 整体密度, 均匀性指标
    """
    # 1. 验证分割参数有效性
    if len(num_bins) != 3 or any(not isinstance(n, int) or n <= 0 for n in num_bins):
        raise ValueError("分割参数必须为3个正整数，例如 (5,5,5)")
    
    # 2. 加载数据
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"文件不存在: {input_file}")
    pipeline = import_file(input_file)
    data = pipeline.compute()
    
    # 3. 获取原子坐标和晶胞
    positions = data.particles.positions
    cell = data.cell
    bounds = np.array([
        np.linalg.norm(cell.matrix[0, :3]),  # X
        np.linalg.norm(cell.matrix[1, :3]),  # Y
        np.linalg.norm(cell.matrix[2, :3])   # Z
    ])
    if np.any(bounds <= 0):
        raise ValueError("晶格尺寸无效，请检查输入文件")
    
    # 4. 周期性边界处理
    positions = positions - np.floor(positions / bounds) * bounds
    
    # 5. 三维直方图统计原子分布
    ranges = [(0, bounds[0]), (0, bounds[1]), (0, bounds[2])]
    atom_counts, _ = np.histogramdd(positions, bins=num_bins, range=ranges)
    
    # 6. 密度计算（原子质量固定为碳）
    total_atoms = data.particles.count
    bin_volume = np.prod([b / n for b, n in zip(bounds, num_bins)])  # 单个小立方体体积(Å³)
    density_grid = (atom_counts * atomic_mass) / (bin_volume * 6.022e23) * 1e24  # → g/cm³
    crystal_density = (total_atoms * atomic_mass) / (np.prod(bounds) * 6.022e23) * 1e24  # 整体密度
    
    # 7. 计算均匀性指标
    N = np.prod(num_bins)  # 小立方体总数
    relative_density = density_grid - crystal_density  # ρ_i-ρ_total
    sum_term = np.sum((relative_density-1)**n) / N  # (1/N)Σ(ρ_i-ρ_total)^n
    uniformity_index = crystal_density / ((sum_term)**(1/n)+1)
    return atom_counts, bounds, total_atoms, density_grid, crystal_density, uniformity_index

def find_model_xyz_files(root_dir: str):
    """使用BFS遍历目录树查找所有model.xyz文件"""
    file_paths = []
    queue = deque([root_dir])
    
    while queue:
        current_dir = queue.popleft()
        for entry in os.listdir(current_dir):
            full_path = os.path.join(current_dir, entry)
            if os.path.isdir(full_path):
                queue.append(full_path)
            elif entry == "model.xyz":
                file_paths.append(full_path)
    return file_paths

def main():
    parser = argparse.ArgumentParser(description='批量处理碳晶体密度分布及均匀性指标')
    parser.add_argument('--root', required=True, help='根目录路径')
    parser.add_argument('--output', required=True, help='输出CSV文件路径')
    parser.add_argument('--atomic_mass', type=float, default=12.01, 
                        help='碳原子质量 (默认12.01 g/mol)')
    args = parser.parse_args()

    # 查找所有model.xyz文件
    xyz_files = find_model_xyz_files(args.root)
    print(f"在 {args.root} 中找到 {len(xyz_files)} 个model.xyz文件")
    print(f"扫描参数配置: bins={list(BINS_RANGE)}，n={list(N_RANGE)}")

    # 准备CSV输出
    with open(args.output, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['FilePath', 'Bins', 'n', 'Lattice_X(Å)', 'Lattice_Y(Å)', 'Lattice_Z(Å)', 
                     'Total_Atoms', 'Overall_Density(g/cm³)', 'Uniformity_Index',
                     'Min_Density(g/cm³)', 'Max_Density(g/cm³)', 'Mean_Density(g/cm³)', 
                     'Std_Density(g/cm³)', 'Fluctuation_Range(%)', 'Error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        total_combinations = len(xyz_files) * len(BINS_RANGE) * len(N_RANGE)
        current_count = 0
        
        # 处理每个文件
        for i, file_path in enumerate(xyz_files):
            print(f"处理文件中 ({i+1}/{len(xyz_files)}): {file_path}")
            
            try:
                # 加载文件数据（只加载一次）
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"文件不存在: {file_path}")
                pipeline = import_file(file_path)
                data = pipeline.compute()
                
                # 获取原子坐标和晶胞（只计算一次）
                positions = data.particles.positions
                cell = data.cell
                bounds = np.array([
                    np.linalg.norm(cell.matrix[0, :3]),
                    np.linalg.norm(cell.matrix[1, :3]),
                    np.linalg.norm(cell.matrix[2, :3])
                ])
                if np.any(bounds <= 0):
                    raise ValueError("晶格尺寸无效，请检查输入文件")
                
                # 周期性边界处理
                positions = positions - np.floor(positions / bounds) * bounds
                total_atoms = data.particles.count
                
                # 新增：扫描不同参数组合
                for bins_val, n_val in itertools.product(BINS_RANGE, N_RANGE):
                    current_count += 1
                    num_bins = (bins_val, bins_val, bins_val)  # 等比例分割
                    bin_config = f"{bins_val}x{bins_val}x{bins_val}"
                    
                    print(f" 扫描组合 [{current_count}/{total_combinations}]: "
                          f"bins={bin_config}, n={n_val}")
                    
                    try:
                        # 执行计算（使用预加载的数据）
                        result = compute_atom_distribution(
                            input_file=None,  # 不使用文件路径
                            num_bins=num_bins,
                            atomic_mass=args.atomic_mass,
                            n=n_val,
                            # 新增：传入预加载数据
                            precomputed_data={
                                'positions': positions,
                                'bounds': bounds,
                                'total_atoms': total_atoms
                            }
                        )
                        atom_counts, _, _, density_grid, crystal_density, uniformity_index = result
                        
                        # 计算密度统计
                        min_density = np.min(density_grid)
                        max_density = np.max(density_grid)
                        mean_density = np.mean(density_grid)
                        std_density = np.std(density_grid)
                        fluctuation_range = 100 * (max_density - min_density) / (2 * crystal_density)
                        
                        # 写入结果
                        writer.writerow({
                            'FilePath': file_path,
                            'Bins': bin_config,
                            'n': n_val,
                            'Overall_Density(g/cm³)': f"{crystal_density:.6f}",
                            'Uniformity_Index': f"{uniformity_index:.6f}",
                        })
                        csvfile.flush()  # 实时写入
                        
                    except Exception as e:
                        print(f"  参数组合失败: bins={bin_config}, n={n_val}, 错误: {str(e)}")
                        writer.writerow({
                            'FilePath': file_path,
                            'Bins': bin_config,
                            'n': n_val,
                            'Error': f"参数组合错误: {str(e)}"
                        })
                
            except Exception as e:
                print(f"文件处理失败: {str(e)}")
                # 标记所有参数组合为失败
                for bins_val, n_val in itertools.product(BINS_RANGE, N_RANGE):
                    writer.writerow({
                        'FilePath': file_path,
                        'Bins': f"{bins_val}x{bins_val}x{bins_val}",
                        'n': n_val,
                        'Error': f"文件加载失败: {str(e)}"
                    })
    
    print(f"扫描完成! 共处理 {len(xyz_files)} 个文件, {total_combinations} 个参数组合")
    print(f"结果已保存到 {args.output}")

# 修改compute_atom_distribution以支持预加载数据
def compute_atom_distribution(input_file: str = None, num_bins: tuple = (10, 10, 10), 
                              atomic_mass: float = 12.01, n: int = 2,
                              precomputed_data: dict = None):
    """
    修改：支持预加载数据
    """
    # 1. 验证分割参数有效性
    if len(num_bins) != 3 or any(not isinstance(nb, int) or nb <= 0 for nb in num_bins):
        raise ValueError(f"分割参数必须为3个正整数: {num_bins}")
    
    # 2. 获取数据（新增预加载支持）
    if precomputed_data:
        positions = precomputed_data['positions']
        bounds = precomputed_data['bounds']
        total_atoms = precomputed_data['total_atoms']
    else:
        if not input_file or not os.path.exists(input_file):
            raise FileNotFoundError(f"文件不存在: {input_file}")
        pipeline = import_file(input_file)
        data = pipeline.compute()
        positions = data.particles.positions
        cell = data.cell
        bounds = np.array([
            np.linalg.norm(cell.matrix[0, :3]),
            np.linalg.norm(cell.matrix[1, :3]),
            np.linalg.norm(cell.matrix[2, :3])
        ])
        total_atoms = data.particles.count
        if np.any(bounds <= 0):
            raise ValueError("晶格尺寸无效，请检查输入文件")
        # 周期性边界处理
        positions = positions - np.floor(positions / bounds) * bounds
    
    # 3. 三维直方图统计原子分布
    ranges = [(0, bounds[0]), (0, bounds[1]), (0, bounds[2])]
    atom_counts, _ = np.histogramdd(positions, bins=num_bins, range=ranges)
    
    # 4. 密度计算
    bin_volume = np.prod([b / nb for b, nb in zip(bounds, num_bins)])  # 单个小立方体体积(Å³)
    density_grid = (atom_counts * atomic_mass) / (bin_volume * 6.022e23) * 1e24  # → g/cm³
    crystal_density = (total_atoms * atomic_mass) / (np.prod(bounds) * 6.022e23) * 1e24  # 整体密度
    
    # 5. 计算均匀性指标
    N = np.prod(num_bins)  # 小立方体总数
    relative_density = density_grid / crystal_density  # ρ_i/ρ_total
    sum_term = np.sum(relative_density**n) / N  # (1/N)Σ(ρ_i/ρ_total)^n
    uniformity_index = crystal_density / sum_term  # 修改后公式
    
    return atom_counts, bounds, total_atoms, density_grid, crystal_density, uniformity_index

if __name__ == "__main__":
    main()