import numpy as np
import argparse
import os
import csv
from ovito.io import import_file
from collections import deque

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
    relative_density = density_grid / crystal_density  # ρ_i/ρ_total
    sum_term = np.sum((relative_density-1)**n) / N  # (1/N)Σ(ρ_i/ρ_total-1)^n
    uniformity_index = sum_term**(1/n)
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
    parser.add_argument('--bins', nargs=3, type=int, default=[10,10,10], 
                        help='空间分割数 (例如 --bins 5 5 5)')
    parser.add_argument('--atomic_mass', type=float, default=12.01, 
                        help='碳原子质量 (默认12.01 g/mol)')
    parser.add_argument('--n', type=int, default=2, 
                        help='均匀性敏感度参数n (默认2)')
    args = parser.parse_args()

    # 查找所有model.xyz文件
    xyz_files = find_model_xyz_files(args.root)
    print(f"在 {args.root} 中找到 {len(xyz_files)} 个model.xyz文件")

    # 准备CSV输出 - 修改开始
    with open(args.output, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = [
            'FilePath', 'Bins', 'Lattice_X(Å)', 'Lattice_Y(Å)', 'Lattice_Z(Å)', 
            'Total_Atoms', 'Overall_Density(g/cm³)', 'Uniformity_Index',
            'Min_Density(g/cm³)', 'Max_Density(g/cm³)', 'Mean_Density(g/cm³)', 
            'Std_Density(g/cm³)', 'Fluctuation_Range(%)', 'Error',
            # 以下是新增的三列
            '体系密度',  # 新增列1 - 体系密度
            'Warmup_n', # 新增列2 - warmup_n中的n
            'uniformity_index'  # 新增列3 - 均匀性指标
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # 处理每个文件
        for i, file_path in enumerate(xyz_files):
            print(f"处理文件中 ({i+1}/{len(xyz_files)}): {file_path}")
            try:
                # 提取warmup_n中的n值
                warmup_n = '未找到'
                if 'warmup_' in file_path:
                    # 从文件路径中提取warmup_后面的数字
                    parts = file_path.split(os.sep)
                    for part in parts:
                        if part.startswith('warmup_'):
                            try:
                                warmup_n = int(part.split('_')[1])
                            except (IndexError, ValueError):
                                warmup_n = '解析错误'
                            break
                
                # 执行计算
                result = compute_atom_distribution(file_path, tuple(args.bins), args.atomic_mass, args.n)
                atom_counts, bounds, total_atoms, density_grid, crystal_density, uniformity_index = result
                
                # 计算密度统计
                min_density = np.min(density_grid)
                max_density = np.max(density_grid)
                mean_density = np.mean(density_grid)
                std_density = np.std(density_grid)
                fluctuation_range = 100 * (max_density - min_density) / (2 * crystal_density)
                warmup_n1=2000+200*warmup_n
                # 写入结果 - 添加新列数据
                writer.writerow({
                    'FilePath': file_path,
                    'Bins': f"{args.bins[0]}x{args.bins[1]}x{args.bins[2]}",
                    'Lattice_X(Å)': f"{bounds[0]:.4f}",
                    'Lattice_Y(Å)': f"{bounds[1]:.4f}",
                    'Lattice_Z(Å)': f"{bounds[2]:.4f}",
                    'Total_Atoms': total_atoms,
                    'Overall_Density(g/cm³)': f"{crystal_density:.6f}",
                    'Uniformity_Index': f"{uniformity_index:.6f}",
                    'Min_Density(g/cm³)': f"{min_density:.6f}",
                    'Max_Density(g/cm³)': f"{max_density:.6f}",
                    'Mean_Density(g/cm³)': f"{mean_density:.6f}",
                    'Std_Density(g/cm³)': f"{std_density:.6f}",
                    'Fluctuation_Range(%)': f"{fluctuation_range:.4f}",
                    'Error': '',
                    
                    # 以下是新增的三列数据

                    
                    '体系密度': f"{crystal_density:.6f}",  # 使用计算得到的整体密度
                    'Warmup_n': warmup_n1,                # 从文件路径中提取的warmup_n值
                    'uniformity_index': f"{uniformity_index:.6f}"  # 均匀性指标
                })
                
            except Exception as e:
                print(f"处理失败: {str(e)}")
                writer.writerow({
                    'FilePath': file_path,
                    'Error': str(e),
                    
                    # 出错时也需要添加新列（设为空或默认值）
                    '体系密度': '',
                    'Warmup_n': '处理失败',
                    'uniformity_index': ''
                })
    # 修改结束
    
    print(f"处理完成! 结果已保存到 {args.output}")

if __name__ == "__main__":
    main()


#python di_batch.py --root C:\Users\USTC\Desktop\杂项\x\x --output C:\Users\USTC\Desktop\杂项\x\density.csv --bins 10 10 10 --n 2
