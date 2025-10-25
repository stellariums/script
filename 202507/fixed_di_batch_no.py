import numpy as np
import argparse
import os
import csv
from ovito.io import import_file
from collections import deque

def compute_atom_distribution(input_file: str, bin_size: float = 5.0, 
                             atomic_mass: float = 12.01, n: int = 2):
    """
    计算碳原子体系的空间分布和均匀性指标
    :param input_file: 输入文件路径
    :param bin_size: 小立方体边长(Å)
    :param atomic_mass: 碳原子质量=12.01 g/mol
    :param n: 均匀性敏感度参数 (默认2)
    :return: 原子分布矩阵, 晶格尺寸, 总原子数, 密度网格, 整体密度, 均匀性指标, 分割数量, 非空区域密度统计
    """
    # 1. 验证小立方体边长有效性
    if bin_size <= 0:
        raise ValueError("小立方体边长必须为正数")
    
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
    
    # 4. 计算每个维度的小立方体数量（向下取整）
    num_bins = tuple(max(1, int(bound / bin_size)) for bound in bounds)
    
    # 5. 周期性边界处理
    positions = positions - np.floor(positions / bounds) * bounds
    
    # 6. 三维直方图统计原子分布
    ranges = [(0, bounds[0]), (0, bounds[1]), (0, bounds[2])]
    atom_counts, _ = np.histogramdd(positions, bins=num_bins, range=ranges)
    
    # 7. 密度计算（原子质量固定为碳）
    total_atoms = data.particles.count
    bin_volumes = np.prod([bounds[i] / num_bins[i] for i in range(3)])  # 单个小立方体体积(Å³)
    density_grid = (atom_counts * atomic_mass) / (bin_volumes * 6.022e23) * 1e24  # → g/cm³
    crystal_density = (total_atoms * atomic_mass) / (np.prod(bounds) * 6.022e23) * 1e24  # 整体密度
    
    # 8. 计算均匀性指标（忽略原子数为0的格子）
    non_zero_mask = atom_counts > 0
    non_zero_densities = density_grid[non_zero_mask]  # 非空区域的密度值
    N_nonzero = np.sum(non_zero_mask)
    
    if N_nonzero > 0:
        relative_density = non_zero_densities / crystal_density
        sum_term = np.sum((relative_density - 1)**n) / N_nonzero
        uniformity_index = sum_term**(1/n)
    else:
        uniformity_index = float('nan')
    
    # 9. 非空区域的密度统计
    min_density = np.min(non_zero_densities) if non_zero_densities.size > 0 else float('nan')
    max_density = np.max(non_zero_densities) if non_zero_densities.size > 0 else float('nan')
    mean_density = np.mean(non_zero_densities) if non_zero_densities.size > 0 else float('nan')
    std_density = np.std(non_zero_densities) if non_zero_densities.size > 0 else float('nan')
    
    return atom_counts, bounds, total_atoms, density_grid, crystal_density, uniformity_index, num_bins, min_density, max_density, mean_density, std_density

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
    parser.add_argument('--bin_size', type=float, default=5.0, 
                        help='小立方体边长(Å) (例如 --bin_size 5.0)')
    parser.add_argument('--atomic_mass', type=float, default=12.01, 
                        help='碳原子质量 (默认12.01 g/mol)')
    parser.add_argument('--n', type=int, default=2, 
                        help='均匀性敏感度参数n (默认2)')
    args = parser.parse_args()

    # === 新增：输出目录权限检查 ===
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not os.access(output_dir if output_dir else ".", os.W_OK):
        raise PermissionError(f"目录 {output_dir} 不可写")

    # 查找所有model.xyz文件
    xyz_files = find_model_xyz_files(args.root)
    print(f"在 {args.root} 中找到 {len(xyz_files)} 个model.xyz文件")

    # 准备CSV输出
    with open(args.output, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = [
            'FilePath', 'Bins', 'Lattice_X(Å)', 'Lattice_Y(Å)', 'Lattice_Z(Å)', 
            'Total_Atoms', 'Overall_Density(g/cm³)', 'Uniformity_Index',
            'Min_Density(g/cm³)', 'Max_Density(g/cm³)', 'Mean_Density(g/cm³)', 
            'Std_Density(g/cm³)', 'Fluctuation_Range(%)', 'Error',
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
                    parts = file_path.split(os.sep)
                    for part in parts:
                        if part.startswith('warmup_'):
                            try:
                                warmup_n = int(part.split('_')[1])
                            except (IndexError, ValueError):
                                warmup_n = '解析错误'
                            break
                
                # 执行计算
                result = compute_atom_distribution(file_path, args.bin_size, args.atomic_mass, args.n)
                atom_counts, bounds, total_atoms, density_grid, crystal_density, uniformity_index, num_bins, min_density, max_density, mean_density, std_density = result
                
                # 波动范围计算（保护除零错误）
                if not np.isnan(min_density) and crystal_density > 0:
                    fluctuation_range = 100 * (max_density - min_density) / (2 * crystal_density)
                else:
                    fluctuation_range = 0
                
                warmup_n1 = 2000 + 200 * warmup_n if isinstance(warmup_n, int) else 'N/A'
                
                # 写入结果
                row_data = {
                    'FilePath': file_path,
                    'Bins': f"{num_bins[0]}x{num_bins[1]}x{num_bins[2]}",
                    'Lattice_X(Å)': f"{bounds[0]:.4f}",
                    'Lattice_Y(Å)': f"{bounds[1]:.4f}",
                    'Lattice_Z(Å)': f"{bounds[2]:.4f}",
                    'Total_Atoms': total_atoms,
                    'Overall_Density(g/cm³)': f"{crystal_density:.6f}",
                    'Uniformity_Index': f"{uniformity_index:.6f}" if not np.isnan(uniformity_index) else 'NaN',
                    'Min_Density(g/cm³)': f"{min_density:.6f}" if not np.isnan(min_density) else 'NaN',
                    'Max_Density(g/cm³)': f"{max_density:.6f}" if not np.isnan(max_density) else 'NaN',
                    'Mean_Density(g/cm³)': f"{mean_density:.6f}" if not np.isnan(mean_density) else 'NaN',
                    'Std_Density(g/cm³)': f"{std_density:.6f}" if not np.isnan(std_density) else 'NaN',
                    'Fluctuation_Range(%)': f"{fluctuation_range:.4f}",
                    'Error': '',
                    '体系密度': f"{crystal_density:.6f}",
                    'Warmup_n': warmup_n1,
                    'uniformity_index': f"{uniformity_index:.6f}" if not np.isnan(uniformity_index) else 'NaN'
                }
                writer.writerow(row_data)
                
            except Exception as e:
                print(f"处理失败: {str(e)}")
                writer.writerow({
                    'FilePath': file_path,
                    'Error': str(e),
                    '体系密度': '',
                    'Warmup_n': '处理失败',
                    'uniformity_index': ''
                })
    
    print(f"处理完成! 结果已保存到 {args.output}")

if __name__ == "__main__":
    main()