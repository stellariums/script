import numpy as np
import argparse
from ovito.io import import_file
import os

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
    uniformity_index = crystal_density / (sum_term)**(1/n)
    
    return atom_counts, bounds, total_atoms, density_grid, crystal_density, uniformity_index

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="计算碳晶体密度分布及均匀性指标")
    parser.add_argument("--input", default="model.xyz", help="输入XYZ文件路径")
    parser.add_argument("--bins", nargs=3, type=int, default=[10,10,10], 
                        help="空间分割数 (例如 '--bins 5 5 5')")
    parser.add_argument("--atomic_mass", type=float, default=12.01, 
                        help="碳原子质量=12.01 g/mol (可覆盖)")
    parser.add_argument("--n", type=int, default=2, 
                        help="均匀性敏感度参数n (默认2)")
    args = parser.parse_args()

    try:
        # 执行计算
        result = compute_atom_distribution(args.input, tuple(args.bins), args.atomic_mass, args.n)
        atom_counts, bounds, total_atoms, density_grid, crystal_density, uniformity_index = result
        
        # 计算密度范围（新增核心功能）
        min_density = np.min(density_grid)
        max_density = np.max(density_grid)
        mean_density = np.mean(density_grid)
        std_density = np.std(density_grid)
        
        # 输出结果
        print("=" * 60)
        print(f"空间分割: {args.bins[0]}x{args.bins[1]}x{args.bins[2]} = {np.prod(args.bins)} 个小立方体")
        print(f"晶格尺寸: X={bounds[0]:.2f} Å, Y={bounds[1]:.2f} Å, Z={bounds[2]:.2f} Å")
        print(f"总原子数: {total_atoms} (全碳体系)")
        print(f"碳原子质量: {args.atomic_mass:.2f} g/mol")
        print(f"整体密度: {crystal_density:.4f} g/cm³")
        print(f"均匀性指标(n={args.n}): {uniformity_index:.4f}")
        
        # 新增密度范围输出（包含统计指标）
        print("\n小立方体密度统计:")
        print(f"- 最小值: {min_density:.4f} g/cm³")
        print(f"- 最大值: {max_density:.4f} g/cm³")
        print(f"- 平均值: {mean_density:.4f} g/cm³")
        print(f"- 标准差: {std_density:.4f} g/cm³")
        print(f"- 波动范围: ±{100*(max_density - min_density)/(2*crystal_density):.2f}% (相对于整体密度)")
        
                        
    except Exception as e:
        print(f"执行失败: {str(e)}")


#python division.py --input model.xyz --bins 20 20 20 --n 2
