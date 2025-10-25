import numpy as np
import sys
import os
from ase import Atoms
from ase.neighborlist import NeighborList
from scipy.signal import find_peaks
from scipy.integrate import simpson

def parse_custom_xyz(xyz_file):
    """
    手动解析自定义的 .xyz 文件
    格式:
        第一行: 原子总数
        第二行: Lattice="ax ay az bx by bz cx cy cz" ...
        后续行: 原子数据 (符号可能是数字)
    """
    with open(xyz_file, 'r') as f:
        # 第一行: 原子总数
        num_atoms = int(f.readline().strip())
        
        # 第二行: 元数据
        header = f.readline().strip()
        
        # 初始化变量
        lattice_vectors = None
        origin = np.zeros(3)
        positions = []
        symbols = []
        
        # 解析晶格信息
        lattice_start = header.find('Lattice="')
        if lattice_start >= 0:
            lattice_start += 9
            lattice_end = header.find('"', lattice_start)
            if lattice_end > lattice_start:
                lattice_str = header[lattice_start:lattice_end]
                lattice_values = [float(x) for x in lattice_str.split()]
                lattice_vectors = np.array(lattice_values).reshape(3, 3)
        
        # 解析属性定义 (格式: Properties=species:S:1:pos:R:3)
        species_index = 0
        pos_index = 1
        
        props_start = header.find('Properties=')
        if props_start >= 0:
            props_str = header[props_start + 11:].split(' ')[0]
            props = props_str.split(':')
            
            # 确定属性列的位置
            col_index = 0
            for i in range(0, len(props), 3):
                if i + 2 >= len(props):
                    break
                    
                prop_name = props[i]
                num_cols = int(props[i+2])
                
                if prop_name == "species":
                    species_index = col_index
                elif prop_name == "pos":
                    pos_index = col_index
                
                col_index += num_cols
        
        # 读取原子数据
        for i in range(num_atoms):
            data = f.readline().split()
            if not data:
                continue
                
            # 获取元素符号 (转换为碳)
            symbols.append('C')  # 强制所有原子为碳
            
            # 获取位置
            pos = [float(x) for x in data[pos_index:pos_index+3]]
            positions.append(pos)
    
    # 创建 ASE Atoms 对象
    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=lattice_vectors,
        pbc=True
    )
    
    return atoms

def calculate_sp3_ratio(atoms, cutoff=1.85, angle_tolerance=15):
    """
    使用ASE高效计算碳体系中sp3杂化原子的比例
    参数:
        atoms: ASE原子对象
        cutoff: 近邻判断距离阈值 (Å)
        angle_tolerance: 键角容忍度 (±度)
    返回:
        sp3杂化原子比例
        每个原子的杂化类型
        sp3原子索引
    """
    # 计算碳-碳键的理想角度
    SP3_IDEAL_ANGLE = 109.5
    SP2_IDEAL_ANGLE = 120.0
    
    num_atoms = len(atoms)
    
    # 初始化结果存储
    hybridization_types = ['other'] * num_atoms
    sp3_atom_indices = []
    
    # 构建近邻列表
    nl = NeighborList(cutoffs=[cutoff/2] * num_atoms, 
                     skin=0.0, 
                     self_interaction=False,
                     bothways=True)
    nl.update(atoms)
    
    # 遍历每个原子
    for i in range(num_atoms):
        # 获取近邻原子索引和位移向量
        indices, offsets = nl.get_neighbors(i)
        num_neighbors = len(indices)
        
        if num_neighbors < 3:
            continue
            
        # 计算所有键角
        bond_angles = []
        vectors = []
        
        # 获取所有键向量
        for j, offset in zip(indices, offsets):
            r = atoms.positions[j] + np.dot(offset, atoms.get_cell()) - atoms.positions[i]
            vectors.append(r)
        
        # 计算所有键对之间的角度
        for m in range(len(vectors)):
            for n in range(m + 1, len(vectors)):
                dot_product = np.dot(vectors[m], vectors[n])
                norm_product = np.linalg.norm(vectors[m]) * np.linalg.norm(vectors[n])
                angle = np.degrees(np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0)))
                bond_angles.append(angle)
        
        # 统计接近理想角度的键角数
        sp3_angles = sum(1 for angle in bond_angles 
                         if abs(angle - SP3_IDEAL_ANGLE) < angle_tolerance)
        sp2_angles = sum(1 for angle in bond_angles 
                         if abs(angle - SP2_IDEAL_ANGLE) < angle_tolerance)
        
        # 确定杂化类型
        if num_neighbors == 4 and sp3_angles >= 3:
            hybridization_types[i] = 'sp3'
            sp3_atom_indices.append(i)
        elif num_neighbors == 3 and sp2_angles >= 2:
            hybridization_types[i] = 'sp2'
    
    # 计算sp3比例
    sp3_count = len(sp3_atom_indices)
    sp3_ratio = sp3_count / num_atoms if num_atoms > 0 else 0.0
    
    return sp3_ratio, hybridization_types, sp3_atom_indices

def calculate_crystallinity(atoms, wavelength=1.5406, two_theta_range=(10, 80), num_points=1000):
    """
    计算碳体系的结晶度
    参数:
        atoms: ASE原子对象
        wavelength: X射线波长(Å) - Cu Kα辐射
        two_theta_range: 2θ扫描范围(度)
        num_points: 生成的衍射点数
    返回:
        结晶度百分比
    """
    # 生成XRD衍射图谱
    two_theta = np.linspace(two_theta_range[0], two_theta_range[1], num_points)
    intensity = np.zeros(num_points)
    
    # 简化的衍射强度计算（实际应用应使用专用库如PyXtal）
    # 基于布拉格定律和德拜公式的简化计算
    for i, angle in enumerate(two_theta):
        theta = np.radians(angle / 2)
        if np.sin(theta) == 0:
            continue
            
        # 计算晶面间距d
        d = wavelength / (2 * np.sin(theta))
        
        # 简化的衍射强度计算
        for j in range(len(atoms)):
            for k in range(j + 1, len(atoms)):
                r = np.linalg.norm(atoms.positions[j] - atoms.positions[k])
                if r > 0.1 and r < 10:  # 合理的原子间距范围
                    intensity[i] += np.sin(4 * np.pi * r * np.sin(theta) / wavelength) / r
    
    # 归一化强度
    intensity = np.abs(intensity)
    intensity /= np.max(intensity)
    
    # 检测衍射峰
    peaks, _ = find_peaks(intensity, height=0.1, distance=20)
    
    # 计算结晶部分面积（峰面积）
    crystal_area = 0
    for peak in peaks:
        # 取峰周围±5个点计算面积
        start = max(0, peak - 5)
        end = min(len(intensity) - 1, peak + 5)
        crystal_area += simpson(intensity[start:end], two_theta[start:end])
    
    # 计算非晶部分面积（总曲线下面积减去峰面积）
    total_area = simpson(intensity, two_theta)
    amorphous_area = total_area - crystal_area
    
    # 计算结晶度 [6,7](@ref)
    crystallinity = (crystal_area / total_area) * 100 if total_area > 0 else 0
    
    return crystallinity, two_theta, intensity, peaks

def analyze_carbon_structure(xyz_file):
    """
    分析碳结构：密度 + sp3比例 + 结晶度
    参数:
        xyz_file (str): .xyz 文件路径
    """
    # 碳原子质量 (单位: amu)
    CARBON_MASS = 12.0107
    
    # 使用自定义解析器读取文件
    atoms = parse_custom_xyz(xyz_file)
    
    # 获取晶格信息
    cell = atoms.get_cell()
    lattice_vectors = cell.array
    
    # 计算总质量
    num_atoms = len(atoms)
    total_mass = num_atoms * CARBON_MASS
    
    # 计算晶胞体积
    volume = atoms.get_volume()
    
    # 计算密度
    density = total_mass * 1.66053906660 / volume
    
    # 计算sp3比例
    sp3_ratio, hybridization_types, sp3_atom_indices = calculate_sp3_ratio(atoms)
    
    # 计算结晶度 [6,7](@ref)
    crystallinity, two_theta, intensity, peaks = calculate_crystallinity(atoms)
    
    return {
        "total_atoms": num_atoms,
        "total_mass_amu": total_mass,
        "volume_A3": volume,
        "density_g_per_cm3": density,
        "lattice_vectors": lattice_vectors,
        "sp3_ratio": sp3_ratio,
        "hybridization_types": hybridization_types,
        "sp3_atom_indices": sp3_atom_indices,
        "crystallinity": crystallinity,
        "xrd_data": (two_theta, intensity, peaks)  # 用于可视化
    }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方式: python carbon_analysis.py <filename.xyz>")
        print("示例: python carbon_analysis.py 1.xyz")
        sys.exit(1)
    
    filename = sys.argv[1]
    try:
        print(f"开始分析: {filename}")
        results = analyze_carbon_structure(filename)
        
        # 基础信息输出
        print(f"\n文件: {filename}")
        print(f"原子总数: {results['total_atoms']:,}")
        print(f"碳原子质量: 12.0107 amu × {results['total_atoms']:,} = {results['total_mass_amu']:.2f} amu")

        print(f"晶胞体积: {results['volume_A3']:.4f} Å³")
        print(f"\n密度: {results['density_g_per_cm3']:.6f} g/cm³")
        
        # sp3杂化信息
        sp3_count = len(results['sp3_atom_indices'])
        sp2_count = results['hybridization_types'].count('sp2')
        other_count = results['hybridization_types'].count('other')
        
        print("\n杂化类型统计:")
        print(f"  sp3 杂化原子数: {sp3_count} ({sp3_count/results['total_atoms']*100:.2f}%)")
        print(f"  sp2 杂化原子数: {sp2_count} ({sp2_count/results['total_atoms']*100:.2f}%)")
        print(f"  其他杂化原子数: {other_count} ({other_count/results['total_atoms']*100:.2f}%)")
        
        # 结晶度信息 [6,7](@ref)
        print(f"\n结晶度: {results['crystallinity']:.2f}%")
        print(f"XRD衍射峰数量: {len(results['xrd_data'][2])}")
        
        # 可选：输出XRD数据用于绘图
        with open(f"{os.path.splitext(filename)[0]}_xrd.dat", "w") as f:
            f.write("# 2Theta(deg) Intensity\n")
            for t, i in zip(results['xrd_data'][0], results['xrd_data'][1]):
                f.write(f"{t:.4f} {i:.6f}\n")
            print(f"\nXRD数据已保存到: {os.path.splitext(filename)[0]}_xrd.dat")
        
    except FileNotFoundError:
        print(f"错误: 文件 '{filename}' 未找到")
        sys.exit(1)
    except Exception as e:
        print(f"处理错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)