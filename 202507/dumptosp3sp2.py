###对导入单个的dump.xyz文件选取特定帧数分析，输出密度，sp2占比，sp3占比,sp2/sp3，以及结晶度
###使用方式：python analyze_dump.py dump.xyz -f -2
import numpy as np
import sys
import argparse
from ovito.io import import_file
from ovito.modifiers import CreateBondsModifier, CoordinationAnalysisModifier, IdentifyDiamondModifier

def analyze_frame(file_path, frame_index):
    """
    分析GPUMD生成的dump.xyz文件的指定帧
    """
    # 导入文件并获取总帧数
    pipeline = import_file(file_path, multiple_frames=True)
    num_frames = pipeline.source.num_frames
    
    # 处理负数索引（-1表示最后一帧）
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
    
    return {
        "Density (g/cm³)": density,
        "sp² Atoms (%)": sp2_percent,
        "sp³ Atoms (%)": sp3_percent,
        "sp³/sp² Ratio": sp3_sp2_ratio,
        "Crystallinity (%)": crystallinity
    }

if __name__ == "__main__":
    # 配置命令行参数解析[6,7,8](@ref)
    parser = argparse.ArgumentParser(description="分析GPUMD生成的dump.xyz文件")
    parser.add_argument("dump_file", help="输入文件路径（dump.xyz）")
    parser.add_argument("-f", "--frame", type=int, default=-1,
                        help="分析指定帧（0-based索引，负数表示倒数，默认：-1=最后一帧）")
    args = parser.parse_args()
    
    try:
        # 分析指定帧
        results = analyze_frame(args.dump_file, args.frame)
        
        # 打印结果
        print(f"\n系统属性分析（第 {args.frame} 帧）:")
        print("-" * 60)
        for prop, value in results.items():
            if "%" in prop:
                print(f"{prop}: {value:.4f}")
            elif "Ratio" in prop:
                print(f"{prop}: {'Infinity (no sp² atoms)' if value == np.inf else f'{value:.4f}'}")
            else:
                print(f"{prop}: {value:.6f}")
    
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)