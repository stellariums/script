###导入一系列dump.xyz文件，并以.csv文件的格式输出某一帧密度，sp2占比，sp3占比以及sp2/sp3比例和结晶度
###使用方式：python analyze_dump.py <root_directory> <output.csv> -f 50
import numpy as np
import sys
import os
import csv
import argparse 
from ovito.io import import_file
from ovito.modifiers import CreateBondsModifier, CoordinationAnalysisModifier, IdentifyDiamondModifier

AVOGADRO = 6.022e23  # 阿伏伽德罗常数
CARBON_MASS = 12.0107  # 碳原子质量(g/mol)

def analyze_frame(file_path, frame_index):
    """
    分析指定帧的数据
    :param file_path: 轨迹文件路径
    :param frame_index: 要分析的帧索引
    :return: 分析结果字典
    """
    pipeline = import_file(file_path, multiple_frames=True)
    num_frames = pipeline.source.num_frames
    
    # 验证帧索引有效性
    if frame_index >= num_frames or frame_index < 0:
        if frame_index == -1:  # 最后一帧的特殊处理
            frame_index = num_frames - 1
        else:
            raise ValueError(f"帧索引 {frame_index} 超出范围 (0-{num_frames-1})")
    
    data = pipeline.compute(frame_index)
    volume = data.cell.volume  # Å³
    num_atoms = data.particles.count
    
    # 密度计算
    mass_grams = (num_atoms * CARBON_MASS) / AVOGADRO
    density = mass_grams / (volume * 1e-24)  # g/cm³
    
    # 键分析
    pipeline.modifiers.append(CreateBondsModifier(cutoff=1.85))
    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=1.85))
    data = pipeline.compute(frame_index)
    coord_numbers = data.particles['Coordination']
    
    # 杂化分析
    sp2_count = np.sum(coord_numbers == 3)
    sp3_count = np.sum(coord_numbers == 4)
    sp2_percent = (sp2_count / num_atoms) * 100
    sp3_percent = (sp3_count / num_atoms) * 100
    sp3_sp2_ratio = sp3_count / sp2_count if sp2_count > 0 else np.inf
    
    # 结晶度分析
    pipeline.modifiers.append(IdentifyDiamondModifier())
    data = pipeline.compute(frame_index)
    structure_types = data.particles['Structure Type']
    crystal_atoms = np.sum((structure_types >= 1) & (structure_types <= 6))
    crystallinity = (crystal_atoms / num_atoms) * 100

    return {
        "Density": density,
        "sp2 Atoms": sp2_percent,
        "sp3 Atoms": sp3_percent,
        "sp3/sp2 Ratio": sp3_sp2_ratio,
        "Crystallinity": crystallinity
    }

def process_directory(root_dir, output_csv, frame_index):
    """
    处理目录结构并输出到CSV文件
    :param root_dir: 根目录路径
    :param output_csv: 输出CSV文件路径
    :param frame_index: 要分析的帧索引
    """
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Folder', 'Frame Index', 'Density (g/cm3)', 'sp2 Atoms (%)', 
                     'sp3 Atoms (%)', 'sp3/sp2 Ratio', 'Crystallinity (%)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            
            if not os.path.isdir(folder_path):
                continue
                
            dump_path = os.path.join(folder_path, "dump.xyz")
            if not os.path.isfile(dump_path):
                print(f"警告: {folder_path} 中没有找到 dump.xyz 文件，跳过")
                continue
                
            try:
                print(f"\n处理文件夹: {folder_name} (帧索引: {frame_index})")
                results = analyze_frame(dump_path, frame_index)
                
                row_data = {
                    'Folder': folder_name,
                    'Frame Index': frame_index,
                    'Density (g/cm3)': results['Density'],
                    'sp2 Atoms (%)': results['sp2 Atoms'],
                    'sp3 Atoms (%)': results['sp3 Atoms'],
                    'sp3/sp2 Ratio': results['sp3/sp2 Ratio'],
                    'Crystallinity (%)': results['Crystallinity']
                }
                
                writer.writerow(row_data)
                print(f"成功写入: {folder_name}")
                
            except Exception as e:
                print(f"处理 {folder_name} 时出错: {str(e)}")
                continue

if __name__ == "__main__":
    # 使用argparse解析命令行参数[1,3](@ref)
    parser = argparse.ArgumentParser(description='分析LAMMPS dump文件的结构特性')
    parser.add_argument('root_dir', help='包含模拟子目录的根目录路径')
    parser.add_argument('output_csv', help='输出CSV文件的路径')
    parser.add_argument('-f', '--frame', type=int, default=-1, 
                        help='指定分析的帧索引(0-based)，默认-1表示最后一帧')
    
    args = parser.parse_args()
    root_dir = args.root_dir
    output_csv = args.output_csv
    frame_index = args.frame
    
    # 检查根目录是否存在
    if not os.path.isdir(root_dir):
        print(f"错误: 目录 '{root_dir}' 不存在")
        sys.exit(1)
    
    print(f"开始处理目录: {root_dir}")
    print(f"输出文件: {output_csv}")
    print(f"分析帧索引: {'最后一帧' if frame_index == -1 else frame_index}")
    
    process_directory(root_dir, output_csv, frame_index)
    
    print("\n处理完成!")
    print(f"结果已保存到: {output_csv}")