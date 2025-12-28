import numpy as np
import os
import csv
from ovito.io import import_file
from ovito.modifiers import CreateBondsModifier, CoordinationAnalysisModifier, IdentifyDiamondModifier

def analyze_xyz_file(file_path):
    """分析单个XYZ文件并返回计算结果"""
    try:
        pipeline = import_file(file_path)
        pipeline.modifiers.append(CreateBondsModifier(cutoff=1.85))
        data = pipeline.compute()
        
        # 基础属性计算
        volume = data.cell.volume
        num_atoms = data.particles.count
        actual_mass_per_atom = 12.01 / 6.022e23
        density = (num_atoms * actual_mass_per_atom * 1e24) / volume

        # 配位数分析
        pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=1.85))
        data = pipeline.compute()
        coord_numbers = data.particles['Coordination']
        sp2_count = np.sum(coord_numbers == 3)
        sp3_count = np.sum(coord_numbers == 4)
        sp2_percent = sp2_count / num_atoms * 100
        sp3_percent = sp3_count / num_atoms * 100
        sp3_sp2_ratio = sp3_percent / sp2_percent if sp2_percent > 0 else float('nan')

        # 结晶度分析 (类型1-6的总和)
        pipeline.modifiers.append(IdentifyDiamondModifier())
        data = pipeline.compute()
        structure_types = data.particles['Structure Type']
        crystal_atoms = np.sum((structure_types >= 1) & (structure_types <= 6))
        crystallinity = crystal_atoms / num_atoms * 100

        return {
            "File": file_path,
            "Density (g/cm3)": density,
            "sp2 Percentage (%)": sp2_percent,
            "sp3 Percentage (%)": sp3_percent,
            "sp3/sp2 Ratio": sp3_sp2_ratio,
            "Crystallinity (%)": crystallinity
        }
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")

    return {
        "File": file_path,
        "Density (g/cm3)": density,              # ³ → 3
        "sp2 Percentage (%)": sp2_percent,        # ² → 2
        "sp3 Percentage (%)": sp3_percent,       # ³ → 3
        "sp3/sp2 Ratio": sp3_sp2_ratio,           # ³/² → 3/2
        "Crystallinity (%)": crystallinity
}   

def batch_process_folder(root_folder, output_csv):
    """批量处理多层文件夹结构并保存结果到CSV"""
    results = []
    
    # 遍历根目录下的所有一级子文件夹
    for level1_dir in os.listdir(root_folder):
        level1_path = os.path.join(root_folder, level1_dir)
        if not os.path.isdir(level1_path):
            continue
            
        print(f"处理一级文件夹: {level1_path}")
        
        # 遍历一级子文件夹下的二级子文件夹
        for level2_dir in os.listdir(level1_path):
            level2_path = os.path.join(level1_path, level2_dir)
            if not os.path.isdir(level2_path):
                continue
                
            print(f"  扫描二级文件夹: {level2_dir}")
            
            # 在二级子文件夹中查找XYZ文件
            for filename in os.listdir(level2_path):
                if filename.endswith(".xyz"):
                    file_path = os.path.join(level2_path, filename)
                    print(f"    分析文件: {filename}")
                    result = analyze_xyz_file(file_path)
                    results.append(result)
    
    # 保存结果到CSV
    if results:
        with open(output_csv, 'w', newline='',encoding='utf-8') as csvfile:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n处理完成! 共分析 {len(results)} 个文件，结果已保存至: {output_csv}")
    else:
        print("未找到任何XYZ文件!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='批量分析多层文件夹中的XYZ文件')
    parser.add_argument('root_folder', type=str, help='包含多级子文件夹的根目录路径')
    parser.add_argument('output_csv', type=str, help='输出CSV文件的路径')
    
    args = parser.parse_args()
    
    batch_process_folder(args.root_folder, args.output_csv)