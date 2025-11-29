"""
批量结构转换脚本
将 structure 目录下的所有 VASP 格式 (CONTCAR) 文件批量转换为扩展 XYZ 格式 (extxyz)，
并将结果保存在 model 目录中。

依赖库:
    - ase (Atomic Simulation Environment)
    - pathlib
"""
from pathlib import Path
from ase.io import read, write
import sys

def main():
    """
    主函数：执行批量转换逻辑。
    1. 检查 structure 目录是否存在。
    2. 创建 model 输出目录。
    3. 遍历并转换所有 CONTCAR 文件。
    """
    # 定义结构目录和输出目录
    structure_dir = Path("structure")
    output_dir = Path("model")
    
    # 检查源目录是否存在
    if not structure_dir.exists():
        print(f"错误: 目录 '{structure_dir}' 不存在。")
        return

    # 如果输出目录不存在，则创建它
    output_dir.mkdir(exist_ok=True)

    # 在 structure 目录中查找文件名包含 'CONTCAR' 的所有文件
    files = [f for f in structure_dir.iterdir() if f.is_file() and 'CONTCAR' in f.name]
    
    if not files:
        print("在 structure 目录中未找到包含 'CONTCAR' 的文件。")
        return

    print(f"找到 {len(files)} 个文件待转换。输出目录: {output_dir}")
    
    success_count = 0
    for file_path in files:
        try:
            # 读取结构 (VASP 格式)
            atoms = read(file_path, format="vasp")
            
            # 移除动量信息 (如果存在)，这通常是转换到 XYZ 时的常见操作
            atoms.arrays.pop("momenta", None)
            
            # 定义输出文件名: 保持原文件名，后缀改为 .xyz，放置在 model 目录下
            output_path = output_dir / f"{file_path.name}.xyz"
            
            # 写入 XYZ 格式
            write(output_path, atoms, format="extxyz")
            print(f"已转换: {file_path.name} -> {output_path}")
            success_count += 1
            
        except Exception as e:
            print(f"转换失败 {file_path.name}: {e}")

    print(f"\n批量转换完成。成功转换 {success_count}/{len(files)} 个文件。")

if __name__ == "__main__":
    main()
