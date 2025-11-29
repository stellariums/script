"""
单文件结构转换脚本
将 VASP 格式 (CONTCAR) 转换为扩展 XYZ 格式 (extxyz)。

用法:
    python convert_to_xyz.py [--in 输入文件] [--out 输出文件]

默认值:
    输入: GWHR-CONTCAR
    输出: model.xyz
"""
import argparse
from pathlib import Path
from ase.io import read, write

def main():
    """
    主函数：解析命令行参数并执行转换。
    """
    # 初始化参数解析器
    parser = argparse.ArgumentParser(description="将 VASP CONTCAR 文件转换为 XYZ 格式")
    # 添加输入文件参数
    parser.add_argument("--in", dest="infile", default="GWHR-CONTCAR", help="输入 VASP CONTCAR 文件路径")
    # 添加输出文件参数
    parser.add_argument("--out", dest="outfile", default="model.xyz", help="输出 XYZ 文件路径")
    
    args = parser.parse_args()
    
    in_path = Path(args.infile)
    out_path = Path(args.outfile)
    
    # 读取 VASP 格式文件
    print(f"正在读取: {in_path}")
    try:
        atoms = read(in_path, format="vasp")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {in_path}")
        return
        
    # 移除动量信息 (如果存在)，因为 XYZ 格式通常只需要坐标和力
    atoms.arrays.pop("momenta", None)
    
    # 写入扩展 XYZ 格式
    write(out_path, atoms, format="extxyz")
    print(f"已转换并保存至: {out_path}")

if __name__ == "__main__":
    main()