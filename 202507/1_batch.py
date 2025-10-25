import numpy as np
from scipy.integrate import simpson
from ovito.io import import_file
from ovito.modifiers import CoordinationAnalysisModifier
import os
import sys
import csv
import argparse
import re

def compute_s_ex(r, g, rho, r_cut):
    """计算过剩熵，自动处理 r=0 处的积分奇点"""
    mask = (r > 0) & (r <= r_cut)
    r_sub, g_sub = r[mask], g[mask]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        g_safe = np.where(g_sub > 0, g_sub, 1.0)
        integrand = g_safe * np.log(g_safe) - (g_safe - 1)
        integrand = np.where(g_sub > 0, integrand, 0.0)
    
    integral = simpson(integrand * r_sub**2, r_sub) * 2 * np.pi
    return -rho * integral

def process_file(file_path, r_cut=10.0):
    try:
        pipeline = import_file(file_path)
        modifier = CoordinationAnalysisModifier(
            number_of_bins=500,
            cutoff=12.0,
            partial=False
        )
        pipeline.modifiers.append(modifier)
        data = pipeline.compute(0)

        table = data.tables["coordination-rdf"]
        
        if "Bin Center" in table.keys():
            r_array = table["Bin Center"].array
        elif "bin_center" in table.keys():
            r_array = table["bin_center"].array
        else:
            r_array = np.linspace(0, modifier.cutoff, modifier.number_of_bins)
        
        g_array = table["g(r)"].array if "g(r)" in table.keys() else table["RDF"].array

        rho = data.particles.count / data.cell.volume
        return compute_s_ex(r_array, g_array, rho, r_cut)
        
    except Exception as e:
        print(f"处理 {os.path.basename(file_path)} 出错: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='计算目录树的过剩熵并导出到CSV')
    parser.add_argument('input_path', help='输入根目录路径')
    parser.add_argument('-o', '--output', default='s_ex_results.csv', help='输出CSV文件名')
    parser.add_argument('--r_cut', type=float, default=10.0, help='截断半径(Å)')
    args = parser.parse_args()

    abs_output = os.path.abspath(args.output)
    print(f"输出文件将保存至: {abs_output}")

    # === 修改为处理 compress_X/warmup_Y/model.xyz 目录结构 ===
    results = []
    for root, dirs, files in os.walk(args.input_path):
        # 从路径中提取压缩率和warmup索引
        parts = os.path.normpath(root).split(os.sep)
        compress_val, warmup_val = None, None
        
        # 检查路径是否包含compress和warmup目录
        if "compress_" in root and "warmup_" in root:
            for part in parts:
                if part.startswith("compress_"):
                    try:
                        compress_val = int(part.split("_")[1])
                    except (IndexError, ValueError):
                        pass
                elif part.startswith("warmup_"):
                    try:
                        warmup_val = int(part.split("_")[1])
                    except (IndexError, ValueError):
                        pass
        
        if compress_val is None or warmup_val is None:
            continue  # 跳过不包含压缩率/Warmup索引的目录
            
        for filename in files:
            if filename == "model.xyz":
                file_path = os.path.join(root, filename)
                print(f"处理: {os.path.relpath(file_path, args.input_path)} (压缩率={compress_val}, Warmup={warmup_val})")
                s_ex = process_file(file_path, args.r_cut)
                if s_ex is not None:
                    results.append((file_path, compress_val, warmup_val, s_ex))
                    print(f"  过剩熵: {s_ex:.6f} k_B")
                else:
                    print(f"  × 未获取有效结果")
    
    if not results:
        print(f"错误: 在 {args.input_path} 中未找到有效的 model.xyz 文件")
        sys.exit(1)
    
    # === 按压缩率和warmup索引排序结果 ===
    results.sort(key=lambda x: (x[1], x[2]))

    try:
        with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['文件路径', '压缩率', 'Warmup索引', '过剩熵(k_B)'])
            for file_path, compress_val, warmup_val, s_ex in results:
                rel_path = os.path.relpath(file_path, args.input_path)
                writer.writerow([rel_path, compress_val, warmup_val, f"{s_ex:.6f}"])
            print(f"\n成功写入 {len(results)} 条数据到 {args.output}")
    except PermissionError:
        print(f"错误：无权限写入 {args.output}，请关闭已打开的CSV文件")
    except OSError as e:
        print(f"文件系统错误: {str(e)}")
        print("建议：尝试指定其他输出路径，如 D:/results.csv")

if __name__ == "__main__":
    main()